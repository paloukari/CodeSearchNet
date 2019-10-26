from collections import Counter
import numpy as np
from typing import Dict, Set, Any, List, Iterable, Optional, Tuple
import random
import re
from collections import defaultdict
from abc import ABC, abstractmethod

from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import convert_and_pad_token_sequence, get_activation, write_to_feed_dict

import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts, get_language_keywords
from dpu_utils.mlutils import Vocabulary

from .encoder import Encoder, QueryType

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
# "0" is PAD, "1" is UNK
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1
USES_SUBTOKEN_EDGE_NAME = "UsesSubtoken"
SELF_LOOP_EDGE_NAME = "SelfLoop"
BACKWARD_EDGE_TYPE_NAME_SUFFIX = "_Bkwd"
__PROGRAM_GRAPH_EDGES_TYPES = ["Child", "NextToken", "LastUse", "LastWrite", "LastLexicalUse", "ComputedFrom",
                               "GuardedByNegation", "GuardedBy", "FormalArgName", "ReturnsTo", USES_SUBTOKEN_EDGE_NAME]
__PROGRAM_GRAPH_EDGES_TYPES_WITH_BKWD = \
    __PROGRAM_GRAPH_EDGES_TYPES + [edge_type_name + BACKWARD_EDGE_TYPE_NAME_SUFFIX
                                   for edge_type_name in __PROGRAM_GRAPH_EDGES_TYPES]
PROGRAM_GRAPH_EDGES_TYPES_VOCAB = {edge_type_name: idx
                                   for idx, edge_type_name in enumerate(__PROGRAM_GRAPH_EDGES_TYPES_WITH_BKWD)}


class GraphEncoder(Encoder):
    
    unsplittable_keywords = None

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'max_nodes_in_batch': 50000,

            'graph_num_layers': 8,
            'graph_num_timesteps_per_layer': 1,

            'graph_layer_input_dropout_keep_prob': 0.8,
            'graph_dense_between_every_num_gnn_layers': 1,
            'graph_model_activation_function': 'tanh',
            'graph_residual_connection_every_num_layers': 2,
            'graph_inter_layer_norm': False,

            'max_epochs': 10000,
            'patience': 25,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'learning_rate_decay': 0.98,
            # The LR is normalised so that we use it for exactly that number of graphs; no normalisation happens if the value is None
            'lr_for_num_graphs_per_batch': None,
            'momentum': 0.85,
            'clamp_gradient_norm': 1.0,
            'random_seed': 0,

            # SeqEncoder
            'token_vocab_size': 10000,
            'token_vocab_count_threshold': 10,
            'token_embedding_size': 128,

            'use_subtokens': False,
            'mark_subtoken_end': False,

            'max_num_tokens': 200,

            'use_bpe': True,
            'pct_bpe': 0.5,
            # VarMisuse
            'max_variable_candidates': 5,
            'graph_node_label_max_num_chars': 19,
            'graph_node_label_representation_size': 64,
            'slot_score_via_linear_layer': True,
            'loss_function': 'max-likelihood',  # max-likelihood or max-margin
            'max-margin_loss_margin': 0.2,
            'out_layer_dropout_rate': 0.2,
            'add_self_loop_edges': False,

        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

        # If required, add the self-loop edge type to the vocab:
        if hyperparameters.get('code_add_self_loop_edges'):
            if SELF_LOOP_EDGE_NAME not in PROGRAM_GRAPH_EDGES_TYPES_VOCAB:
                PROGRAM_GRAPH_EDGES_TYPES_VOCAB[SELF_LOOP_EDGE_NAME] = \
                    len(PROGRAM_GRAPH_EDGES_TYPES_VOCAB)

    def _make_placeholders(self):
        """
        Creates placeholders "num_graphs" and "graph_layer_input_dropout_keep_prob" for graph encoders.
        """
        super()._make_placeholders()
        self.placeholders['num_graphs'] = \
            tf.placeholder(dtype=tf.int64, shape=[], name='num_graphs')
        self.placeholders['graph_layer_input_dropout_keep_prob'] = \
            tf.placeholder_with_default(
                1.0, shape=[], name='graph_layer_input_dropout_keep_prob')

    @property
    def num_edge_types(self) -> int:
        return len(PROGRAM_GRAPH_EDGES_TYPES_VOCAB)

    @property
    def initial_node_feature_size(self) -> int:
        return self.params['graph_node_label_representation_size']

    @property
    def output_representation_size(self):
        return self.get_hyper('hidden_size')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("graph_encoder"):
            self._make_placeholders()
            model = self.__make_input_model()
            model = self.__build_graph_propagation_model(model)
            return model

    def __build_graph_propagation_model(self, model: tf.Tensor) -> tf.Tensor:
        _propagation_model = None

        h_dim = self.get_hyper('hidden_size')
        activation_fn = get_activation(
            self.get_hyper('graph_model_activation_function'))
        if self.get_hyper('graph_node_label_representation_size') != self.get_hyper('hidden_size'):
            _propagation_model = tf.keras.layers.Dense(units=h_dim,
                                                       use_bias=False,
                                                       activation=activation_fn,
                                                       )(model)
        else:
            _propagation_model = model

        cur_node_representations = _propagation_model
        last_residual_representations = tf.zeros_like(cur_node_representations)
        for layer_idx in range(self.get_hyper('graph_num_layers')):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                cur_node_representations = \
                    tf.nn.dropout(cur_node_representations,
                                  keep_prob=self.placeholders['graph_layer_input_dropout_keep_prob'])
                if layer_idx % self.get_hyper('graph_residual_connection_every_num_layers') == 0:
                    t = cur_node_representations
                    if layer_idx > 0:
                        cur_node_representations += last_residual_representations
                        cur_node_representations /= 2
                    last_residual_representations = t
                cur_node_representations = \
                    self._apply_gnn_layer(
                        cur_node_representations,
                        self.placeholders['adjacency_lists'],
                        self.placeholders['type_to_num_incoming_edges'],
                        self.get_hyper('graph_num_timesteps_per_layer'))
                if self.get_hyper('graph_inter_layer_norm'):
                    cur_node_representations = tf.contrib.layers.layer_norm(
                        cur_node_representations)
                if layer_idx % self.get_hyper('graph_dense_between_every_num_gnn_layers') == 0:
                    cur_node_representations = \
                        tf.keras.layers.Dense(units=h_dim,
                                              use_bias=False,
                                              activation=activation_fn,
                                              name="Dense",
                                              )(cur_node_representations)

        return cur_node_representations

    @abstractmethod
    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: Dict[int, tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        """
        Run a GNN layer on a graph.

        Arguments:
            node_features: float32 tensor of shape [V, D], where V is the number of nodes.
            adjacency_lists: list of L int32 tensors of shape [E, 2], where L is the number
                of edge types and E the number of edges of that type.
                Hence, adjacency_lists[l][e,:] == [u, v] means that u has an edge of type l
                to v.
            type_to_num_incoming_edges: int32 tensor of shape [L, V], where L is the number
                of edge types.
                type_to_num_incoming_edges[l, v] = k indicates that node v has k incoming
                edges of type l.
            num_timesteps: Number of propagation steps in to run in this GNN layer.
        """
        raise Exception("Models have to implement _apply_gnn_layer!")

    def __get_node_label_charcnn_embeddings(self,
                                            unique_labels_as_characters: tf.Tensor,
                                            node_labels_to_unique_labels: tf.Tensor,
                                            ) -> tf.Tensor:
        """
        Compute representation of node labels using a 2-layer character CNN.

        Args:
            unique_labels_as_characters: int32 tensor of shape [U, C]
                representing the unique (node) labels occurring in a
                batch, where U is the number of such labels and C the
                maximal number of characters.
            node_labels_to_unique_labels: int32 tensor of shape [V],
                mapping each node in the batch to one of the unique
                labels.

        Returns:
            float32 tensor of shape [V, D] representing embedded node
            label information about each node.
        """
        label_embedding_size = self.get_hyper(
            'graph_node_label_representation_size')  # D
        # U ~ num unique labels
        # C ~ num characters (self.get_hyper('graph_node_label_max_num_chars'))
        # A ~ num characters in alphabet
        unique_label_chars_one_hot = tf.one_hot(indices=unique_labels_as_characters,
                                                depth=len(ALPHABET),
                                                axis=-1)  # Shape: [U, C, A]

        # Choose kernel sizes such that there is a single value at the end:
        char_conv_l1_kernel_size = 5
        char_conv_l2_kernel_size = \
            self.get_hyper('graph_node_label_max_num_chars') - \
            2 * (char_conv_l1_kernel_size - 1)

        char_conv_l1 = \
            tf.keras.layers.Conv1D(filters=16,
                                   kernel_size=char_conv_l1_kernel_size,
                                   activation=tf.nn.leaky_relu,
                                   )(unique_label_chars_one_hot)  # Shape: [U, C - (char_conv_l1_kernel_size - 1), 16]
        char_pool_l1 = \
            tf.keras.layers.MaxPool1D(pool_size=char_conv_l1_kernel_size,
                                      strides=1,
                                      )(inputs=char_conv_l1)      # Shape: [U, C - 2*(char_conv_l1_kernel_size - 1), 16]
        char_conv_l2 = \
            tf.keras.layers.Conv1D(filters=label_embedding_size,
                                   kernel_size=char_conv_l2_kernel_size,
                                   activation=tf.nn.leaky_relu,
                                   )(char_pool_l1)                # Shape: [U, 1, D]
        unique_label_representations = tf.squeeze(
            char_conv_l2, axis=1)  # Shape: [U, D]
        node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=node_labels_to_unique_labels)
        return node_label_representations

    def __make_input_model(self) -> None:
        node_label_char_length = self.get_hyper(
            'graph_node_label_max_num_chars')
        self.placeholders['unique_labels_as_characters'] = \
            tf.placeholder(dtype=tf.int32, shape=[
                           None, node_label_char_length], name='unique_labels_as_characters')
        self.placeholders['node_labels_to_unique_labels'] = \
            tf.placeholder(dtype=tf.int32, shape=[
                           None], name='node_labels_to_unique_labels')
        self.placeholders['adjacency_lists'] = \
            dict((e, tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e))
                for e in range(self.num_edge_types))
        self.placeholders['type_to_num_incoming_edges'] = \
            tf.placeholder(dtype=tf.float32, shape=[
                           self.num_edge_types, None], name='type_to_num_incoming_edges')

        self.placeholders['initial_node_features'] = \
            self.__get_node_label_charcnn_embeddings(self.placeholders['unique_labels_as_characters'],
                                                     self.placeholders['node_labels_to_unique_labels'])

        return self.placeholders['initial_node_features']

        # self.placeholders['initial_node_features'] = \
        #     tf.placeholder(dtype=tf.float32, shape=[None, self.initial_node_feature_size], name='initial_node_features')
        # self.placeholders['adjacency_lists'] = \
        #     [tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
        #         for e in range(self.num_edge_types)]
        # self.placeholders['type_to_num_incoming_edges'] = \
        #     tf.placeholder(dtype=tf.float32, shape=[self.num_edge_types, None], name='type_to_num_incoming_edges')

        # model_ops['initial_node_features'] = self.placeholders['initial_node_features']
        # model_ops['adjacency_lists'] = self.placeholders['adjacency_lists']
        # model_ops['type_to_num_incoming_edges'] = self.placeholders['type_to_num_incoming_edges']

        # node_label_char_length = self.get_hyper('graph_node_label_max_num_chars')
        # self.placeholders['unique_labels_as_characters'] = \
        #     tf.placeholder(dtype=tf.int32, shape=[
        #                    None, node_label_char_length], name='unique_labels_as_characters')
        # self.placeholders['node_labels_to_unique_labels'] = \
        #     tf.placeholder(dtype=tf.int32, shape=[
        #                    None], name='node_labels_to_unique_labels')
        # self.placeholders['adjacency_lists'] = \
        #     [tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
        #         for e in range(self.num_edge_types)]
        # self.placeholders['type_to_num_incoming_edges'] = \
        #     tf.placeholder(dtype=tf.float32, shape=[
        #                    self.num_edge_types, None], name='type_to_num_incoming_edges')

        # return self.__get_node_label_charcnn_embeddings(self.placeholders['unique_labels_as_characters'],
        #                                              self.placeholders['node_labels_to_unique_labels'])
        # self.ops['adjacency_lists'] = self.placeholders['adjacency_lists']
        # self.ops['type_to_num_incoming_edges'] = self.placeholders['type_to_num_incoming_edges']

    # TODO: remove/refactor this
    # def embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
    #     # TODO: this meeds to change/go
    #     """
    #     Creates embedding layer that is in common between many encoders.

    #     Args:
    #         token_inp:  2D tensor that is of shape (batch size, sequence length)

    #     Returns:
    #         3D tensor of shape (batch size, sequence length, embedding dimension)
    #     """

    #     token_embeddings = tf.get_variable(name='token_embeddings',
    #                                        initializer=tf.glorot_uniform_initializer(),
    #                                        shape=[len(self.metadata['token_vocab']),
    #                                               self.get_hyper('token_embedding_size')],
    #                                        )
    #     self.__embeddings = token_embeddings

    #     token_embeddings = tf.nn.dropout(token_embeddings,
    #                                      keep_prob=self.placeholders['dropout_keep_rate'])

    #     return tf.nn.embedding_lookup(hypers=token_embeddings, ids=token_inp)

    @classmethod
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata['token_counter'] = Counter()
        return raw_metadata

    @classmethod
    def _to_subtoken_stream(cls, input_stream: Iterable[str], mark_subtoken_end: bool) -> Iterable[str]:
        for token in input_stream:
            if SeqEncoder.IDENTIFIER_TOKEN_REGEX.match(token):
                yield from split_identifier_into_parts(token)
                if mark_subtoken_end:
                    yield '</id>'
            else:
                yield token

    @classmethod
    def load_metadata_from_sample(cls, sample: Dict[str, Any], data_to_load: Iterable[str], raw_metadata: Dict[str, Any],
                                  use_subtokens: bool = False, mark_subtoken_end: bool = False) -> None:

        if use_subtokens:
            data_to_load = cls._to_subtoken_stream(
                data_to_load, mark_subtoken_end=mark_subtoken_end)
        raw_metadata['token_counter'].update(data_to_load)

        # TODO: validate the list of unsplitted tokens

    @classmethod
    def finalise_metadata(cls, encoder_label: str, language: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        #TODO: move this to init
        if not cls.unsplittable_keywords: 
            cls.unsplittable_keywords = get_language_keywords(language)            

        final_metadata = super().finalise_metadata(
            encoder_label, language, hyperparameters, raw_metadata_list)
        merged_token_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']

        #graph_node_labels = sample['graph_V']
        for token in list(merged_token_counter):
            if token in cls.unsplittable_keywords:
                del merged_token_counter[token]

        if hyperparameters['%s_use_bpe' % encoder_label]:
            token_vocabulary = BpeVocabulary(vocab_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                             pct_bpe=hyperparameters['%s_pct_bpe' %
                                                                     encoder_label]
                                             )
            token_vocabulary.fit(merged_token_counter)
        else:
            token_vocabulary = Vocabulary.create_vocabulary(tokens=merged_token_counter,
                                                            max_size=hyperparameters['%s_token_vocab_size' %
                                                                                     encoder_label],
                                                            count_threshold=hyperparameters['%s_token_vocab_count_threshold' % encoder_label])

        # TODO: update the edges based on the calculated tokens
        final_metadata['token_vocab'] = token_vocabulary
        # Save the most common tokens for use in data augmentation:
        final_metadata['common_tokens'] = merged_token_counter.most_common(50)
        return final_metadata

    @classmethod
    def _add_per_subtoken_nodes(cls, graph_dict: Dict[str, Any]) -> None:
        
        graph_node_labels = graph_dict['graph_V']
        subtoken_to_using_nodes = defaultdict(set)

        max_used_node_id = 0
        for node_id, node_label in enumerate(graph_node_labels):
            node_id = int(node_id)
            max_used_node_id = max(node_id, max_used_node_id)

            # Skip AST nodes and punctuation:
            if node_label in cls.unsplittable_keywords:
                continue

            for subtoken in split_identifier_into_parts(node_label):
                if re.search('[a-zA-Z0-9]', subtoken):
                    subtoken_to_using_nodes[subtoken].add(node_id)

        subtoken_node_id = max_used_node_id
        new_edges = []
        for subtoken, using_nodes in subtoken_to_using_nodes.items():
            subtoken_node_id += 1
            graph_node_labels.extend(subtoken)
            graph_dict['graph_E'].extend([[USES_SUBTOKEN_EDGE_NAME, using_node_id, subtoken_node_id]
                               for using_node_id in using_nodes])
     
        return

    @classmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
                              raw_sample: Dict[str, Any],
                              metadata: Dict[str, Any],
                              data_to_load: Any,
                              function_name: Optional[str],
                              result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:
        """
        Saves two versions of both the code and the query: one using the docstring as the query and the other using the
        function-name as the query, and replacing the function name in the code with an out-of-vocab token.
        Sub-tokenizes, converts, and pads both versions, and rejects empty samples.
        """
        # Save the two versions of the code and query:
        data_holder = {QueryType.DOCSTRING.value: data_to_load,
                       QueryType.FUNCTION_NAME.value: None}
        # Skip samples where the function name is very short, because it probably has too little information
        # to be a good search query.
        if not is_test and hyperparameters['fraction_using_func_name'] > 0. and function_name and \
                len(function_name) >= hyperparameters['min_len_func_name_for_query']:
            # In the code, replace the function name with the out-of-vocab token everywhere it appears:
            data_holder[QueryType.FUNCTION_NAME.value] = [Vocabulary.get_unk() if token == function_name else token
                                                              for token in data_to_load]

        cls._add_per_subtoken_nodes(raw_sample)
        num_nodes = len(raw_sample['graph_V'])

        graph_node_label_max_num_chars = hyperparameters[f'{encoder_label}_graph_node_label_max_num_chars']
        node_label_chars = np.zeros(shape=(num_nodes, graph_node_label_max_num_chars),
                                    dtype=np.uint8)
        for (node, label) in enumerate(raw_sample['graph_V']):
            for (char_idx, label_char) in enumerate(label[:graph_node_label_max_num_chars].lower()):
                node_label_chars[int(node), char_idx] = ALPHABET_DICT.get(
                    label_char, 1)
        node_label_chars_unique, node_label_chars_indices = np.unique(node_label_chars,
                                                                        axis=0,
                                                                        return_inverse=True)

        # Split edges according to edge_type and count their numbers:
        num_edge_types = len(PROGRAM_GRAPH_EDGES_TYPES_VOCAB)
        adjacency_lists = dict((_, np.zeros((0, 2), dtype=np.int32))
                            for _ in range(num_edge_types))
        num_incoming_edges_per_type = np.zeros(
            (num_edge_types, num_nodes), dtype=np.uint16)
        
        # convert adjacency list to dense representation 
        raw_edges: DefaultDict[str, List] = defaultdict(list)
        for raw_edge in raw_sample['graph_E']:
            e_type = raw_edge[0]
            if e_type == 'child':
                e_type = 'Child'
            e_type_edge = [raw_edge[1], raw_edge[2]]
            raw_edges[e_type].append(e_type_edge)

        for raw_edge, e_type_edges in raw_edges.items():
            if len(e_type_edges) > 0:
                e_type_bkwd = e_type + BACKWARD_EDGE_TYPE_NAME_SUFFIX
                e_type_idx = PROGRAM_GRAPH_EDGES_TYPES_VOCAB[e_type]
                e_type_bkwd_idx = PROGRAM_GRAPH_EDGES_TYPES_VOCAB[e_type_bkwd]

                fwd_edges = np.array(e_type_edges, dtype=np.int32)
                bkwd_edges = np.flip(fwd_edges, axis=1)

                adjacency_lists[e_type_idx] = fwd_edges
                adjacency_lists[e_type_bkwd_idx] = bkwd_edges
                num_incoming_edges_per_type[e_type_idx, :] = \
                    np.bincount(
                        adjacency_lists[e_type_idx][:, 1], minlength=num_nodes)
                num_incoming_edges_per_type[e_type_bkwd_idx, :] = \
                    np.bincount(
                        adjacency_lists[e_type_bkwd_idx][:, 1], minlength=num_nodes)

        if hyperparameters.get('code_add_self_loop_edges'):
            self_loop_edge_type_idx = PROGRAM_GRAPH_EDGES_TYPES_VOCAB[SELF_LOOP_EDGE_NAME]
            adjacency_lists[self_loop_edge_type_idx] = \
                np.stack(
                    [np.arange(num_nodes), np.arange(num_nodes)], axis=1)
            num_incoming_edges_per_type[self_loop_edge_type_idx, :] = \
                np.ones(shape=(num_nodes,))
        
        # It's always encoder_label == 'code'
        result_holder[f'code_adjacency_lists']=adjacency_lists
        result_holder[f'code_type_to_num_incoming_edges']=num_incoming_edges_per_type
        result_holder[f'code_unique_labels_as_characters']=node_label_chars_unique
        result_holder[f'code_node_labels_to_unique_labels']=node_label_chars_indices

        return True

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        """
        Implements various forms of data augmentation.
        """
        # Graph structure:
        for i in range(self.num_edge_types):
            batch_data['code_adjacency_lists'][i].append(sample['code_adjacency_lists'][i] + batch_data['code_node_offset'])
        batch_data['code_type_to_num_incoming_edges'].append(sample['code_type_to_num_incoming_edges'])

        # Node labels:
        batch_data['code_unique_labels_as_characters'].append(sample['code_unique_labels_as_characters'])
        batch_data['code_node_labels_to_unique_labels'].append(
            sample['code_node_labels_to_unique_labels'] + batch_data['code_unique_label_offset'])
        batch_data['code_unique_label_offset'] += sample['code_unique_labels_as_characters'].shape[0]

        # Finally, update the offset we use to shift things during batch construction:
        batch_data['code_num_graphs'] += 1
        batch_data['code_node_offset'] += len(sample['code_node_labels_to_unique_labels'])

        if batch_data['code_node_offset'] >= self.get_hyper('max_nodes_in_batch'):
            return True
        return False

    # TODO: remove/refactor this
    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return (self.__embeddings,
                list(self.metadata['token_vocab'].id_to_token))

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['code_num_graphs'] = 0
        batch_data['code_node_offset'] = 0
        batch_data['code_unique_label_offset'] = 0
        batch_data['code_adjacency_lists'] = dict((_, []) for _ in range(self.num_edge_types))
        batch_data['code_type_to_num_incoming_edges'] = []
        batch_data['code_unique_labels_as_characters'] = []
        batch_data['code_node_labels_to_unique_labels'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)

        write_to_feed_dict(
            feed_dict, self.placeholders['type_to_num_incoming_edges'], batch_data['code_type_to_num_incoming_edges'])
        write_to_feed_dict(
            feed_dict, self.placeholders['adjacency_lists'], batch_data['code_adjacency_lists'])
        write_to_feed_dict(
            feed_dict, self.placeholders['unique_labels_as_characters'], batch_data['code_unique_labels_as_characters'])
        write_to_feed_dict(
            feed_dict, self.placeholders['node_labels_to_unique_labels'], batch_data['code_node_labels_to_unique_labels'])
