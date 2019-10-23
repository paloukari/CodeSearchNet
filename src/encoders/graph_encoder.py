from collections import Counter
import numpy as np
from typing import Dict, Any, List, Iterable, Optional, Tuple
import random
import re
from abc import ABC, abstractmethod

from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import convert_and_pad_token_sequence

import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from .encoder import Encoder, QueryType


class GraphEncoder(Encoder):
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
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

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
    def output_representation_size(self):
        # TODO: fix this
        return self.get_hyper('self_attention_hidden_size')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("graph_encoder"):
            self._make_placeholders()
            self.__make_input_model()
            self.__build_graph_propagation_model()
            # TODO: create self.ops['code_representations'] from self.ops['final_node_representations']
            #if self.params['graph_node_label_representation_size'] != self.params['hidden_size']:
            # self.ops['projected_node_features'] = \
            #     tf.keras.layers.Dense(units=h_dim,
            #                           use_bias=False,
            #                           activation=activation_fn,
            #                           )(self.ops['initial_node_features'])
            return self.ops['final_node_representations']



    def __build_graph_propagation_model(self) -> tf.Tensor:
        h_dim = self.params['hidden_size']
        activation_fn = get_activation(
            self.params['graph_model_activation_function'])
        if self.params['graph_node_label_representation_size'] != self.params['hidden_size']:
            self.ops['projected_node_features'] = \
                tf.keras.layers.Dense(units=h_dim,
                                      use_bias=False,
                                      activation=activation_fn,
                                      )(self.ops['initial_node_features'])
        else:
            self.ops['projected_node_features'] = self.ops['initial_node_features']

        cur_node_representations = self.ops['projected_node_features']
        last_residual_representations = tf.zeros_like(cur_node_representations)
        for layer_idx in range(self.params['graph_num_layers']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                cur_node_representations = \
                    tf.nn.dropout(cur_node_representations, rate=1.0 -
                                  self.__placeholders['graph_layer_input_dropout_keep_prob'])
                if layer_idx % self.params['graph_residual_connection_every_num_layers'] == 0:
                    t = cur_node_representations
                    if layer_idx > 0:
                        cur_node_representations += last_residual_representations
                        cur_node_representations /= 2
                    last_residual_representations = t
                cur_node_representations = \
                    self._apply_gnn_layer(
                        cur_node_representations,
                        self.ops['adjacency_lists'],
                        self.ops['type_to_num_incoming_edges'],
                        self.params['graph_num_timesteps_per_layer'])
                if self.params['graph_inter_layer_norm']:
                    cur_node_representations = tf.contrib.layers.layer_norm(
                        cur_node_representations)
                if layer_idx % self.params['graph_dense_between_every_num_gnn_layers'] == 0:
                    cur_node_representations = \
                        tf.keras.layers.Dense(units=h_dim,
                                              use_bias=False,
                                              activation=activation_fn,
                                              name="Dense",
                                              )(cur_node_representations)

        self.ops['final_node_representations'] = cur_node_representations

    @abstractmethod
    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: List[tf.Tensor],
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
        label_embedding_size = self.params['graph_node_label_representation_size']  # D
        # U ~ num unique labels
        # C ~ num characters (self.params['graph_node_label_max_num_chars'])
        # A ~ num characters in alphabet
        unique_label_chars_one_hot = tf.one_hot(indices=unique_labels_as_characters,
                                                depth=len(ALPHABET),
                                                axis=-1)  # Shape: [U, C, A]

        # Choose kernel sizes such that there is a single value at the end:
        char_conv_l1_kernel_size = 5
        char_conv_l2_kernel_size = \
            self.params['graph_node_label_max_num_chars'] - \
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

        node_label_char_length = self.params['graph_node_label_max_num_chars']
        self.placeholders['unique_labels_as_characters'] = \
            tf.placeholder(dtype=tf.int32, shape=[
                           None, node_label_char_length], name='unique_labels_as_characters')
        self.placeholders['node_labels_to_unique_labels'] = \
            tf.placeholder(dtype=tf.int32, shape=[
                           None], name='node_labels_to_unique_labels')
        self.placeholders['adjacency_lists'] = \
            [tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
                for e in range(self.num_edge_types)]
        self.placeholders['type_to_num_incoming_edges'] = \
            tf.placeholder(dtype=tf.float32, shape=[
                           self.num_edge_types, None], name='type_to_num_incoming_edges')

        self.ops['initial_node_features'] = \
            self.__get_node_label_charcnn_embeddings(self.placeholders['unique_labels_as_characters'],
                                                     self.placeholders['node_labels_to_unique_labels'])
        self.ops['adjacency_lists'] = self.placeholders['adjacency_lists']
        self.ops['type_to_num_incoming_edges'] = self.placeholders['type_to_num_incoming_edges']

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

    #     return tf.nn.embedding_lookup(params=token_embeddings, ids=token_inp)

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

        # TODO: load the code (graph?) metadata
        if use_subtokens:
            data_to_load = cls._to_subtoken_stream(
                data_to_load, mark_subtoken_end=mark_subtoken_end)
        raw_metadata['token_counter'].update(data_to_load)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super().finalise_metadata(
            encoder_label, hyperparameters, raw_metadata_list)
        merged_token_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']

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

        final_metadata['token_vocab'] = token_vocabulary
        # Save the most common tokens for use in data augmentation:
        final_metadata['common_tokens'] = merged_token_counter.most_common(50)
        return final_metadata

    @classmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
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
            if encoder_label == 'query':
                # Set the query tokens to the function name, broken up into its sub-tokens:
                data_holder[QueryType.FUNCTION_NAME.value] = split_identifier_into_parts(
                    function_name)
            elif encoder_label == 'code':
                # In the code, replace the function name with the out-of-vocab token everywhere it appears:
                data_holder[QueryType.FUNCTION_NAME.value] = [Vocabulary.get_unk() if token == function_name else token
                                                              for token in data_to_load]

        # Sub-tokenize, convert, and pad both versions:
        for key, data in data_holder.items():
            if not data:
                result_holder[f'{encoder_label}_tokens_{key}'] = None
                result_holder[f'{encoder_label}_tokens_mask_{key}'] = None
                result_holder[f'{encoder_label}_tokens_length_{key}'] = None
                continue
            if hyperparameters[f'{encoder_label}_use_subtokens']:
                data = cls._to_subtoken_stream(data,
                                               mark_subtoken_end=hyperparameters[
                                                   f'{encoder_label}_mark_subtoken_end'])
            tokens, tokens_mask = \
                convert_and_pad_token_sequence(metadata['token_vocab'], list(data),
                                               hyperparameters[f'{encoder_label}_max_num_tokens'])
            # Note that we share the result_holder with different encoders, and so we need to make our identifiers
            # unique-ish
            result_holder[f'{encoder_label}_tokens_{key}'] = tokens
            result_holder[f'{encoder_label}_tokens_mask_{key}'] = tokens_mask
            result_holder[f'{encoder_label}_tokens_length_{key}'] = int(
                np.sum(tokens_mask))

        if result_holder[f'{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}'] is None or \
                int(np.sum(result_holder[f'{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}'])) == 0:
            return False

        return True

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        """
        Implements various forms of data augmentation.
        """
        current_sample = dict()

        # Train with some fraction of samples having their query set to the function name instead of the docstring, and
        # their function name replaced with out-of-vocab in the code:
        current_sample['tokens'] = sample[f'{self.label}_tokens_{query_type}']
        current_sample['tokens_mask'] = sample[f'{self.label}_tokens_mask_{query_type}']
        current_sample['tokens_lengths'] = sample[f'{self.label}_tokens_length_{query_type}']

        # In the query, randomly add high-frequency tokens:
        # TODO: Add tokens with frequency proportional to their frequency in the vocabulary
        if is_train and self.label == 'query' and self.hyperparameters['query_random_token_frequency'] > 0.:
            total_length = len(current_sample['tokens'])
            length_without_padding = current_sample['tokens_lengths']
            # Generate a list of places in which to insert tokens:
            insert_indices = np.array([random.uniform(0., 1.) for _ in range(
                length_without_padding)])  # don't allow insertions in the padding
            # insert at the correct frequency
            insert_indices = insert_indices < self.hyperparameters['query_random_token_frequency']
            insert_indices = np.flatnonzero(insert_indices)
            if len(insert_indices) > 0:
                # Generate the random tokens to add:
                tokens_to_add = [random.randrange(0, len(self.metadata['common_tokens']))
                                 for _ in range(len(insert_indices))]  # select one of the most common tokens for each location
                # get the word corresponding to the token we're adding
                tokens_to_add = [self.metadata['common_tokens']
                                 [token][0] for token in tokens_to_add]
                # get the index within the vocab of the token we're adding
                tokens_to_add = [self.metadata['token_vocab'].get_id_or_unk(
                    token) for token in tokens_to_add]
                # Efficiently insert the added tokens, leaving the total length the same:
                to_insert = 0
                output_query = np.zeros(total_length, dtype=int)
                # iterate only through the beginning of the array where changes are being made
                for idx in range(min(length_without_padding, total_length - len(insert_indices))):
                    if to_insert < len(insert_indices) and idx == insert_indices[to_insert]:
                        output_query[idx +
                                     to_insert] = tokens_to_add[to_insert]
                        to_insert += 1
                    output_query[idx +
                                 to_insert] = current_sample['tokens'][idx]
                current_sample['tokens'] = output_query
                # Add the needed number of non-padding values to the mask:
                current_sample['tokens_mask'][length_without_padding:length_without_padding + len(
                    tokens_to_add)] = 1.
                current_sample['tokens_lengths'] += len(tokens_to_add)

        # Add the current sample to the minibatch:
        [batch_data[key].append(current_sample[key])
         for key in current_sample.keys() if key in batch_data.keys()]

        return False

    # TODO: remove/refactor this
    # def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
    #     return (self.__embeddings,
    #             list(self.metadata['token_vocab'].id_to_token))
