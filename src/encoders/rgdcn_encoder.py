from collections import Counter
import numpy as np
from typing import Dict, Any, List, Iterable, Optional, Tuple
import random
import re

from utils.tfutils import convert_and_pad_token_sequence

import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from .encoder import Encoder, QueryType
from .graph_encoder import GraphEncoder
from gnns import sparse_rgdcn_layer

class RGDCN_Encoder(GraphEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'max_nodes_in_batch': 25000,
            'hidden_size': 128,
            'num_channels': 8,
            "use_full_state_for_channel_weights": False,
            "tie_channel_weights": False,
            "graph_activation_function": "ReLU",
            "message_aggregation_function": "sum",
            'graph_inter_layer_norm': True,
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        hyperparameters['code_channel_dim'] = hyperparameters['code_hidden_size'] // hyperparameters['code_num_channels']
        super().__init__(label, hyperparameters, metadata)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: Dict[int, tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        return sparse_rgdcn_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            num_channels=self.hyperparameters['code_num_channels'],
            channel_dim=self.hyperparameters['code_channel_dim'],
            num_timesteps=num_timesteps,
            use_full_state_for_channel_weights=self.hyperparameters['code_use_full_state_for_channel_weights'],
            tie_channel_weights=self.hyperparameters['code_tie_channel_weights'],
            activation_function=self.hyperparameters['code_graph_activation_function'],
            message_aggregation_function=self.hyperparameters['code_message_aggregation_function'],
        )
