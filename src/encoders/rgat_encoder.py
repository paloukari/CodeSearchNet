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
from gnns import sparse_rgat_layer

class RGAT_Encoder(GraphEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'hidden_size': 128,
            'num_heads': 4,
            'graph_activation_function': 'tanh',
            'graph_layer_input_dropout_keep_prob': 1.0,
            'graph_dense_between_every_num_gnn_layers': 10000,
            'graph_residual_connection_every_num_layers': 10000,
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    def _apply_gnn_layer(self,
                         node_representations: tf.Tensor,
                         adjacency_lists: Dict[int, tf.Tensor],
                         type_to_num_incoming_edges: tf.Tensor,
                         num_timesteps: int) -> tf.Tensor:
        return sparse_rgat_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            state_dim=self.hyperparameters['code_hidden_size'],
            num_timesteps=num_timesteps,
            num_heads=self.hyperparameters['code_num_heads'],
            activation_function=self.hyperparameters['code_graph_activation_function'],
        )
