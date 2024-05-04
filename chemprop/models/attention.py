from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function


class AtomAttention(nn.Module):

    def __init__(self,
                 args: TrainArgs = None,
                 input_size: int = None,
                 output_size: int = None,
                 aggregation: str = 'mean',
                 aggregation_norm: float = 100.0,
                 dropout: float = 0.0,
                 cached_vector: torch.Tensor = None,
                 bias: bool = True,
                 mol_feature: bool = True,
                 activation_func: nn.Module = None,
                 dropout_func: nn.Module = None):
        super(AtomAttention, self).__init__()
        if input_size is None:
            input_size = args.hidden_size
        if output_size is None:
            output_size = args.hidden_size
        self.query = nn.Linear(input_size, output_size, bias=bias)
        self.key = nn.Linear(input_size, output_size, bias=bias)
        self.value = nn.Linear(input_size, output_size, bias=bias)
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm
        self.cached_vector = cached_vector
        self.dropout = nn.Dropout(dropout)
        self.mol_feature = mol_feature
        self.activation = activation_func
        self.dropout = dropout_func

    def forward(self,
                hidden_states,
                scope,
                s2_state=None):
        """
        This is an attention module
        When only hidden state is given, following self-attention workflow will be done
        1. Q = W_q * hidden_state, K = W_k * hidden_state, V = W_v * hidden_state
        2. Attention(Q, K, V) = matmul( softmax(matmul(Q, K) / sqrt(d_k)),  V)
        V is the desired output, and softmax(matmul(Q, K) / sqrt(d_k)) is the Attention filter matrix

        When hidden state and s2_state are given, cross-attention works:
        1. Q = W_q * s2_state, K = W_k * hidden_state, V = W_v * hidden_state
        2. Attention(Q, K, V) = matmul( softmax(matmul(Q, K) / sqrt(d_k)),  V)

        :param hidden_states:
        :param scope:
        :param s2_state:
        :return:
        """

        if s2_state is not None:
            Q = self.query(s2_state)
        else:
            Q = self.query(hidden_states)
        V = self.value(hidden_states)
        K = self.key(hidden_states)
        return_vecs = []
        if not self.mol_feature:
            return_vecs.append(self.cached_vector.unsqueeze(0))  # append zero padding for message passing
        for start, size in scope:
            if size == 0:
                return_vecs.append(self.cached_zero_vector)
            else:
                cur_q = Q.narrow(0, start, size)
                cur_k = K.narrow(0, start, size)
                cur_v = V.narrow(0, start, size)

                attention_scores = torch.matmul(cur_q, cur_k.transpose(-1, -2))
                attention_scores = attention_scores / np.sqrt(np.shape(cur_q)[-1])
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                cur_hiddens = torch.matmul(attention_probs, cur_v)
                if self.activation:
                    cur_hiddens = self.activation(cur_hiddens)
                if self.dropout:
                    cur_hiddens = self.dropout(cur_hiddens)

                mol_vec = cur_hiddens   # (num_atoms, hidden_size)

                if self.mol_feature:
                    if self.aggregation == 'mean':
                        mol_vec = cur_hiddens.sum(dim=0) / size
                    elif self.aggregation == 'sum':
                        mol_vec = cur_hiddens.sum(dim=0)
                    elif self.aggregation == 'norm':
                        mol_vec = cur_hiddens.sum(dim=0) / self.aggregation_norm
                    return_vecs.append(mol_vec)
                else:
                    return_vecs.append(mol_vec)

        if self.mol_feature:
            return torch.stack(return_vecs, dim=0)
        else:
            return torch.concat(return_vecs, dim=0)

