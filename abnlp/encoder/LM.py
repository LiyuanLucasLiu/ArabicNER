"""
.. module:: LM
    :synopsis: language modeling
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_scope

class LM(nn.Module):
    """
    The language model model.
    
    Parameters
    ----------
    rnn : ``torch.nn.Module``, required.
        The RNNs network.
    soft_max : ``torch.nn.Module``, required.
        The softmax layer.
    c_num : ``int`` , required.
        The number of words.
    c_dim : ``int`` , required.
        The dimension of word embedding.
    droprate : ``float`` , required
        The dropout ratio.
    label_dim : ``int`` , required.
        The input dimension of softmax.    
    """

    def __init__(self, rnn, c_num, c_dim, droprate):

        super(LM, self).__init__()

        self.rnn = rnn

        self.c_num = c_num
        self.c_dim = c_dim

        self.char_embed = nn.Embedding(c_num, c_dim)

        self.rnn_output = self.rnn.output_dim

        # self.decode_char = nn.Linear(self.rnn_output, c_num)
        # self.decode_type = nn.Linear(self.rnn_output, t_num)

        self.drop = nn.Dropout(p=droprate)

        self.forward = self.forward_rl

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.rnn.init_hidden()

    def move_hidden(self, device):
        """
        Initialize hidden states.
        """
        self.rnn.move_hidden(device)

    def forward_rl(self, x):
        """
        Calculate the loss.
        """

        bat_size, seq_len = x['text'].size()

        x_emb = self.char_embed(x['text'])

        x_emb = self.drop(x_emb)

        x_emb = pack_padded_sequence(x_emb, x['len'], batch_first=True)
        rnn_out = self.rnn(x_emb)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first = True)

        rnn_out = rnn_out.contiguous().view(-1, self.rnn_output)
        select_out = rnn_out.index_select(0, x['pos']).view(bat_size, -1, self.rnn_output)

        return select_out
