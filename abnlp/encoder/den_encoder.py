"""
.. module:: pre-pipeline
    :synopsis: template for encoding pipeline

.. moduleauthor:: Liyuan Liu
"""

#########################################
#### REMARK: Batch Mode only for Now ####
#########################################
import torch_scope

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch

import logging

from abnlp.common.utils import init_linear, VariationalDropout, WordDropout

logger = logging.getLogger(__name__)

class denRNNEncoder(nn.Module):
    def __init__(self, arg):
        super(denRNNEncoder, self).__init__()
        logger.info('Building RNN DenseEncoder...')

        if type(arg) is not dict:
            arg = vars(arg)

        self.input_fields = arg['denEncoder']['input_fields']

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        tmp_rnn_droprate = arg['denEncoder']['droprate'] if arg['denEncoder']['rnn_layer'] > 1 else 0
        self.rnn = rnnunit_map[arg['denEncoder']['rnn_unit']](arg['denEncoder']['rnn_input'], arg['denEncoder']['rnn_hidden'], arg['denEncoder']['rnn_layer'], dropout = tmp_rnn_droprate, bidirectional = True, batch_first = True)
        
        if arg['denEncoder'].get('dropunit', 'variational') == 'variational':
            self.dropout = VariationalDropout(p = arg['denEncoder']['droprate'])
        else:
            self.dropout = nn.Dropout(p = arg['denEncoder']['droprate'])

        logger.info('RNN DenseEncoder has been built successfully.')

        if 'emb_project_from' in arg['denEncoder'] and arg['denEncoder']['emb_project_from'] > 0:
            self.project = nn.Linear(arg['denEncoder']['emb_project_from'], arg['denEncoder']['rnn_input'])
            init_linear(self.project)
        else:
            self.project = None

        self.word_dropout = WordDropout(arg['denEncoder']['word_droprate'])

    def forward(self, x):

        emb = list()
        
        for key in self.input_fields:
            emb.append(x[key]['rep'])
            emblen = x[key]['len']
                
        assert len(emb) > 0
        emb = torch.cat(emb, dim = -1)

        if self.project:
            emb = self.project(self.dropout(emb))

        sorted_len, argsort_len = emblen.sort(descending=True)
        emb = emb[argsort_len, :, :]
        emb = self.dropout(emb)
        emb = pack_padded_sequence(emb, sorted_len, batch_first=True)
        rnn_out, _ = self.rnn(emb)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first = True)
        new_rnn_out = rnn_out.new_empty(rnn_out.size())
        new_rnn_out[argsort_len] = rnn_out

        return {'rep': new_rnn_out, 'len': emblen}
