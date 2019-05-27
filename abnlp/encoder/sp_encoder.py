"""
.. module:: string-pre-pipeline
    :synopsis: implementation of strEncoder for string match and lm

.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import codecs
import logging
import itertools
import unicodedata
from torch_scope import basic_wrapper as bw

logger = logging.getLogger(__name__)

from abnlp.encoder.LM import LM
from abnlp.encoder.basic import BasicRNN
from abnlp.common.utils import init_linear, WordDropout, VariationalDropout

class spEncoder(nn.Module):
    """
    sparse encoder (embedding model): convert one-hot tensors to dense tensors.
    """
    def __init__(self, arg):
        super(spEncoder, self).__init__()

        if type(arg) is not dict:
            arg = vars(arg)

        self.pipeline_dict = None
        self.input_mapping = None
        self.build_pipelines(arg)

    def build_pipelines(self, arg):
        raise NotImplementedError

    def forward(self, x):
        # set_trace()
        output_dict = dict()

        for k, v in self.pipeline_dict.items():
            output_dict[k] = v(x[self.input_mapping[k]])

        return output_dict

class spPipeline(nn.Module):
    """
    string encoder: convert string into list of one-hot tensors.    
    """
    def __init__(self, arg):
        super(spPipeline, self).__init__()

        if type(arg) is not dict:
            arg = vars(arg)

        self.build_pipeline(arg)

    def build_pipeline(self, arg):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

class spLMPipeline(spPipeline):
    """
    """
    def build_pipeline(self, arg):
        logger.info('Building language models...')

        f_rnn = BasicRNN(arg['frnn_layer_num'], arg['frnn_unit'], arg['frnn_emb_dim'], arg['frnn_hid_dim'], arg['lm_droprate'])
        self.f_lm = LM(f_rnn, arg['c_num'], arg['c_dim_lm'], arg['lm_droprate'])

        b_rnn = BasicRNN(arg['brnn_layer_num'], arg['brnn_unit'], arg['brnn_emb_dim'], arg['brnn_hid_dim'], arg['lm_droprate'])
        self.b_lm = LM(b_rnn, arg['c_num'], arg['c_dim_lm'], arg['lm_droprate'])

        logger.info('Loading forward language model from: {}...'.format(arg['flm_weight']))
        try: 
            flm_weight = bw.restore_checkpoint(arg['flm_weight'])['model']
            self.f_lm.load_state_dict(flm_weight, False)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['flm_weight']))
            raise

        logger.info('Loading backward language model from: {}...'.format(arg['blm_weight']))
        try: 
            blm_weight = bw.restore_checkpoint(arg['blm_weight'])['model']
            self.b_lm.load_state_dict(blm_weight, False)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['blm_weight']))
            raise

        for param in self.f_lm.parameters():
            param.requires_grad = False

        for param in self.b_lm.parameters():
            param.requires_grad = False

        try:
            with open(arg['lm_dict'], 'r') as fin:
                self.char_dict = json.load(fin)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['lm_dict']))
            raise

        self.wp = self.char_dict[' ']
        self.eof0 = self.char_dict['\n']
        self.eof1 = self.char_dict['<eof>']

        if 'emb_project_to' in arg and arg['emb_project_to'] > 0 and (arg['emb_project_to'] != arg['frnn_hid_dim'] + arg['brnn_hid_dim']):
            self.project = nn.Linear(arg['frnn_hid_dim'] + arg['brnn_hid_dim'], arg['emb_project_to'])
            if arg.get('dropunit', 'vanilla') == 'variational':
                self.dropout = VariationalDropout(arg['droprate'])
            else:
                self.dropout = nn.Dropout(arg['droprate'])
            init_linear(self.project)
        else:
            self.project = None

        self.word_dropout = WordDropout(arg['word_droprate'])

        logger.info('Language models building completed.')

    def forward(self, x):
        device = next(self.parameters()).device

        #padding 
        x = [(ind, tup) for ind, tup in enumerate(x)]
        x.sort(key=lambda t: t[1]['tot_len'], reverse=True)
        x_rev_ind = [tup[0] for tup in x]
        x_ind = [0 for tup in x_rev_ind]
        for rev_ind, ind in enumerate(x_rev_ind):
            x_ind[ind] = rev_ind

        x = [tup[1] for tup in x]

        max_length = x[0]['tot_len']
        word_max_length = max([len(tup['forward_len']) for tup in x]) + 1

        flm_text, flm_len, flm_pos = list(), list(), list() 
        blm_text, blm_len, blm_pos = list(), list(), list()
        char_len = list()

        shift = 1

        def pad_sentence(input_list):
            result = []
            for ins in input_list:
                result += ins + [self.wp]
            return result[:-1]

        for instance in x:
            pad_char = max_length - instance['tot_len']

            tmp_text = pad_sentence(instance['forward_text'])
            flm_text.append(tmp_text + [self.eof0, self.eof1] + [self.eof1] * (pad_char))
            flm_len.append(len(tmp_text) + 2)

            tmp_pos = list(itertools.accumulate(instance['forward_len'] + [2]))
            tmp_pos = [tup + shift for tup in tmp_pos]
            flm_pos.extend(tmp_pos + [tmp_pos[-1]] * (word_max_length - len(tmp_pos)))

            tmp_text = pad_sentence(instance['backward_text'])
            blm_text.append([self.eof1, self.eof0] + tmp_text + [self.eof1] * (pad_char))
            blm_len.append(len(tmp_text) + 2)

            tmp_pos = list(itertools.accumulate([2] + instance['backward_len'][::-1]))
            tmp_pos = [tup + shift for tup in tmp_pos[::-1]]
            blm_pos.extend(tmp_pos + [tmp_pos[-1]] * (word_max_length - len(tmp_pos)))

            shift += max_length + 1
            char_len.append(len(instance['forward_len']) + 1)

        # set_trace()
        flm_text = torch.LongTensor(flm_text).to(device)
        flm_len = torch.LongTensor(flm_len).to(device)
        flm_pos = torch.LongTensor(flm_pos).to(device)

        blm_text = torch.LongTensor(blm_text).to(device)
        blm_len = torch.LongTensor(blm_len).to(device)
        blm_pos = torch.LongTensor(blm_pos).to(device)

        char_len = torch.LongTensor(char_len).to(device)
        x_ind = torch.LongTensor(x_ind).to(device)
        #run

        # set_trace()

        flm_out = self.f_lm({'text': flm_text, 'len': flm_len, 'pos': flm_pos})
        blm_out = self.b_lm({'text': blm_text, 'len': blm_len, 'pos': blm_pos})

        lm_out = torch.cat([flm_out, blm_out], dim = -1)

        lm_out = lm_out.index_select(0, x_ind)
        char_len = char_len.index_select(0, x_ind)

        if self.project:
            # lm_out = self.activate(self.project(self.dropout(lm_out)))
            lm_out = self.project(self.dropout(lm_out))

        lm_out = self.word_dropout(lm_out)

        return {'rep': lm_out, 'len': char_len}
        
class spWEPipeline(spPipeline):
    """
    """
    def build_pipeline(self, arg):

        embedding_list = list()
        with codecs.open(arg['emb_path'], 'r', 'utf-8') as fin:
            logger.info('Loading embedding from: {}'.format(arg['emb_path']))
            line = fin.readline()
            num_emb, emb_dim = [int(tup) for tup in line.split()]
            for line in fin:
                if '\t' == arg['embed_seperator']:
                    line = line.rstrip().split('\t')[1].split()
                else:
                    assert ' ' == arg['embed_seperator']
                    line = line.rstrip().split()[1:]
                embedding_list.append([float(tup) for tup in line])

        self.embed = nn.Embedding(num_emb, emb_dim).cpu()
        self.embed.weight = nn.Parameter(torch.FloatTensor(embedding_list))
        self.embed.weight.requires_grad = False

        self.eof_embedding = nn.Parameter(torch.FloatTensor(1, emb_dim))
        self.unk_embedding = nn.Parameter(torch.FloatTensor(1, emb_dim))

        bias = np.sqrt(3.0 / emb_dim)
        nn.init.uniform_(self.eof_embedding.data, -bias, bias)
        nn.init.uniform_(self.unk_embedding.data, -bias, bias)

        self.word_dropout = WordDropout(arg['word_droprate'])

        if 'emb_project_to' in arg and arg['emb_project_to'] > 0:
            self.project = nn.Linear(emb_dim, arg['emb_project_to'])
            if arg.get('dropunit', 'vanilla') == 'variational':
                self.dropout = VariationalDropout(arg['droprate'])
            else:
                self.dropout = nn.Dropout(arg['droprate'])
            # self.activate = nn.ReLU()
            init_linear(self.project)
        else:
            self.project = None

        logger.info('Embedding loading completed, {} words are imported'.format(num_emb))

    def cuda(self, device = None):
        self.eof_embedding.cuda(device)
        self.unk_embedding.cuda(device)
        
    def to(self, *args, **kwargs):
        self.eof_embedding.to(*args, **kwargs)
        self.unk_embedding.to(*args, **kwargs)

    def forward(self, x):
        device = self.eof_embedding.device

        batch_size = len(x)

        #padding
        max_length = max(x, key=lambda t: t['len'])['len'] + 1
        word_ids = list()
        char_len = list()
        eof_ind = list()

        shift = 0
        for instance in x:
            word_ids.append(instance['text'] + [0] * (max_length - instance['len']))
            char_len.append(instance['len'] + 1)

            eof_ind.append(shift + instance['len'])
            shift += max_length

        # run
        # set_trace()
        # word_ids = torch.LongTensor(word_ids).cpu()
        word_ids = torch.LongTensor(word_ids).to(device)
        sp_mask = word_ids < 0
        word_ids[sp_mask] = 0
        word_embed = self.embed(word_ids).view(batch_size * max_length, -1)

        # sp_mask.to(device)
        word_embed[eof_ind, :] = self.eof_embedding
        word_embed[sp_mask.view(-1), :] = self.unk_embedding

        word_embed = word_embed.view(batch_size, max_length, -1)
        char_len = torch.LongTensor(char_len).to(device)

        word_embed = self.word_dropout(word_embed)

        if self.project:
            word_embed = self.project(self.dropout(word_embed))

        return {'rep': word_embed, 'len': char_len}

class spWSPipeline(spPipeline):
    """
    """
    def build_pipeline(self, arg):

        self.ws_vocab = {'ar':0, 'en':1, 'num': 2}

        self.embed = nn.Embedding(len(self.ws_vocab), arg['emb_dim']).cpu()

        self.eof_embedding = nn.Parameter(torch.FloatTensor(1, arg['emb_dim']))
        bias = np.sqrt(3.0 / arg['emb_dim'])
        nn.init.uniform_(self.eof_embedding.data, -bias, bias)

        self.word_dropout = WordDropout(arg['word_droprate'])

        logger.info('Embedding building completed, {} word shape are considered'.format(len(self.ws_vocab)))

    def cuda(self, device = None):
        self.eof_embedding.cuda(device)
        
    def to(self, *args, **kwargs):
        self.eof_embedding.to(*args, **kwargs)

    def forward(self, x):
        device = self.eof_embedding.device

        batch_size = len(x)

        #padding
        max_length = max(x, key=lambda t: t['len'])['len'] + 1
        word_ids = list()
        char_len = list()
        eof_ind = list()

        shift = 0
        for instance in x:
            word_ids.append(instance['text'] + [0] * (max_length - instance['len']))
            char_len.append(instance['len'] + 1)

            eof_ind.append(shift + instance['len'])
            shift += max_length

        word_ids = torch.LongTensor(word_ids).to(device)
        word_embed = self.embed(word_ids).view(batch_size * max_length, -1)

        word_embed[eof_ind, :] = self.eof_embedding

        word_embed = word_embed.view(batch_size, max_length, -1)
        char_len = torch.LongTensor(char_len).to(device)

        word_embed = self.word_dropout(word_embed)

        return {'rep': word_embed, 'len': char_len}

class spEncoderWrapper(spEncoder):

    def build_pipelines(self, arg):

        tmp_pipeline_dict = dict()
        self.input_mapping = dict()
        
        for key in arg['spEncoder']:

            if 'lm' in key:    
                tmp_pipeline_dict[key] = spLMPipeline(arg['spEncoder'][key])
                self.input_mapping[key] = arg['spEncoder'][key]['input']

            elif 'we' in key:    
                tmp_pipeline_dict[key] = spWEPipeline(arg['spEncoder'][key])
                self.input_mapping[key] = arg['spEncoder'][key]['input']

            elif 'ws' in key:
                tmp_pipeline_dict[key] = spWSPipeline(arg['spEncoder'][key])
                self.input_mapping[key] = arg['spEncoder'][key]['input']
                
        self.pipeline_dict = nn.ModuleDict(tmp_pipeline_dict)
