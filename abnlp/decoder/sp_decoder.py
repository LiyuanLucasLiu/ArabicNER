
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import logging
import unicodedata
import torch_scope

from abnlp.common.utils import init_linear, log_sum_exp
from abnlp.common.utils import VariationalDropout

logger = logging.getLogger(__name__)

class spCRFDecoder(nn.Module):

    def __init__(self, arg):

        super(spCRFDecoder, self).__init__()

        logger.info('Building CRF spDecoder')

        if type(arg) is not dict:
            arg = vars(arg)

        try:
            logger.info('Loading label dictionary from: {}'.format(arg['spDecoder']['label_dict']))
            with open(arg['spDecoder']['label_dict'], 'r') as fin:
                self.label_dict = json.load(fin)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['spDecoder']['label_dict']))
            raise

        self.sof = len(self.label_dict)
        self.label_dict['<sof>'] = len(self.label_dict)
        self.eof = len(self.label_dict)
        self.label_dict['<eof>'] = len(self.label_dict)

        self.reverse_label_dict = {v: k for k, v in self.label_dict.items()}

        self.tagset_size = len(self.label_dict)
        self.hidden2tag = nn.Linear(arg['spDecoder']['input_dim'], self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        init_linear(self.hidden2tag)
        self.transitions.data.zero_()

        logger.info('CRF spDecoder has been built successfully.')

        if arg['spDecoder'].get('dropunit', 'vanilla') == 'variational':
            self.dropout = VariationalDropout(p= arg['spDecoder']['droprate'])
        else:
            self.dropout = nn.Dropout(p= arg['spDecoder']['droprate'])

    def forward(self, x, y = None):

        device = next(self.parameters()).device

        bat_size = x['rep'].size(0)
        seq_len = x['rep'].size(1)
        scores = self.dropout(x['rep'])
        scores = self.hidden2tag(scores).view(-1, 1, self.tagset_size)
        ins_num = seq_len * bat_size
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)
        crf_scores = crf_scores.view(bat_size, seq_len, self.tagset_size, self.tagset_size)

        if y is not None:

            tgt = list()
            for ind, instance in enumerate(y):
                assert (x['len'][ind].item() == instance['len'] + 1)
                tgt.append([self.sof * self.tagset_size + instance['label'][0]] + [instance['label'][ind] * self.tagset_size + instance['label'][ind+1] for ind in range(instance['len'] - 1)] + [instance['label'][-1] * self.tagset_size + self.eof] + [self.eof * self.tagset_size + self.eof] * (seq_len - instance['len'] - 1))
            tgt = torch.LongTensor(tgt).to(device)
            tg_energy = torch.gather(crf_scores.view(bat_size, seq_len, -1), 2, tgt.unsqueeze(2)).view(bat_size, seq_len)
            # tg_energy = tg_energy.masked_select(mask).sum()

            partition = crf_scores[:, 0, self.sof, :]
            tgt_partition = tg_energy[:, 0]

            for ind in range(1, seq_len):
                cur_values = crf_scores[:, ind, :, :] + partition.contiguous().unsqueeze(2).expand(bat_size, self.tagset_size, self.tagset_size)
                cur_partition = log_sum_exp(cur_values)

                cur_tgt = tgt_partition + tg_energy[:, ind]

                # cur_mask = (x['len'] <= ind)
                cur_mask = (ind < x['len'])
                tgt_partition.masked_scatter_(cur_mask, cur_tgt.masked_select(cur_mask))

                cur_mask = cur_mask.view(bat_size, 1).expand(bat_size, self.tagset_size)
                partition.masked_scatter_(cur_mask, cur_partition.masked_select(cur_mask))

            tgt_partition = tgt_partition.sum()
            partition = partition[:, self.eof].sum()

            return {'loss': (partition - tgt_partition) / bat_size}

        else:

            decode_idx = torch.LongTensor(bat_size, seq_len-1).to(device)

            forscores = crf_scores[:, 0, self.sof, :]
            back_points = list()

            for ind in range(1, seq_len):
                cur_values = crf_scores[:, ind, :, :] + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size, self.tagset_size)
                forscores, cur_bp = torch.max(cur_values, 1)

                # cur_mask = (x['len'] > ind)
                cur_mask = (ind >= x['len'])
                cur_bp.masked_fill_(cur_mask.view(bat_size, 1).expand(bat_size, self.tagset_size), self.eof)
                back_points.append(cur_bp)

            pointer = back_points[-1][:, self.eof]
            decode_idx[:, -1] = pointer
            for idx in range(seq_len-2, -1, -1):
                back_point = back_points[idx]
                index = pointer.contiguous().view(-1, 1)
                pointer = torch.gather(back_point, 1, index).view(-1)
                decode_idx[:, idx] = pointer

            return {'label': decode_idx}

    def to_spans(self, sequence):
        chunks = []
        current = None

        for i, y in enumerate(sequence):

            label = self.reverse_label_dict[y]

            if label.startswith('B-'):

                if current is not None:
                    chunks.append('@'.join(current))
                current = [label.replace('B-', ''), '%d' % i]

            elif label.startswith('S-'):

                if current is not None:
                    chunks.append('@'.join(current))
                    current = None
                base = label.replace('S-', '')
                chunks.append('@'.join([base, '%d' % i]))

            elif label.startswith('I-'):

                if current is not None:
                    base = label.replace('I-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]

                else:
                    current = [label.replace('I-', ''), '%d' % i]

            elif label.startswith('E-'):

                if current is not None:
                    base = label.replace('E-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                        chunks.append('@'.join(current))
                        current = None
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]
                        chunks.append('@'.join(current))
                        current = None

                else:
                    current = [label.replace('E-', ''), '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                if current is not None:
                    chunks.append('@'.join(current))
                current = None

        if current is not None:
            chunks.append('@'.join(current))

        return set(chunks)
