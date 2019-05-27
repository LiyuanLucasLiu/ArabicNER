import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import json
import logging
import torch_scope

from abnlp.decoder import spCRFDecoder, strDecoder
from abnlp.encoder import spEncoderWrapper, denRNNEncoder

logger = logging.getLogger(__name__)

class seqLabel(nn.Module):

    def __init__(self, arg):

        super(seqLabel, self).__init__()
        restart = arg.get('restart_from', "")
        if not restart or restart.isspace():
            self.spEncoder = spEncoderWrapper(arg)
        self.denEncoder = denRNNEncoder(arg)
        self.spDecoder = spCRFDecoder(arg)
        self.strDecoder = strDecoder(arg)

    def restart(self, previous_model):
        self.spEncoder = previous_model.spEncoder

    def forward(self, x, y=None):
        sp_out = self.spEncoder(x)
        den_out = self.denEncoder(sp_out)
        crf_out = self.spDecoder(den_out, y)

        return crf_out

    def decode(self, dataset_loader, file_name):

        self.eval()

        result = ""
        for x, _ in dataset_loader:
            result += self.decode_batch(x, file_name)

        return result

    def decode_batch(self, x, file_name):
        decoded = self.forward(x)['label'].cpu()
        decoded = torch.unbind(decoded, 0)

        result = ""
        for x_ins, decoded_ins in zip(x['ori'], decoded):
            length = x_ins['len']
            decoded_tmp = decoded_ins[:length]
            result += self.strDecoder(x_ins['text'], decoded_tmp.numpy(), file_name)
            
        return result

class seqLabelEvaluator(object):

    def __init__(self, decoder):
        self.decoder = decoder
        self.reset()

    def reset(self):
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, decoded_data, target_data):
        decoded_data = decoded_data['label'].cpu()
        batch_decoded = torch.unbind(decoded_data, 0)

        for decoded, target in zip(batch_decoded, target_data):
            
            target = target['label']
            length = len(target)
            best_path = decoded[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path.numpy(), target)
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def calc_acc_batch(self, decoded_data, target_data):
        decoded_data = decoded_data['label'].cpu()
        batch_decoded = torch.unbind(decoded_data, 0)

        for decoded, target in zip(batch_decoded, target_data):
            
            target = target['label']
            length = len(target)
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):

        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):

        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy        

    def eval_instance(self, best_path, gold):

        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = self.decoder(gold)
        gold_count = len(gold_chunks)

        guess_chunks = self.decoder(best_path)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def calc_score(self, seq_model, dataset_loader):

        seq_model.eval()
        self.reset()

        for x, y in dataset_loader:
            decoded = seq_model(x)
            self.calc_f1_batch(decoded, y)

        return self.f1_score()

class ensembledSeqLabel(nn.Module):

    def __init__(self, model):
        super(ensembledSeqLabel, self).__init__()

        self.model = model
        self.strDecoder = model.strDecoder
        self.spDecoder = model.spDecoder
        self.tagset_size = model.spDecoder.tagset_size
        self.sof = model.spDecoder.sof
        self.eof = model.spDecoder.eof

    def ensemble(self, x):
        
        crf_label = self.model(x)['label'].cpu()

        batch_size, seq_len = crf_label.size(0), crf_label.size(1)

        if 'crf_score_0' not in x:
            x['crf_score_0'] = crf_score_0 = torch.zeros((batch_size, seq_len+1, self.tagset_size, self.tagset_size), dtype=torch.float32)
            x['crf_score_1'] = torch.zeros((batch_size, seq_len+1, self.tagset_size), dtype=torch.float32)

        for b_index in range(batch_size):
            tmp_index = crf_label[b_index, 0]
            x['crf_score_0'][b_index, 0, self.sof, tmp_index] += 1
            x['crf_score_1'][b_index, 0, tmp_index] += 1
            pre_index = tmp_index

            for s_index in range(1, seq_len):
                tmp_index = crf_label[b_index, s_index]
                x['crf_score_0'][b_index, s_index, pre_index, tmp_index] += 1
                x['crf_score_1'][b_index, s_index, tmp_index] += 1
                pre_index = tmp_index

            x['crf_score_0'][b_index, seq_len, pre_index, self.eof] += 1
            x['crf_score_1'][b_index, seq_len, self.eof] += 1

        return x

    def forward(self, x, y=None):

        batch_size, seq_len = x['crf_score_0'].size(0), x['crf_score_0'].size(1)
        seq_len -= 1

        decode_idx = torch.LongTensor(batch_size, seq_len).cpu()

        forscores = (x['crf_score_0'][:, 0, self.sof, :]) * (x['crf_score_1'][:, 0, :])
        back_points = list()

        for ind in range(1, seq_len+1):
            cur_values = (forscores.contiguous().view(batch_size, self.tagset_size, 1).expand(batch_size, self.tagset_size, self.tagset_size)) + (x['crf_score_1'][:, ind, :].contiguous().view(batch_size, 1, self.tagset_size).expand(batch_size, self.tagset_size, self.tagset_size)) + (x['crf_score_0'][:, ind, :, :])

            forscores, cur_bp = torch.max(cur_values, 1)

            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.eof]
        decode_idx[:, -1] = pointer
        for idx in range(seq_len-1, -1, -1):
            back_point = back_points[idx]
            index = pointer.contiguous().view(-1, 1)
            pointer = torch.gather(back_point, 1, index).view(-1)
            decode_idx[:, idx] = pointer

        return {'label': decode_idx}

    def decode(self, dataset_loader, file_name):

        self.eval()

        result = ""
        for x, _ in dataset_loader:
            result += self.decode_batch(x, file_name)

        return result

    def decode_batch(self, x, file_name):
        decoded = self.forward(x)['label'].cpu()
        decoded = torch.unbind(decoded, 0)

        result = ""
        for x_ins, decoded_ins in zip(x['ori'], decoded):
            length = x_ins['len']
            decoded_tmp = decoded_ins[:length]
            result += self.strDecoder(x_ins['text'], decoded_tmp.numpy(), file_name)
            
        return result
