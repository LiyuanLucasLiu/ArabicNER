"""
.. module:: string-pre-pipeline
    :synopsis: implementation of strEncoder for string match and lm

.. moduleauthor:: Liyuan Liu
"""
import re
import sys
import json
import codecs
import logging
import unicodedata
import torch
import random

logger = logging.getLogger(__name__)

from abnlp.common.utils import iob_iobes, identify_cross_code, is_english

from tqdm import tqdm

punc = {"،", "و", "٪", "×", "[", "]", "–", "\\", "/", '"', "'", "·", ".", "(", ")", "-", ":", "+", "#", "$", "«", "!", "%", "^", "=", "»", "©"}

class strEncoder(object):
    """
    string encoder: convert string into list of one-hot tensors.
    """
    def __init__(self, arg):
        super(strEncoder, self).__init__()
        if type(arg) is not dict:
            arg = vars(arg)

        self.pipeline_dict = dict()
        self.label_pipeline = None
        self.build_pipelines(arg)

    def build_pipelines(self, arg):
        raise NotImplementedError

    def forward(self, x, y = None):
        output_dict = dict()

        for k, v in self.pipeline_dict.items():
            output_dict[k] = v(x)

        if y is not None and self.label_pipeline is not None:
            label = self.label_pipeline(y)
        else:
            label = None
            
        return output_dict, label

    def __call__(self, x, y = None):
        return self.forward(x, y)

class strPipeline(object):
    """
    string encoder: convert string into list of one-hot tensors.
    """
    def __init__(self, arg):
        super(strPipeline, self).__init__()

        if type(arg) is not dict:
            arg = vars(arg)

        self.build_pipeline(arg)

    def build_pipeline(self, arg):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

class strOriRealTimePipeline(strPipeline):
    """
    string encoder: convert string without changing anything.
    """
    def build_pipeline(self, arg):
        return

    def forward(self, x):
        return [{'text': tup, 'len': len(tup)} for tup in x]

class strLMRealTimePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors for language models.
    """
    def build_pipeline(self, arg):
        assert 'lm_dict' in arg
        try: 
            with open(arg['lm_dict'], 'r') as fin:
                self.char_dict = json.load(fin)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['lm_dict']))
            raise
        self.unk_ind = self.char_dict['<unk>']
        # self.eof_ind = char_dict['<eof>']

    def forward(self, x):
        instances = list()

        for line in x:
            f_t = list()
            b_t = list()
            cl = list()

            for word in line:

                new_word = ""
                flag = False
                
                for char in word[0]:
                    if flag:
                        if char in punc:
                            new_word += char + " "
                            flag = True
                        elif char.isspace():
                            flag = True
                        else:
                            new_word += char
                            flag = False
                    else:
                        if char in punc:
                            new_word += " " + char + " "
                            flag = True
                        elif char.isspace():
                            flag = True
                            new_word += char
                        else:
                            flag = False
                            new_word += char

                word = new_word.rstrip()

                f_t.append([self.char_dict.get(char, self.unk_ind) for char in word])
                b_t.append([self.char_dict.get(char, self.unk_ind) for char in word[::-1]])
                cl.append(len(word) + 1)

            f_cl = [tup for tup in cl]
            b_t = b_t[::-1]
            b_cl = cl[::-1]

            instances.append({'forward_text': f_t, 'forward_len': f_cl, 'backward_text': b_t, 'backward_len': b_cl, 'tot_len': sum(f_cl)})# + len(f_t)})

        return instances

class strWERealTimePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors for language models.
    """
    def build_pipeline(self, arg):
        self.we_vocab = {}
        vocab_size = 0
        self.unk_index = -1
        with codecs.open(arg['emb_path'], 'r', 'utf-8') as f:
            for line in f:
                line = line.split()
                if len(line) == 2:
                    continue # vocab_size, emb_size
                self.we_vocab[line[0]] = vocab_size
                vocab_size += 1
        
    def forward(self, x):

        instances = list()

        for line in x:

            word_list = [unicodedata.normalize('NFKD', tup[0]).encode('ascii','ignore').decode('ascii') if identify_cross_code(tup[0]) else tup[0] for tup in line]

            result = [self.we_vocab.get(word, self.unk_index) for word in word_list]
            instances.append({'text': result, 'len': len(line)})

        return instances

class strWSRealTimePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors for language models.
    """
    def build_pipeline(self, arg):
        self.ws_vocab = {'ar':0, 'en':1, 'num': 2}
        
    def forward(self, x):

        instances = list()

        for line in x:
            result = list()
            for word in line:
                if is_english(word[0]) or identify_cross_code(word[0]):
                    if word[0].isdigit():
                        result.append(self.ws_vocab['num'])
                    else:
                        result.append(self.ws_vocab['en'])
                else:
                    result.append(self.ws_vocab['ar'])

            instances.append({'text': result, 'len': len(line)})

        return instances

class strLabelRealTimePipeline(strPipeline):
    def build_pipeline(self, arg):

        self.generate_label_dict = arg.get('generate_label_dict', False)
        self.label_dict_file = arg['label_dict']
        self.label_dict = dict()

        self.convert_to_iobes = arg.get('convert_to_iobes', False)

        if not self.generate_label_dict:
            try: 
                with open(arg['label_dict'], 'r') as fin:
                    self.label_dict = json.load(fin)
            except FileNotFoundError as err:
                logger.error('File not exist: {}'.format(arg['label_dict']))
                raise

    def forward(self, y):

        instances = list()

        for line in y:
            
            encoded = list()
            if self.convert_to_iobes:
                line = iob_iobes(line)

            for ins in line:
                if ins not in self.label_dict:
                    assert self.generate_label_dict
                    self.label_dict[ins] = len(self.label_dict)
                encoded.append(self.label_dict[ins])
            instances.append({'label': encoded, 'len': len(encoded)})

        if self.generate_label_dict:
            with open(self.label_dict_file, 'w') as fout:
                json.dump(self.label_dict, fout)

        return instances


class strFromFilePipeline(strPipeline):
    """
    string encoder: convert string into list of one-hot tensors.    
    """
    def __init__(self, arg, name):
        self.name = name
        super(strFromFilePipeline, self).__init__(arg)

    def build_pipeline(self, arg):
        assert 'processed_file' in arg
        try: 
            with open(arg['processed_file'], 'r') as fin:
                self.instances = json.load(fin)[self.name]
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['processed_file']))
            raise

        return len(self.instances)

    def forward(self, x):
        return [self.instances[ins] for ins in x]

    def size(self):
        return len(self.instances)

class strRealTimeEncoderWrapper(strEncoder):

    def build_pipelines(self, arg):

        for key in arg['strEncoder']:

            if 'lm' in key:
                self.pipeline_dict[key] = strLMRealTimePipeline(arg['strEncoder'][key])

            elif 'we' in key:
                self.pipeline_dict[key] = strWERealTimePipeline(arg['strEncoder'][key])

            elif 'ori' in key:
                self.pipeline_dict[key] = strOriRealTimePipeline(arg['strEncoder'][key])
            elif 'ws' in key:
                self.pipeline_dict[key] = strWSRealTimePipeline(arg['strEncoder'][key])

        if 'label' in arg['strEncoder']:
            self.label_pipeline = strLabelRealTimePipeline(arg['strEncoder']['label'])

class strFromFileEncoderWrapper(strEncoder):

    def __init__(self, arg, processed_file=None):

        if type(arg) is not dict:
            arg = vars(arg)

        if processed_file is not None:
            arg['processed_file'] = processed_file

        super(strFromFileEncoderWrapper, self).__init__(arg)

    def build_pipelines(self, arg):
        length = -1

        for key in arg['strEncoder']:
            arg['strEncoder'][key]['processed_file'] = arg['processed_file']
            logger.info('Building {} pipeline...'.format(key))
            self.pipeline_dict[key] = strFromFilePipeline(arg['strEncoder'][key], key)
            length = self.pipeline_dict[key].size()

        if 'label' in arg['strEncoder']:
            arg['strEncoder']['label']['processed_file'] = arg['processed_file']
            logger.info('Building label pipeline...')
            self.label_pipeline = strFromFilePipeline(arg['strEncoder']['label'], 'label')
            assert (length > 0)
            assert (length == self.label_pipeline.size())
        else:
            self.label_pipeline = None

        self.index_length = length
        logger.info('All pipeline has been built for {}'.format(arg['processed_file']))

    def get_tqdm(self, device, batch_size, shuffle = True, include_last = True):

        return tqdm(self.reader(device, batch_size, shuffle, include_last), mininterval=2, total=self.index_length // batch_size, leave=False, file=sys.stdout, ncols=80)
    
    def reader(self, device, batch_size, shuffle = True, include_last = True):

        index = list(range(self.index_length))
        if shuffle:
            random.shuffle(index)

        cur_idx, end_idx = 0, min(batch_size, self.index_length)
        while cur_idx < self.index_length:
            yield self.forward(index[cur_idx: end_idx], index[cur_idx: end_idx])
            cur_idx = end_idx
            end_idx += batch_size
            if include_last:
                end_idx = min(end_idx, self.index_length)
            elif end_idx > self.index_length:
                break
