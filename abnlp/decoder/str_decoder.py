import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import logging
import unicodedata
import torch_scope

from abnlp.common.utils import init_linear, log_sum_exp, is_english, identify_cross_code, conlidate_code_switch

logger = logging.getLogger(__name__)

from ipdb import set_trace

connect_punc = ['/', '\\']

class strDecoder(object):

    def __init__(self, arg):

        super(strDecoder, self).__init__()

        logger.info('Building String Decoder')

        if type(arg) is not dict:
            arg = vars(arg)

        try:
            logger.info('Loading label dictionary from: {}'.format(arg['strDecoder']['label_dict']))
            with open(arg['strDecoder']['label_dict'], 'r') as fin:
                self.label_dict = json.load(fin)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['strDecoder']['label_dict']))
            raise

        self.reverse_label_dict = {v: k for k, v in self.label_dict.items()}

        logger.info('str Decoder has been built successfully.')

        self.forward = self.forward_submit

    def __call__(self, x, y, file_name):
        return self.forward(x, y, file_name)

    def forward_submit(self, x, y, file_name):
        """
        decode a sentence in the format of <>
        Parameters
        ----------
        feature: ``list``, required.
            Words list
        label: ``list``, required.
            Label list.
        """
        result = ""

        start, surface_name, current = None, None, None
        line_index = -1

        if len(x) != len(y):
            logger.error('Decoding length not equal!')
            
        length = min(len(x), len(y))
        for ind in range(length):
            f = x[ind]
            y_ins = y[ind]

            label = self.reverse_label_dict[y_ins]

            if ind > 0 and ind < length - 2 and f[0] in connect_punc and label == 'O' and (is_english(x[ind-1][0]) or identify_cross_code(x[ind-1][0])) and (is_english(x[ind+1][0]) or identify_cross_code(x[ind+1][0])) and 'MIS' in self.reverse_label_dict[y[ind-1]] and 'MIS' in self.reverse_label_dict[y[ind+1]]:
                # print('{}, {}, {}'.format(x[ind-1][0], x[ind][0], x[ind+1][0]))
                label = 'I-MIS'
                y[ind+1] = self.label_dict['I-MIS']
            
            if label.startswith('I-') and current is not None and label[2:] == current:
                surface_name.append(f[0])
            else:
                if current is not None:
                    result += conlidate_code_switch(file_name, start, f[1]-1, current, surface_name)

                if label == 'O':
                    start, surface_name, current = None, None, None
                else:
                    start = f[1]
                    current = label[2:]
                    surface_name = [f[0]]

            line_index = f[1]

        if current is not None:
            result += conlidate_code_switch(file_name, start, line_index, current, surface_name)

        return result
