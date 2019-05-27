import argparse
import torch_scope
import logging
import random
import json
import codecs

from abnlp.encoder import strRealTimeEncoderWrapper

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/twitter_ner.json")
    parser.add_argument('--train_file_input', default="./config/ner0.json")
    parser.add_argument('--train_file_output', default="./config/ner0.json")
    conf = parser.parse_args()
    with codecs.open(conf.config, 'r', 'utf-8') as fin:
        args = json.load(fin)

    encoder = strRealTimeEncoderWrapper(args)

    input_file, save_to = conf.train_file_input, conf.train_file_output

    x_list = list()
    y_list = list()
    tmp_x = list()
    tmp_y = list()
    index = 1

    with codecs.open(input_file, 'r', 'utf-8') as fin:
        for line in fin:
            if line and not line.isspace():
                cl = line.split()

                if len(cl) > 1:
                    tmp_y.append(cl[1])
                    tmp_x.append([cl[0], index])
            else:
                y_list.append(tmp_y)
                x_list.append(tmp_x)
                tmp_x, tmp_y = list(), list()
            index += 1

    logger.info('Size: {}'.format(len(x_list)))

    processed_data, processed_label = encoder(x_list, y_list)

    processed_data['label'] = processed_label
    
    with open(save_to, 'w') as fout:
        json.dump(processed_data, fout)
