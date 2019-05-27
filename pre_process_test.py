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
    parser.add_argument('--dev_file_input', default="./config/ner0.json")
    parser.add_argument('--test_file_input', default="./config/ner0.json")
    parser.add_argument('--dev_file_output', default="./config/ner0.json")
    parser.add_argument('--test_file_output', default="./config/ner0.json")
    conf = parser.parse_args()
    with codecs.open(conf.config, 'r', 'utf-8') as fin:
        args = json.load(fin)

    encoder = strRealTimeEncoderWrapper(args)

    for input_file, save_to in zip([conf.dev_file_input, conf.test_file_input], [conf.dev_file_output, conf.test_file_output]):

        print(input_file)

        x_list = list()
        tmp_x = list()
        index = 1

        with codecs.open(input_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line and not line.isspace():
                    cl = line.split()[0]
                    tmp_x.append([cl, index])
                else:
                    x_list.append(tmp_x)
                    tmp_x = list()
                index += 1

        logger.info('Size: {}'.format(len(x_list)))

        processed_data, _ = encoder(x_list, None)
        
        with open(save_to, 'w') as fout:
            json.dump(processed_data, fout)
