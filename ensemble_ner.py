from torch_scope import basic_wrapper as bw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
import logging
import functools
import codecs
import json

from abnlp.model import seqLabel, ensembledSeqLabel
from abnlp.encoder import strFromFileEncoderWrapper
from abnlp.common.utils import adjust_learning_rate, rank_by_number, conlidate_code_switch

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='auto')
    parser.add_argument('--cp_root', default="./checkpoint")
    parser.add_argument('--config_list', nargs='+', default=["./config/sner0.json", "./config/sner1.json", "./config/sner2.json", "./config/sner3.json", "./config/sner4.json", "./config/sner5.json", "./config/sner6.json", "./config/sner7.json", "./config/sner8.json"])
    parser.add_argument('-i', '--input', nargs='+', default = ["../processed_data/dev.json", "../processed_data/test.json"])
    parser.add_argument('-n', '--name', nargs='+', default = ["dev.txt", "test.txt"])
    parser.add_argument('-o', '--output', default = "../processed_data/dev_output_0.csv")
    conf = parser.parse_args()

    gpu_index = bw.auto_device() if 'auto' == conf.gpu else int(conf.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)

    with torch.no_grad():
        dev_data, test_data = None, None

        for config_file in conf.config_list:
            with open(config_file, 'r') as fin:
                args = json.load(fin)

            if dev_data is None:

                logger.info('Loading the data...')
                args['strEncoder'] = {k: v for k, v in args['strEncoder'].items() if k != 'label'}
                dev_data = strFromFileEncoderWrapper(args, processed_file = conf.input[0])
                test_data = strFromFileEncoderWrapper(args, processed_file = conf.input[1])
                dev_data = [tup for tup in dev_data.get_tqdm(device, args['batch_size'], shuffle = False)]
                test_data = [tup for tup in test_data.get_tqdm(device, args['batch_size'], shuffle = False)]

            logger.info("Model: {}".format(args['checkpoint_name']))
            logger.info("Config: {}".format(args))

            model_path = os.path.join(conf.cp_root, args['checkpoint_name'], 'best.th')
            logger.info('Building the model from:{}...'.format(model_path))
            model = torch.load(model_path, map_location=lambda storage, loc: storage)
            logger.info('Loading to GPU: {}'.format(gpu_index))
            model.to(device)

            model.spEncoder.pipeline_dict['lm'].f_lm.move_hidden(device)
            model.spEncoder.pipeline_dict['lm'].b_lm.move_hidden(device)

            model.eval()
            ensembledModel = ensembledSeqLabel(model)

            for x, _ in dev_data:
                ensembledModel.ensemble(x)

            for x, _ in test_data:
                ensembledModel.ensemble(x)

            model.cpu()

        logger.info('Ensemble completed')
        
        dev_output = ensembledModel.decode(dev_data, conf.name[0])

        dev_output = rank_by_number(dev_output)

        test_output = ensembledModel.decode(test_data, conf.name[1])

        test_output = rank_by_number(test_output)
        
    with codecs.open(conf.output, 'w', 'utf-8') as fout:
        fout.write('Filename,Start,End,Type,Score,Surface\n')
        fout.write(dev_output+'\n')
        fout.write(test_output)
