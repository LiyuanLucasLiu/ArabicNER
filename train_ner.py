from torch_scope import wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
import logging
import functools
import json

from abnlp.model import seqLabel
from abnlp.encoder import strFromFileEncoderWrapper
from abnlp.common.utils import adjust_learning_rate
from abnlp.optim import Nadam

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_root', default="./checkpoint")
    parser.add_argument('--config', default="./config/ner0.json")
    parser.add_argument('--train_file', default="./config/ner0.json")
    conf = parser.parse_args()

    with open(conf.config, 'r') as fin:
        args = json.load(fin)

    pw = wrapper(os.path.join(conf.cp_root, args["checkpoint_name"]), args["checkpoint_name"])

    logger.info('Loading the data...')
    train_data = strFromFileEncoderWrapper(args, processed_file = conf.train_file)

    logger.info('Checking the device...')    
    gpu_index = pw.auto_device() if 'auto' == args["gpu"] else int(args["gpu"])
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)

    logger.info("Exp: {}".format(args['checkpoint_name']))
    logger.info("Config: {}".format(args))

    logger.info('Saving the configure...')
    pw.save_configue(args)

    logger.info('Building the model...')
    model = seqLabel(args)
    
    logger.info('Loading to GPU: {}'.format(gpu_index))
    model.to(device)

    print(model)

    logger.info('Constructing optimizer')
    param_dict = filter(lambda t: t.requires_grad, model.parameters())
    optim_map = {'Nadam': Nadam, 'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': optim.SGD, 'RMSprop': optim.RMSprop}
    weight_decay = 0.0 if 'weight_decay' not in args else args['weight_decay']
    if args['lr'] > 0:
        optimizer=optim_map[args['update']](param_dict, lr=args['lr'], weight_decay=weight_decay)
    else:
        optimizer=optim_map[args['update']](param_dict, weight_decay=weight_decay)

    logger.info('Setting up training environ.')

    normalizer=0
    tot_loss = 0

    for indexs in range(args['epoch']):
        model.spEncoder.pipeline_dict['lm'].f_lm.init_hidden()
        model.spEncoder.pipeline_dict['lm'].b_lm.init_hidden()
    
        logger.info('###### {} ######'.format(args['checkpoint_name']))
        logger.info('Epoch: {}'.format(indexs))

        model.train()
        for x, y in train_data.get_tqdm(device, args['batch_size'], include_last = False):

            model.zero_grad()
            loss = model(x, y)['loss']

            tot_loss += loss.item()
            normalizer += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            optimizer.step()

        if args['lr'] > 0 and args['lr_decay'] > 0:
            current_lr = args['lr'] / (1 + (indexs + 1) * args['lr_decay'])
            adjust_learning_rate(optimizer, current_lr)

        if args['lr'] > 0 and args['burn_in_ratio'] > 0 and indexs == args['burn_in']:
            current_lr = args['lr'] * args['burn_in_ratio']
            adjust_learning_rate(optimizer, current_lr)
            logger.info('lr is modified to: {}'.format(current_lr))

        pw.add_loss_vs_batch({'training_loss': tot_loss / (normalizer + 1e-9)}, indexs, use_logger = True)
        tot_loss = 0
        normalizer = 0

    torch.save(model, os.path.join(conf.cp_root, args['checkpoint_name'], 'best.th'))

    pw.close()
