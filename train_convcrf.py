#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import pdb
import torch
import yaml
from distutils.version import LooseVersion
import torchfcn
import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
from train_fcn32s import get_parameters
from train_fcn32s import git_hash
from convcrf import convcrf
from convcrf import trainer as crftrainer
import logging

here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=25, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-7, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    args = parser.parse_args()

    args.model = 'FCN8sAtOnce'
    args.git_hash = git_hash()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.DAVISClassSeg(root, split='train', transform=True),
        batch_size=100, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.DAVISClassSeg(
            root, split='val', transform=True),
        batch_size=100, shuffle=False, **kwargs)

    # 2. model
    config = convcrf.default_conf
    config['filter_size'] = 5
    config['pyinn'] = False
    config['trainable'] = True
    logging.info("Build ConvCRF.")
    ##
    # Create CRF module
    model = convcrf.GaussCRF(conf=config, shape=[224, 224], nclasses=2)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']

    if cuda:
        model = model.cuda()

    # for m in model.modules():
    #     print(dir(m))
    # for p in model.parameters():
    #     print(p.name)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = crftrainer.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()






if __name__ == '__main__':
    main()
