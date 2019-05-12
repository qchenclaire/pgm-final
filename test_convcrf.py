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
import logging

here = osp.dirname(osp.abspath(__file__))
def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--max-iteration', type=int, default=3000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-4, help='learning rate',
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
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.DAVISClassSeg(
            root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)
    loader = val_loader
    # 2. model

    model = torchfcn.models.FCN8sAtOnce(n_class=2)
    epoch = 0
    iteration = 0

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])

    if cuda:
        model = model.cuda()

    training = False
    model.eval()
    n_class = len(loader.dataset.class_names)

    val_loss = 0
    visualizations = []
    label_trues, label_preds = [], []
    timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

    config = convcrf.default_conf
    config['filter_size'] = 7
    config['pyinn'] = False
    logging.info("Build ConvCRF.")
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=[224, 224], nclasses=n_class)
    # Cuda computation is required.
    # A CPU implementation of our message passing is not provided.
    gausscrf.cuda()

    for batch_idx, (data, target, unary) in tqdm.tqdm(
            enumerate(loader), total=len(loader),
            desc='Valid iteration=%d' % iteration, ncols=80,
            leave=False):

        data, target, unary = data.cuda(), target.cuda(), unary.cuda()
        shape = unary.shape[-2:]
        unary = unary.reshape([loader.batch_size, n_class, shape[0], shape[1]])
        data, target, unary_var = Variable(data), Variable(target), Variable(unary)

        with torch.no_grad():
            score = gausscrf.forward(unary=unary_var, img=data)

        loss = cross_entropy2d(score, target,
                               size_average=False)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while validating')
        val_loss += loss_data / len(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                visualizations.append(viz)
    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class)

    out = osp.join(args.out, 'visualization_viz')
    if not osp.exists(out):
        os.makedirs(out)
    out_file = osp.join(out, 'iter%012d.jpg' % iteration)
    scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

    val_loss /= len(loader)

    with open(osp.join(args.out, 'log.csv'), 'a') as f:
        elapsed_time = (
            datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
            timestamp_start).total_seconds()
        log = [epoch, iteration] + [''] * 5 + \
              [val_loss] + list(metrics) + [elapsed_time]
        log = map(str, log)
        f.write(','.join(log) + '\n')

    mean_iu = metrics[2]
    print(mean_iu)


if __name__ == '__main__':
    main()
