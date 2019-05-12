#!/usr/bin/env python

import argparse
import datetime
import os, cv2
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


here = osp.dirname(osp.abspath(__file__))
method = 'qpbo'
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
        torchfcn.datasets.DAVISClassSeg(root, split='train', transform=True, crf=False),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.DAVISClassSeg(
            root, split='val', transform=True, crf=False),
        batch_size=1, shuffle=False, **kwargs)
    loader = val_loader
    # 2. model

    epoch = 0
    iteration = 0

    n_class = len(loader.dataset.class_names)

    val_loss = 0
    visualizations = []
    label_trues, label_preds = [], []
    timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

    for batch_idx, (imgs, lbl_true, img_file) in tqdm.tqdm(
            enumerate(loader), total=len(loader),
            desc='Valid iteration=%d' % iteration, ncols=80,
            leave=False):

        img_file = img_file[0]
        pred_file = img_file.replace('DAVIS', 'out').replace('JPEGImage', method).replace('jpg', 'png')

        lp = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)//255
        lp = cv2.resize(lp, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = imgs[0]
        lt = lbl_true[0]

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

    mean_iu = metrics[2]
    print(mean_iu)


if __name__ == '__main__':
    main()
