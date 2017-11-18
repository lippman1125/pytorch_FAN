from __future__ import print_function
import argparse
import os


def argparser():
    P = argparse.ArgumentParser(description='Train network script')
    P.add_argument('--data', type=str, default='data/300W_LP', help='path to dataset')
    P.add_argument('--seed', type=int, default=0, help='maunlly set RNG seed')
    P.add_argument('--nGpu', type=int, default=1, help='number of gpu(s) to use')
    P.add_argument('--snapshot', type=int, default=3, help='save a snapshot every n epoch(s)')
    P.add_argument('--epochs', type=int, default=40, help='Number of total epochs to run')
    P.add_argument('--workers', type=int, default=4, help='number of data loader threads')
    # for a single GPU.
    P.add_argument('--train-batch', type=int, default=24, help='minibatch size')
    P.add_argument('--val-batch', type=int, default=10, help='minibatch size')
    P.add_argument('-c', '--checkpoint', type=str, default='checkpoint', help='model save path')
    P.add_argument('--resume', type=str, default='', help='resume from lasteset saved checkpoints')
    P.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    P.add_argument('--momentum', type=float, default=0.0, help='momentum')
    P.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    P.add_argument('--netType', type=str, default='fan', help='options: fan')
    P.add_argument(
        '--pointType', type=str, default='2D', choices=['2D', '3D'], help='2D or 3D face alignment')
    P.add_argument('--nModules', type=int, default=1, help='number of modules per level')
    P.add_argument('--nStacks', type=int, default=4, help='number of stacked network(s)')
    P.add_argument('--use-se', action='store_true', help='use SE layer or not')
    P.add_argument('--use-attention', action='store_true', help='use SE layer or not')
    P.add_argument(
        '--schedule', type=int, nargs="+", default=[15, 30], help='adjust lr at this epoch')
    P.add_argument('--gamma', type=float, default=0.1, help='lr decay')
    P.add_argument(
        '--nFeats', type=int, default=256, help='block width (number of intermediate channels)')
    P.add_argument('--retrain', type=str, default='', help='path to model to retrain with')
    P.add_argument('--optimState', type=str, default='', help='path to optimState to reload from')
    P.add_argument('--scale-factor', type=float, default=0.3, help='scaling factor')
    P.add_argument('--rot-factor', type=float, default=30, help='rotation factor(in degrees)')
    P.add_argument('-e', '--evaluation', action='store_true', help='show intermediate results')
    # P.add_argument('--reval', type=float, default='checkpoint/fan3d/300W-LP')
    P.add_argument('--debug', action='store_true', help='show intermediate results')
    P.add_argument('--flip', action='store_true', help='Flip input image')
    P.add_argument(
        '--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    args = P.parse_args()

    return args
