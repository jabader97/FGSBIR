#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import utils
import argparse
import pretrainedmodels
import os


class Options:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

        parser.add_argument('--dataset_name', type=str, default='ShoeV2')
        parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
        parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                            help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
        parser.add_argument('--root_dir', type=str, default='./../')
        parser.add_argument('--batchsize', type=int, default=16)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--nThreads', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--max_epoch', type=int, default=200)
        parser.add_argument('--eval_freq_iter', type=int, default=100)
        parser.add_argument('--print_freq_iter', type=int, default=1)
        parser.add_argument('--path_aux', type=str, default=os.getcwd())

        parser = self.wandb_parse(parser)
        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()

    def wandb_parse(self, parser):
        # wandb args
        parser.add_argument('--log_online', action='store_true',
                            help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
        parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')
        parser.add_argument('--project', default='Sample_Project', type=str,
                            help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
        parser.add_argument('--group', default='Sample_Group', type=str, help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                                   In --savename default setting part of the savename.')
        parser.add_argument('--savename', default='group_plus_seed', type=str,
                            help='Run savename - if default, the savename will comprise the project and group name (see wandb_parameters()).')
        return parser
