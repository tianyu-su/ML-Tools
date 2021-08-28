# -*- coding: utf-8 -*-
# Date: 2021/08/28 22:10

"""

"""
__author__ = 'tianyu'

import os
import random

import numpy as np
import torch


def fix_seed():
    seed = 1541233601
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("fix seed: {}".format(seed))


def fixed_length_padding(sequences, batch_first=False, max_len=-1, padding_value=0.0):
    r"""
    ref:    torch.nn.utils.rnn.pad_sequence
    """
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len < 0:
        max_len = max([s.size(0) for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

def get_lr(optimizer):
    r"""
    get the now learning rate
    :param optimizer:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
