# -*- coding: utf-8 -*-
# Date: 2020/11/21 15:26

"""
tool lib
"""
__author__ = 'tianyu'

import argparse
import glob
import os
import shutil
import subprocess
import time
from os import path as osp
from pathlib import Path

import toml
import torch
import yaml




def create_dir(d_path):
    dir_path = Path(d_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def repeat_dir_name(log_path):
    if os.path.isdir(log_path):
        rep_len = len(list(glob.glob(f"{log_path}*")))
        log_path += f"_repflag_{rep_len}"
    return log_path


def code_snapshot(remark):
    r"""
    save code latest
    :param remark: the experiment name
    :return:
    """
    import sys
    if sys.platform.startswith("win"):
        return "win platform"
    collect_files = ["*.py", "utils/", "models/"]

    code_dir = Path(osp.join("code_snapshots", remark))
    create_dir(code_dir)
    print("code snapshot dir:", code_dir)

    for stir in collect_files:
        for filename in glob.glob(stir):
            if os.path.isdir(filename):
                subprocess.call(["cp", "-rf", filename, code_dir])
            else:
                shutil.copyfile(filename, code_dir / filename)
    return code_dir


def model_save(remark, model, epoch, is_best, fn=None, **kwargs):
    r"""
    save model to disk
    :param fn: save checkpoint name
    :param remark: experiment name
    :param model:
    :param epoch:
    :param is_best:
    :param kwargs: the duck type of some params e.g. a=1, b=1 => save_dict.update({a:1,b:1})
    """
    if fn is None:
        fn = 'now_model.pt'
    save_dir = Path(osp.join("checkpoints", remark))
    create_dir(save_dir)

    save_dict = {"state_dict": model.state_dict(), "epoch": epoch}
    save_dict.update(kwargs)
    torch.save(save_dict, save_dir / fn)
    best_path = save_dir / 'best_model.pt'
    if is_best:
        shutil.copyfile(save_dir / fn, best_path)

    return best_path


def model_size(model):
    r"""
    calculate the params size of model
    :param model:
    :return: model size of MB
    """
    n_parameters = sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])
    return n_parameters * 4 / 1024 / 1024


def nest_deep_update(cfg, new_cfg, __path=''):
    """
    update some k,v in nest dict by toml formatting
    different from the function of the dict.update()
    """
    for k, v in new_cfg.items():
        if cfg.get(k, None) is None:
            print(f'Invalid toml option: {__path[1:]}.{k}')
            continue
        if isinstance(v, dict):
            nest_deep_update(cfg[k], v, '.'.join([__path, str(k)]))
        else:
            cfg[k] = v


def load_config(config_file):
    """
    update file yaml config by commandline args dynamically


    A yaml file has been created by your experiments for saving stable hyper-parameters.
    You want to update some hyper-parameters in one experiment, but most of them remain the same.
    You can use command line flag -up_cfp [toml string] to update new hyper parameters together.

    E.g., # config.yaml

    model:
      name: 'resnet'
    training:
      batch_size: 64
      loss_fn: 'ce'
      optimizer:
        name: 'sgd'
        lr: 0.001

    In this experiment, you want to set lr=0.005 and model_name="res2net", you can add commandline args like:
    -up_cfg training.optimizer.lr=0.005 "model.name='res2net'"
    Note: the string type params should add '' and "" for the total toml string, or throw parser error.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-up_cfg', dest='update_cfg', type=str, nargs='*', default=None,
                        help="new config by toml formatting")
    args = parser.parse_args()

    with open(config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.CLoader)

    if args.update_cfg is not None:
        new_cfg = toml.loads('\n'.join(args.update_cfg))
        nest_deep_update(cfg, new_cfg)

    return cfg


class Timer(object):
    _cid = {}

    @classmethod
    def start(cls, name):
        if name not in cls._cid:
            cls._cid[name] = time.time()
        else:
            raise InterruptedError(f"{name} is running...")

    @classmethod
    def end(cls, name):
        if name not in cls._cid:
            raise InterruptedError(f"{name} not define! All timer:{list(cls._cid.keys())}")
        return print(f"Timer[{name}]: {cls.end_time(name):.2f}s")

    @classmethod
    def end_time(cls, name):
        elapse = time.time() - cls._cid.get(name)
        del cls._cid[name]
        return elapse


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
