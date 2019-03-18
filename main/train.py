import sys, os
sys.path.insert(0, os.getcwd())
import json
import torch
import torch.nn as nn
import argparse
from lib.train import train
from lib.data.sym_dict import SymbolDictionary
from lib.data.dataset import GQATriplesDataset
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lsvrd-512.json')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--n_gpus', type=int, default=1, help="number of gpus to use")
    parser.add_argument('--val_freq', type=int, default=1, help="run validation between how many epochs")
    parser.add_argument('--checkpoint', type=str, help="checkpoint path if wish to resume")
    parser.add_argument('--out_dir', type=str, default='out')
    args = parser.parse_args()
    assert torch.cuda.device_count() >= args.n_gpus
    return args

if __name__ == "__main__":

    args = parse_args()
    cfgs = edict(json.load(open(args.config)))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print("run args:")
    for arg in vars(args):
        print("%10s: %s" % (arg, str(getattr(args, arg))))
    print()

    print("model configs:")
    print(json.dumps(cfgs, indent=2))
    print()