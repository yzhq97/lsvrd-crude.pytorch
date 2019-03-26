import sys, os
sys.path.insert(0, os.getcwd())
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from lib.data.h5io import H5DataWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='model configs')
    parser.add_argument('--checkpoint', type=str, help='model checkpoint path, e.g. out/model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=1, help='number of dataloader workers')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--max_entities', type=int, default=36)
    parser.add_argument('--gqa_objects_dir', type=str, default='data/gqa/objects')
    parser.add_argument('--out_dir', type=str, default='data/gqa')
    args = parser.parse_args()
    _, cfg_name = os.path.split(args.config)
    cfg_name, _ = os.path.splitext(cfg_name)
    args.cfg_name = cfg_name
    return args

def infer(args, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_dir = os.path.join(args.out_dir, args.cfg_name)
    torch.backends.cudnn.benchmark = True

    print("model configs:")
    print(json.dumps(cfg, indent=2))
    print()

    print("run args:")
    for arg in vars(args):
        print("%10s: %s" % (arg, str(getattr(args, arg))))
    print()



    # create h5
    n_ent = args.max_entities
    emb_dim = cfg.vision_model.emb_dim
    fields = [
        { "name": "entities", "shape": [ n_ent, emb_dim ], "dtype": "float32" },
        { "name": "relations", "shape": [ n_ent, n_ent, emb_dim ], "dtype": "float32" }
    ]

