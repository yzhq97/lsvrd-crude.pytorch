import sys, os
sys.path.insert(0, os.getcwd())
import json
import torch
import torch.nn as nn
import argparse
from lib.train import train
from lib.model.vision_model import VisionModel
from lib.model.language_model import LanguageModel
from lib.model.loss_model import LossModel
from lib.data.sym_dict import SymbolDictionary
from lib.data.dataset import GQATriplesDataset
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lsvrd-vgg-512.json')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--val_freq', type=int, default=1, help="run validation between how many epochs")
    parser.add_argument('--out_dir', type=str, default='out')
    args = parser.parse_args()
    assert torch.cuda.device_count() >= args.n_gpus
    return args

if __name__ == "__main__":

    args = parse_args()
    cfg = edict(json.load(open(args.config)))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print("model configs:")
    print(json.dumps(cfg, indent=2))
    print()

    print("run args:")
    for arg in vars(args):
        print("%10s: %s" % (arg, str(getattr(args, arg))))
    print()

    print("parsing dictionaries")
    word_dict = SymbolDictionary.load_from_file(cfg.word_dict)
    ent_dict = SymbolDictionary.load_from_file(cfg.ent_dict)
    pred_dict = SymbolDictionary.load_from_file(cfg.pred_dict)

    print("loading train data...")
    train_set = GQATriplesDataset.create(cfg, word_dict, ent_dict, pred_dict,
                                         cfg.train.triples_path, cfg.train.image_dir,
                                         mode="train", preload=False)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    print("loading val data...")
    val_set = GQATriplesDataset.create(cfg, word_dict, ent_dict, pred_dict,
                                       cfg.val.triples_path, cfg.val.image_dir,
                                       mode="eval", preload=False)
    val_loader = DataLoader(train_set, batch_size=cfg.val.batch_size,
                            shuffle=True, num_workers=args.n_workers)

    print("building model")
    vision_model = VisionModel.build_from_config(cfg.vision_model)
    language_model = LanguageModel.build_from_config(cfg.language_model, word_dict)
    loss_model = LossModel.build_from_config(cfg.loss_model)

    print("training started...")
    train(vision_model, language_model, loss_model,
          train_loader, val_loader, word_dict, ent_dict, pred_dict,
          args.n_epochs, args.val_freq, args.out_dir, cfg)