import sys, os
sys.path.insert(0, os.getcwd())
import json
import torch
import argparse
import torch.nn as nn
from lib.train import train
from lib.model.vision_model import VisionModel
from lib.model.language_model import LanguageModel, WordEmbedding
from lib.model.loss_model import LossModel
from lib.data.sym_dict import SymbolDictionary
from lib.data.dataset import GQATriplesDataset
from lib.utils import count_parameters
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--val_freq', type=int, default=1, help="run validation between how many epochs")
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--grad_freq', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--preload', action="store_true")
    args = parser.parse_args()
    _, cfg_name = os.path.split(args.config)
    cfg_name, _ = os.path.splitext(cfg_name)
    args.cfg_name = cfg_name
    return args

def train_with_config(args, cfg):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_dir = os.path.join(args.out_dir, args.cfg_name)

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
    attr_dict = SymbolDictionary.load_from_file(cfg.attr_dict)
    pred_dict = SymbolDictionary.load_from_file(cfg.pred_dict)

    print("building model")
    word_emb = WordEmbedding.build_from_config(cfg.language_model, word_dict).cuda()
    word_emb.init_embedding(cfg.language_model.word_emb_init)
    word_emb.freeze()
    vision_model = VisionModel.build_from_config(cfg.vision_model).cuda()
    language_model = LanguageModel.build_from_config(cfg.language_model).cuda()
    ent_loss = LossModel.build_from_config(cfg.ent_loss).cuda()
    rel_loss = LossModel.build_from_config(cfg.rel_loss).cuda()

    n_v_params = count_parameters(vision_model)
    n_l_params = count_parameters(language_model)
    print("vision model: {:,} parameters".format(n_v_params))
    print("language model: {:,} parameters".format(n_l_params))
    print()

    print("loading train data...")
    train_set = GQATriplesDataset.create(cfg, word_dict, ent_dict, attr_dict, pred_dict, cfg.train.triples_path,
                                         mode="train", preload=args.preload)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    print("loading val data...")
    val_set = GQATriplesDataset.create(cfg, word_dict, ent_dict, attr_dict, pred_dict, cfg.val.triples_path,
                                       mode="eval", preload=args.preload)
    val_loader = DataLoader(val_set, batch_size=cfg.val.batch_size,
                            shuffle=True, num_workers=args.n_workers)

    print("training started...")
    train(word_emb, vision_model, language_model, ent_loss, rel_loss,
          train_loader, val_loader, word_dict, ent_dict, pred_dict,
          args.n_epochs, args.val_freq, out_dir, cfg, args.grad_freq)

if __name__ == "__main__":
    args = parse_args()
    cfg = edict(json.load(open(args.config)))
    train_with_config(args, cfg)
