import sys, os
sys.path.insert(0, os.getcwd())
import argparse
import json
import h5py
import math
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from lib.data.sym_dict import SymbolDictionary
from lib.infer import infer_rel_only
from lib.evaluate import get_sym_emb
from lib.utils import count_parameters
from lib.model.vision_model import VisionModel
from lib.model.language_model import LanguageModel, WordEmbedding
from lib.data.h5io import H5DataLoader, H5DataWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='model configs')
    parser.add_argument('--vckpt', type=str, help='vision model checkpoint path')
    parser.add_argument('--lckpt', type=str, help='language model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--n_obj', type=int, default=36)
    parser.add_argument('--objects_dir', type=str)
    parser.add_argument('--out_dir', type=str, default='data/gqa')
    args = parser.parse_args()
    _, cfg_name = os.path.split(args.config)
    cfg_name, _ = os.path.splitext(cfg_name)
    args.cfg_name = cfg_name
    return args

def infer_with_cfg(args, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_dir = os.path.join(args.out_dir, "lsvrd_features", args.cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    print("model configs:")
    print(json.dumps(cfg, indent=2))
    print()

    print("run args:")
    for arg in vars(args):
        print("%10s: %s" % (arg, str(getattr(args, arg))))
    print()

    word_dict = SymbolDictionary.load_from_file(cfg.word_dict)
    pred_dict = SymbolDictionary.load_from_file(cfg.pred_dict)

    print("building language model")
    word_emb = WordEmbedding.build_from_config(cfg.language_model, word_dict).cuda()
    word_emb.init_embedding(cfg.language_model.word_emb_init)
    word_emb.freeze()
    language_model = LanguageModel.build_from_config(cfg.language_model)
    language_model = language_model.cuda()
    lckpt = torch.load(args.lckpt)
    language_model.load_state_dict(lckpt)
    language_model.train(False)
    language_model.eval()
    n_l_params = count_parameters(language_model)
    print("language model: {:,} parameters".format(n_l_params))

    print("obtaining predicate embeddings")
    pred_emb = get_sym_emb(word_emb, language_model, word_dict, pred_dict, cfg.language_model.tokens_length)

    print("building vision model")
    with torch.no_grad():
        vision_model = VisionModel.build_from_config(cfg.vision_model)
        vision_model = vision_model.cuda()
        vckpt = torch.load(args.vckpt)
        vision_model.load_state_dict(vckpt)
        vision_model.train(False)
        vision_model.eval()
    n_v_params = count_parameters(vision_model)
    print("vision model: {:,} parameters".format(n_v_params))

    # load objects h5
    info_path = os.path.join(args.objects_dir, "info.json")
    info = json.load(open(info_path))
    h5_names = [ name for name in os.listdir(args.objects_dir) if name.endswith('.h5') ]
    h5_names = sorted(h5_names)
    h5_paths = [ os.path.join(args.objects_dir, name) for name in h5_names ]
    h5s = [ h5py.File(path, "r+") for path in h5_paths ]

    # load pre extracted features
    print("creating h5 loader")
    fields = [{ "name": "features",
                "shape": [cfg.vision_model.feature_dim, cfg.vision_model.feature_height, cfg.vision_model.feature_width],
                "dtype": "float32",
                "preload": False }]
    image_ids = [image_id for image_id in info["indices"].keys()]
    image_ids = list(set(image_ids))
    loader = H5DataLoader.load_from_directory(cfg.vision_model.cache_dir, fields, image_ids)

    print("inference started")
    infer_rel_only(vision_model, pred_emb, loader, h5s, info, args, cfg)

    for h5 in h5s: h5.close()

if __name__ == "__main__":
    args = parse_args()
    cfg = edict(json.load(open(args.config)))
    infer_with_cfg(args, cfg)
