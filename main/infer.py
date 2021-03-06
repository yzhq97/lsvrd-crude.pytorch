import sys, os
sys.path.insert(0, os.getcwd())
import argparse
import json
import h5py
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from lib.data.sym_dict import SymbolDictionary
from lib.infer import infer
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
    parser.add_argument('--dataset', type=str, default='gqa')
    parser.add_argument('--objects_dir', type=str, default='data/gqa/objects')
    parser.add_argument('--cache_dir', type=str, default='cache/gqa/resnet101_14x14')
    parser.add_argument('--out_dir', type=str, default='data/gqa/lsvrd_features')
    args = parser.parse_args()

    _, cfg_name = os.path.split(args.config)
    cfg_name, _ = os.path.splitext(cfg_name)
    args.cfg_name = cfg_name

    assert args.dataset in ["gqa", "vqa2"]

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

    print("getting boxes from objects_dir")

    if args.dataset == "gqa":
        info_path = os.path.join(args.objects_dir, "gqa_objects_info.json")
        indices = json.load(open(info_path))
        h5_paths = [ os.path.join(args.objects_dir, "gqa_objects_%d.h5" % i) for i in range(16) ]
        h5s = [ h5py.File(h5_path) for h5_path in h5_paths ]
        h5_boxes = [ h5["bboxes"] for h5 in h5s ]
        h5_features = [ h5["features"] for h5 in h5s]
        all_boxes = {}
        rearange_inds = np.argsort([ 1, 0, 3, 2 ]) # (x1, y1, x2, y2) -> (y1, x1, y2, x2)
        for image_id, meta in tqdm(indices.items()):
            file_idx = meta["file"]
            idx = meta["idx"]
            n_use = min(meta["objectsNum"], args.n_obj)
            width = float(meta["width"])
            height = float(meta["height"])
            boxes = h5_boxes[file_idx][idx, :n_use, :]
            boxes = boxes[:, rearange_inds] # (x1, y1, x2, y2) -> (y1, x1, y2, x2)
            boxes = boxes / np.array([height, width, height, width])
            all_boxes[image_id] = boxes
        n_entries = len(all_boxes)
        indices = { key: (val["file"], val["idx"]) for key, val in indices.items() }
    elif args.dataset == "vqa2":
        info_path = os.path.join(args.objects_dir, "info.json")
        info = json.load(open(info_path))
        indices = info['indices']
        meta = info['meta']
        h5_paths = [os.path.join(args.objects_dir, "data_%d.h5" % i) for i in range(16)]
        h5s = [h5py.File(h5_path) for h5_path in h5_paths]
        h5_boxes = [h5["boxes"] for h5 in h5s]
        h5_features = [h5["v_objs"] for h5 in h5s]
        all_boxes = {}
        rearange_inds = np.argsort([1, 0, 3, 2])  # (x1, y1, x2, y2) -> (y1, x1, y2, x2)
        for image_id, idx_pair in tqdm(indices.items()):
            block_idx = idx_pair["block"]
            idx = idx_pair["idx"]
            height, width = meta[image_id]['height'], meta[image_id]['width']
            boxes = h5_boxes[block_idx][idx]
            keep = np.where(np.sum(boxes, axis=1) > 0)
            boxes = boxes[keep]
            boxes = boxes[:, rearange_inds]  # (x1, y1, x2, y2) -> (y1, x1, y2, x2)
            boxes = boxes / np.array([height, width, height, width])
            all_boxes[image_id] = boxes
        n_entries = len(all_boxes)
        indices = {key: (val["block"], val["idx"]) for key, val in indices.items()}
    else:
        raise NotImplementedError

    print("creating h5 loader for pre-extracted feature")
    fields = [{ "name": "features",
                "shape": [cfg.vision_model.feature_dim, cfg.vision_model.feature_height, cfg.vision_model.feature_width],
                "dtype": "float32",
                "preload": False }]
    image_ids = [image_id for image_id in all_boxes.keys()]
    image_ids = list(set(image_ids))
    loader = H5DataLoader.load_from_directory(args.cache_dir, fields, image_ids)

    print("creating h5 writer")
    n_obj = args.n_obj
    emb_dim = cfg.vision_model.emb_dim
    fields = [
        { "name": "objects", "shape": [n_obj, 2048], "dtype": "float32"},
        { "name": "boxes", "shape": [n_obj, 4], "dtype": "float32" },
        # { "name": "entities", "shape": [ n_obj, emb_dim ], "dtype": "float32" },
        { "name": "relations", "shape": [ n_obj, n_obj, emb_dim ], "dtype": "float32" },
        { "name": "rel_mat", "shape": [ n_obj, n_obj ], "dtype": "int32" }
    ]
    writer = H5DataWriter(out_dir, "gqa_lsvrd_features", n_entries, 16, fields)

    print("inference started")
    infer(vision_model, all_boxes, pred_emb, loader, writer, h5_features, indices, args, cfg)
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    cfg = edict(json.load(open(args.config)))
    infer_with_cfg(args, cfg)
