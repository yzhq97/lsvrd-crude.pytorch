import sys, os
sys.path.insert(0, os.getcwd())
import cv2
import time
import json
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm, trange
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from lib.data.sym_dict import SymbolDictionary
from lib.module.backbone import backbones
from lib.data.h5io import H5DataLoader, H5DataWriter


def get_pos_mat(labels):
    N = labels.size(0)
    labels_0 = labels.unsqueeze(0).repeat(N, 1)
    labels_1 = labels.unsqueeze(1).repeat(1, N)
    pos_mat = labels_0 == labels_1
    return pos_mat

def box_union(box_a, box_b):
    ya1, xa1, ya2, xa2 = box_a
    yb1, xb1, yb2, xb2 = box_b
    return min(ya1, yb1), min(xa1, xb1), max(ya2, yb2), max(xa2, xb2)

def box_convert_and_normalize(box, width, height):
    x1, y1, x2, y2 = box
    return y1/height, x1/width, y2/height, x2/width

class GQATriplesDataset(Dataset):

    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.224]).reshape([3, 1, 1])

    train = 0
    eval = 1

    def __init__(self, name, entries,
                 word_dict, ent_dict, pred_dict, tokens_length,
                 image_dir, image_width, image_height,
                 mode, preload,
                 pre_extract=False, cache_dir=None, backbone=None,
                 feature_height=None, feature_width=None, feature_dim=None):
        """
        :param name: dataset name
        :param entries: (subject, object, predicate ) entries. see scripts/generate_triples.py for details
        :param ent_dict: entity dictionary
        :param pred_dict: predicate dictionary
        :param word_dict: language word dictionary
        :param image_dir: folder containing all gqa images
        :param cache_dir: folder containing extracted features
        :param mode: "train" or "eval"
        :param pre_extract: whether or not to extract all feature maps prior to training
        :param preload: whether or not to load all images into memory for faster data loading
        """

        assert mode in ["train", "eval"]
        self.mode = self.train if mode == "train" else self.eval

        self.name = name

        self.word_dict = word_dict
        self.ent_dict = ent_dict
        self.pred_dict = pred_dict
        self.tokens_length = tokens_length

        self.entries = self.preprocess_entries(entries)
        self.image_dir = image_dir
        self.image_width = image_width
        self.image_height = image_height

        self.pre_extract = pre_extract
        self.cache_dir = cache_dir
        self.backbone = backbone
        self.feature_dim = feature_dim

        self.pre_extract = pre_extract
        if pre_extract:
            fields = [ { "name": "features",
                         "shape": [ feature_dim, feature_height, feature_width ],
                         "dtype": "float32",
                         "preload": preload
                         } ]
            image_ids = [ entry.image_id for entry in self.entries ]
            image_ids = list(set(image_ids))
            self.h5_loader = H5DataLoader.load_from_directory(self.cache_dir, fields, image_ids)

        self.preload = preload
        self.images = self.preload_images() if not pre_extract and preload else None

    def __len__(self):

        return len(self.entries)

    def __getitem__(self, idx):

        entry = self.entries[idx]
        image_id = entry.image_id

        if self.pre_extract: image = self.h5_loader[image_id][0]
        elif self.preload: image = self.images[idx]
        else: image = self.preprocess_image(self.load_image(image_id))

        ret = [ image, entry.sbj_box, entry.obj_box, entry.pred_box ]

        if self.mode == self.train:
            ret.extend([entry.sbj_tokens, entry.obj_tokens, entry.pred_tokens])
            ret.extend([entry.sbj_seq_len, entry.obj_seq_len, entry.pred_seq_len])
        else:
            ret.extend([entry.sbj_label, entry.obj_label, entry.pred_label])

        ret = [ image_id ] + [ torch.tensor(item) for item in ret ]

        return ret

    def load_image(self, image_id):

        image_path = os.path.join(self.image_dir, "%s.jpg" % image_id)
        image = cv2.imread(image_path)
        if image is None: raise Exception("image %s does not exist" % image_path)
        return image

    def tokenize(self, text):

        tokens = self.word_dict.tokenize(text)[:self.tokens_length]
        seq_len = len(tokens)
        tokens = tokens + [ len(self.word_dict) ] * (self.tokens_length - len(tokens))
        return tokens, seq_len

    def preprocess_image(self, image):

        h, w, c = image.shape
        if c == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # convert to RGB
        else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_width, self.image_height)) # unify size
        image = image.transpose([2, 0, 1]) # HWC to CHW
        image = image.astype(np.float32) # convert to float
        image = (image / 255.0 - self.mean) / self.std # normalize

        return image

    def preload_images(self):

        print("loading images into memory ...")
        channels = self.feature_dim if self.pre_extract else 3
        images = np.zeros([len(self), channels, self.image_height, self.image_width], dtype=np.float32)

        for i in trange(len(self)):
            image_id = self.entries[i].image_id
            image = self.load_image(image_id)
            if not self.pre_extract: image = self.preprocess_image(image)
            images[i] = image

        return images

    def preprocess_entries(self, entries):

        print("processing entries ...")
        new_entries = []

        for entry in tqdm(entries):

            entry = edict(entry)

            # tokenize text
            if self.mode == self.train:
                sbj_text = self.ent_dict.idx2sym[entry.sbj_label]
                entry.sbj_tokens, entry.sbj_seq_len = self.tokenize(sbj_text)
                obj_text = self.ent_dict.idx2sym[entry.obj_label]
                entry.obj_tokens, entry.obj_seq_len = self.tokenize(obj_text)
                pred_text = self.pred_dict.idx2sym[entry.pred_label]
                entry.pred_tokens, entry.pred_seq_len = self.tokenize(pred_text)

            # convert boxes
            entry.sbj_box = box_convert_and_normalize(entry.sbj_box, entry.width, entry.height)
            entry.obj_box = box_convert_and_normalize(entry.obj_box, entry.width, entry.height)

            # create pred_box
            entry.pred_box = box_union(entry.sbj_box, entry.obj_box)

            new_entries.append(entry)

        return new_entries

    @classmethod
    def create(cls, cfg, word_dict, ent_dict, pred_dict, triples_path, mode, preload):

        entries = pickle.load(open(triples_path, "rb"))
        return cls(cfg.dataset,
                   entries, word_dict, ent_dict, pred_dict,
                   cfg.language_model.tokens_length,
                   cfg.vision_model.image_dir,
                   cfg.vision_model.image_width,
                   cfg.vision_model.image_height,
                   mode, preload,
                   cfg.vision_model.pre_extract,
                   cfg.vision_model.cache_dir,
                   cfg.vision_model.backbone,
                   cfg.vision_model.feature_height,
                   cfg.vision_model.feature_width,
                   cfg.vision_model.feature_dim,)

