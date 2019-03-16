from __future__ import print_function
import os
import cv2
import json
import h5py
import torch
import pickle
import numpy as np
from tqdm import tqdm, trange
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from lib.data.sym_dict import SymbolDictionary


class GQATripleDataset(Dataset):

    resnet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    resnet_std = np.array([0.229, 0.224, 0.224]).reshape(3, 1, 1)

    def __init__(self, word_dict, ent_dict, pred_dict, max_tokens, entries, image_dir, image_size, preload):
        """
        :param entries: (subject, object, predicate ) entries. see scripts/generate_balanced_triples.py for details
        :param ent_dict: entity dictionary
        :param pred_dict: predicate dictionary
        :param word_dict: language word dictionary
        :param image_dir: folder containing all gqa images
        :param image_size: unified image size to feed into resnet101, represented as (width, height)
        :param preload: whether or not to load all images into memory for faster data loading
        """

        self.word_dict = word_dict
        self.ent_dict = ent_dict
        self.pred_dict = pred_dict
        self.max_tokens = max_tokens

        self.entries = self.preprocess_entries(entries)
        self.image_dir = image_dir
        self.image_size = image_size
        self.preload = preload
        self.images = self.preload_images() if preload else None

    def __len__(self):

        return len(self.entries)

    def __getitem__(self, idx):

        entry = self.entries[idx]
        image_id = entry.image_id
        image_size = (entry.width, entry.height)
        sbj_tokens = self.word_dict.tokenize

        if self.preload: image = self.images[idx]
        else: image = self.preprocess_image(self.load_image(image_id))

        return image_id, image,

    def load_image(self, image_id):

        image_path = os.path.join(self.image_dir, "%s.jpg" % image_id)
        image = cv2.imread(image_path)
        if image is None: raise Exception("image %s does not exist" % image_path)
        return image

    def preprocess_image(self, image):

        h, w, c = image.shape
        if c == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # convert to RGB
        else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.image_size) # unify size
        image = image.transpose([2, 0, 1]) # HWC to CHW
        image = image.astype(np.float16) # convert to float
        image = (image / 255.0 - self.resnet_mean) / self.resnet_std # normalize

        return image

    def preload_images(self):

        print("loading images into memory ...")
        images = np.zeros([len(self)] + self.image_size, dtype=np.float16)

        for i in trange(len(self)):
            image_id = self.entries[i].image_id
            image = self.load_image(image_id)
            image = self.preprocess_image(image)
            images[i] = image

        return images

    def preprocess_entries(self, entries):

        print("processing entries ...")
        new_entries = []

        for entry in tqdm(entries):
            entry = edict(entry)
            sbj_text = self.ent_dict.idx2sym[entry.sbj_label]
            entry.sbj_tokens = self.word_dict.tokenize(sbj_text)[:self.max_tokens]
            obj_text = self.ent_dict.idx2sym[entry.obj_label]
            entry.obj_tokens = self.word_dict.tokenize(obj_text)[:self.max_tokens]
            pred_text

        return new_entries

    @classmethod
    def load_from_cfgs(cls, cfgs):

        entries = pickle.load(open(cfgs.triples_path, "rb"))
        word_dict = SymbolDictionary.load_from_file(cfgs.word_dict_path)
        ent_dict = SymbolDictionary.load_from_file(cfgs.ent_dict_path)
        pred_dict = SymbolDictionary.load_from_file(cfgs.pred_dict_path)
        return cls(word_dict, ent_dict, pred_dict, cfgs.max_tokens, entries, cfgs.image_dir, cfgs.image_size, cfgs.preload)