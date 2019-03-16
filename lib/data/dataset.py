import sys, os
sys.path.insert(0, os.getcwd())
import cv2
import time
import json
import h5py
import torch
import pickle
import numpy as np
from tqdm import tqdm, trange
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from lib.data.sym_dict import SymbolDictionary


def get_pos_mat(labels):
    N = labels.size(0)
    labels_0 = labels.unsqueeze(0).repeat(N, 1)
    labels_1 = labels.unsqueeze(1).repeat(1, N)
    pos_mat = labels_0 == labels_1
    return pos_mat

def box_union(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    return (min(xa1, xb1), min(ya1, yb1), max(xa2, xb2), max(ya2, yb2))

class GQATriplesDataset(Dataset):

    resnet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    resnet_std = np.array([0.229, 0.224, 0.224]).reshape(3, 1, 1)

    def __init__(self, word_dict, ent_dict, pred_dict, tokens_length,
                 entries, image_dir, image_width, image_height, preload):
        """
        :param entries: (subject, object, predicate ) entries. see scripts/generate_balanced_triples.py for details
        :param ent_dict: entity dictionary
        :param pred_dict: predicate dictionary
        :param word_dict: language word dictionary
        :param image_dir: folder containing all gqa images
        :param preload: whether or not to load all images into memory for faster data loading
        """

        self.word_dict = word_dict
        self.ent_dict = ent_dict
        self.pred_dict = pred_dict
        self.tokens_length = tokens_length

        self.entries = self.preprocess_entries(entries)
        self.image_dir = image_dir
        self.image_width = image_width
        self.image_height = image_height
        self.preload = preload
        self.images = self.preload_images() if preload else None

    def __len__(self):

        return len(self.entries)

    def __getitem__(self, idx):

        entry = self.entries[idx]
        image_id = entry.image_id
        image_size = (entry.width, entry.height)

        if self.preload: image = self.images[idx]
        else: image = self.preprocess_image(self.load_image(image_id))

        ret = (
            image_id, image, torch.tensor(image_size),
            torch.tensor(entry.sbj_box), torch.tensor(entry.obj_box), torch.tensor(entry.pred_box),
            entry.sbj_label, entry.obj_label, entry.pred_label,
            torch.tensor(entry.sbj_tokens), torch.tensor(entry.obj_tokens), torch.tensor(entry.pred_tokens)
        )

        return ret

    def load_image(self, image_id):

        image_path = os.path.join(self.image_dir, "%s.jpg" % image_id)
        image = cv2.imread(image_path)
        if image is None: raise Exception("image %s does not exist" % image_path)
        return image

    def tokenize(self, text):
        tokens = self.word_dict.tokenize(text)[:self.tokens_length]
        tokens = tokens + [ 0 ] * (self.tokens_length - len(tokens))
        return tokens

    def preprocess_image(self, image):

        h, w, c = image.shape
        if c == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # convert to RGB
        else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_width, self.image_height)) # unify size
        image = image.transpose([2, 0, 1]) # HWC to CHW
        image = image.astype(np.float32) # convert to float
        image = (image / 255.0 - self.resnet_mean) / self.resnet_std # normalize

        return image

    def preload_images(self):

        print("loading images into memory ...")
        images = np.zeros([len(self), 3, self.image_height, self.image_width], dtype=np.float32)

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
            entry.sbj_tokens = self.tokenize(sbj_text)
            obj_text = self.ent_dict.idx2sym[entry.obj_label]
            entry.obj_tokens = self.tokenize(obj_text)
            pred_text = self.pred_dict.idx2sym[entry.pred_label]
            entry.pred_tokens = self.tokenize(pred_text)
            entry.pred_box = box_union(entry.sbj_box, entry.obj_box)
            new_entries.append(entry)

        return new_entries

    @classmethod
    def load_from_cfgs(cls, cfgs):

        entries = pickle.load(open(cfgs.triples_path, "rb"))
        word_dict = SymbolDictionary.load_from_file(cfgs.word_dict_path)
        ent_dict = SymbolDictionary.load_from_file(cfgs.ent_dict_path)
        pred_dict = SymbolDictionary.load_from_file(cfgs.pred_dict_path)
        return cls(word_dict, ent_dict, pred_dict, cfgs.tokens_length,
                   entries, cfgs.image_dir, cfgs.image_width, cfgs.image_height, cfgs.preload)


if __name__ == "__main__":
    """
    a unit test for GQATriplesDataset
    """

    print("cwd: %s" % os.getcwd())

    cfgs = edict()

    cfgs.triples_path = "data/gqa/vrd/val_balanced_triples.pkl"
    cfgs.word_dict_path = "data/gqa/vrd/word_dict.json"
    cfgs.ent_dict_path = "data/gqa/vrd/ent_dict.json"
    cfgs.pred_dict_path = "data/gqa/vrd/pred_dict_311.json"

    cfgs.tokens_length = 5

    cfgs.image_dir = "data/gqa/images"
    cfgs.image_width = 448
    cfgs.image_height = 448
    cfgs.preload = False

    cfgs.batch_size = 32

    dataset = GQATriplesDataset.load_from_cfgs(cfgs)
    dataloader = DataLoader(dataset=dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True)

    tic_0 = time.time()
    for i, data in enumerate(dataloader):
        data = [ d.cuda() if isinstance(d, torch.Tensor) else d for d in data ]
        tic_1 = time.time()
        elapsed = tic_1 - tic_0
        print("batch_size %d, %dms per batch, %dms per sample" %
              (cfgs.batch_size, int(1000 * elapsed), int(1000 * elapsed / cfgs.batch_size)))
        tic_0 = time.time()