import os
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.utils import Logger
from lib.module.similarity_model import PairwiseCosineSimilarity

def get_sym_emb(language_model, word_dict, sym_dict, tokens_length, batch_size = 32):

    sym_tokens = [word_dict.tokenize(sym)[:tokens_length] for sym in sym_dict.idx2sym]
    sym_tokens = [tokens + [0] * (tokens_length - len(tokens)) for tokens in sym_tokens]
    sym_tokens = torch.tensor(sym_tokens)

    n_syms = len(sym_tokens)
    n_batches = int(math.ceil(1.0 * n_syms / batch_size))

    sym_embs = []

    for batch in range(n_batches):
        batch_tokens = sym_tokens[batch * batch_size: (batch + 1) * batch_size]
        with torch.cuda.device(0):
            batch_tokens = batch_tokens.cuda()
        batch_embs = language_model(batch_tokens)
        sym_embs.append(batch_embs)

    sym_embs = torch.cat(sym_embs, 0)

    return sym_embs

def topk_acc(k, x_emb, label_emb, x_labels, similarity):

    s = similarity(x_emb, label_emb) # [ N, n_labels ]
    _, top_labels = s.topk(k, largest=True, sorted=False)



def evaluate(vision_model, loader, ent_embs, pred_embs):

    n_batches = len(loader)
    similarity = PairwiseCosineSimilarity()

    for i, data in enumerate(loader):

        with torch.cuda.device(0):
            data = [item.cuda() for item in data]

        image_ids, images, \
        sbj_boxes, obj_boxes, rel_boxes, \
        sbj_labels, obj_labels, rel_labels = data

        sbj_embs, obj_embs, rel_embs = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)


