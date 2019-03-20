import os
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.utils import Logger
from lib.module.similarity_model import PairwiseCosineSimilarity

def get_sym_emb(word_emb, language_model, word_dict, sym_dict, tokens_length, batch_size=32):

    sym_tokens = [word_dict.tokenize(sym)[:tokens_length] for sym in sym_dict.idx2sym]
    sym_tokens = [tokens + [len(word_dict)] * (tokens_length - len(tokens)) for tokens in sym_tokens]
    sym_tokens = torch.tensor(sym_tokens)

    n_syms = len(sym_tokens)
    n_batches = int(math.ceil(1.0 * n_syms / batch_size))

    sym_embs = []

    for batch in range(n_batches):
        batch_tokens = sym_tokens[batch * batch_size: (batch + 1) * batch_size]
        batch_tokens = batch_tokens.cuda()
        batch_w_embs = word_emb(batch_tokens)
        batch_embs = language_model(batch_w_embs)
        sym_embs.append(batch_embs)

    sym_embs = torch.cat(sym_embs, 0)

    return sym_embs


def topk_match(k, x_emb, label_emb, x_labels, similarity):

    s = similarity(x_emb, label_emb) # [ N, n_labels ]
    _, top_predictions = s.topk(k, largest=True, sorted=False)
    matches = torch.eq(x_labels, top_predictions)
    matches, _ = matches.max(dim=1)

    return matches


def accuracy(vision_model, loader, ent_embs, pred_embs, k=3):

    n_batches = len(loader)
    similarity = PairwiseCosineSimilarity()

    ent_matches = []
    rel_matches = []

    for i, data in enumerate(loader):

        print("evaluating batch %4d/%4d" % (i+1, n_batches), end="\r")

        images = data[1].float().cuda()
        sbj_boxes = data[2].float().cuda()
        obj_boxes = data[3].float().cuda()
        rel_boxes = data[4].float().cuda()
        sbj_labels = data[5].cuda()
        obj_labels = data[6].cuda()
        rel_labels = data[7].cuda()

        sbj_embs, obj_embs, rel_embs = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)

        batch_sbj_matches = topk_match(k, sbj_embs, ent_embs, sbj_labels, similarity)
        batch_obj_matches = topk_match(k, obj_embs, ent_embs, obj_labels, similarity)
        batch_rel_matches = topk_match(k, rel_embs, pred_embs, rel_labels, similarity)

        ent_matches.append(batch_sbj_matches)
        ent_matches.append(batch_obj_matches)
        rel_matches.append(batch_rel_matches)

    ent_matches = torch.cat(ent_matches, 0)
    rel_matches = torch.cat(rel_matches, 0)

    ent_acc = ent_matches.mean().item()
    rel_acc = rel_matches.mean().item()

    return ent_acc, rel_acc

