import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from lib.utils import Logger
from lib.module.similarity_model import PairwiseCosineSimilarity

def get_sym_emb(word_emb, language_model, word_dict, sym_dict, tokens_length, batch_size=32):

    sym_tokens = [word_dict.tokenize(sym)[:tokens_length] for sym in sym_dict.idx2sym]
    seq_lens = [len(tokens) for tokens in sym_tokens]
    sym_tokens = [tokens + [len(word_dict)] * (tokens_length - len(tokens)) for tokens in sym_tokens]
    sym_tokens = torch.tensor(sym_tokens)
    seq_lens = torch.tensor(seq_lens).long()

    n_syms = len(sym_tokens)
    n_batches = int(math.ceil(1.0 * n_syms / batch_size))

    sym_embs = []

    for batch in range(n_batches):
        batch_tokens = sym_tokens[batch * batch_size: (batch + 1) * batch_size].cuda()
        batch_seq_lens = seq_lens[batch * batch_size: (batch + 1) * batch_size].cuda()
        batch_w_embs = word_emb(batch_tokens)
        batch_embs = language_model(batch_w_embs, batch_seq_lens)
        sym_embs.append(batch_embs)

    sym_embs = torch.cat(sym_embs, 0)

    return sym_embs


def topk_predictions(k, x_emb, label_emb, similarity):

    s = similarity(x_emb, label_emb) # [ N, n_labels ]
    score = F.softmax(s, dim=1)
    _, top_predictions = score.topk(k, sorted=False)

    return top_predictions

def matches(predictions, labels):
    labels = labels.unsqueeze(1).repeat(1, predictions.size(1))
    matches = torch.eq(predictions, labels)
    _, matches = matches.max(dim=1)
    return matches


def accuracy(vision_model, loader, ent_t_embs, pred_embs, k_ent=50, k_rel=20):

    n_batches = len(loader)
    similarity = PairwiseCosineSimilarity()

    ent_matches = []
    rel_matches = []

    n_ent_labels = ent_t_embs.size(0)
    n_preds = pred_embs.size(0)
    ent_bincounts = np.zeros([n_ent_labels], dtype=np.int32)
    rel_bincounts = np.zeros([n_preds], dtype=np.int32)

    for i, data in enumerate(loader):

        print("evaluating batch %4d/%4d" % (i+1, n_batches), end="\r")

        images = data[1].float().cuda()
        sbj_boxes = data[2].float().cuda()
        obj_boxes = data[3].float().cuda()
        rel_boxes = data[4].float().cuda()
        sbj_labels = data[5].cuda()
        obj_labels = data[6].cuda()
        rel_labels = data[7].cuda()

        sbj_v_embs, obj_v_embs, rel_v_embs = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)

        batch_sbj_predictions = topk_predictions(k_ent, sbj_v_embs, ent_t_embs, similarity)
        batch_obj_predictions = topk_predictions(k_ent, obj_v_embs, ent_t_embs, similarity)
        batch_rel_predictions = topk_predictions(k_rel, rel_v_embs, pred_embs, similarity)

        ent_bincounts += np.bincount(batch_sbj_predictions.view(-1).cpu().numpy(), minlength=n_ent_labels)
        ent_bincounts += np.bincount(batch_obj_predictions.view(-1).cpu().numpy(), minlength=n_ent_labels)
        rel_bincounts += np.bincount(batch_rel_predictions.view(-1).cpu().numpy(), minlength=n_preds)

        batch_sbj_matches = matches(batch_sbj_predictions, sbj_labels)
        batch_obj_matches = matches(batch_obj_predictions, obj_labels)
        batch_rel_matches = matches(batch_rel_predictions, rel_labels)

        ent_matches.append(batch_sbj_matches)
        ent_matches.append(batch_obj_matches)
        rel_matches.append(batch_rel_matches)

    ent_matches = torch.cat(ent_matches, 0)
    rel_matches = torch.cat(rel_matches, 0)

    ent_acc = ent_matches.float().mean().item()
    rel_acc = rel_matches.float().mean().item()

    ent_distribution_std = ent_bincounts.std().item()
    rel_distribution_std = rel_bincounts.std().item()


    return ent_acc, rel_acc, ent_distribution_std, rel_distribution_std

