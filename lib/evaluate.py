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

def compute_matches(predictions, labels):
    labels = labels.unsqueeze(1).repeat(1, predictions.size(1))
    matches = torch.eq(predictions, labels)
    matches, _ = matches.max(dim=1)
    return matches


def accuracy(vision_model, loader, ent_t_embs, pred_embs, tfb_logger, step, k_ent=20, k_rel=20):

    n_batches = len(loader)
    similarity = PairwiseCosineSimilarity()

    ent_predictions = []
    rel_predictions = []
    ent_matches = []
    rel_matches = []

    n_ent_labels = ent_t_embs.size(0)
    n_preds = pred_embs.size(0)
    for i, data in enumerate(loader):

        print("evaluating batch %4d/%4d" % (i+1, n_batches), end="\r")

        images = data[1].cuda().float()
        sbj_boxes = data[2].cuda().float()
        obj_boxes = data[3].cuda().float()
        rel_boxes = data[4].cuda().float()
        sbj_labels = data[5].cuda()
        obj_labels = data[6].cuda()
        rel_labels = data[7].cuda()

        sbj_v_embs, obj_v_embs, rel_v_embs = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)

        batch_sbj_predictions = topk_predictions(k_ent, sbj_v_embs, ent_t_embs, similarity)
        batch_obj_predictions = topk_predictions(k_ent, obj_v_embs, ent_t_embs, similarity)
        batch_rel_predictions = topk_predictions(k_rel, rel_v_embs, pred_embs, similarity)

        ent_predictions.append(batch_sbj_predictions)
        ent_predictions.append(batch_obj_predictions)
        rel_predictions.append(batch_rel_predictions)

        batch_sbj_matches = compute_matches(batch_sbj_predictions, sbj_labels)
        batch_obj_matches = compute_matches(batch_obj_predictions, obj_labels)
        batch_rel_matches = compute_matches(batch_rel_predictions, rel_labels)

        ent_matches.append(batch_sbj_matches)
        ent_matches.append(batch_obj_matches)
        rel_matches.append(batch_rel_matches)

    ent_matches = torch.cat(ent_matches, 0)
    rel_matches = torch.cat(rel_matches, 0)

    ent_acc = ent_matches.float().mean().item()
    rel_acc = rel_matches.float().mean().item()

    tfb_logger.scalar_summary("acc/ent(top%d)" % k_ent, ent_acc, step)
    tfb_logger.scalar_summary("acc/rel(top%d)" % k_rel, rel_acc, step)

    ent_predictions = torch.cat(ent_predictions, dim=0)
    rel_predictions = torch.cat(rel_predictions, dim=0)
    tfb_logger.histo_summary("predictions/ent", ent_predictions.data.cpu().numpy(), step)
    tfb_logger.histo_summary("predictions/rel", rel_predictions.data.cpu().numpy(), step)

    return ent_acc, rel_acc

