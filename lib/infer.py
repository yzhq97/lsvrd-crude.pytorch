import os
import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from lib.data.dataset import box_union

def get_triple_boxes(boxes):

    sbj_boxes = []
    obj_boxes = []
    rel_boxes = []

    n_boxes = len(boxes)
    for i in range(n_boxes):
        for j in range(n_boxes):
            sbj_boxes.append([boxes[i]])
            obj_boxes.append([boxes[j]])
            rel_boxes.append(box_union(boxes[i], boxes[j]))

    return sbj_boxes, obj_boxes, rel_boxes

def infer(vision_model, all_ent_boxes, loader, writer, args, cfg):

    for image_id, ent_boxes in tqdm(all_ent_boxes.items()):

        n_ent = len(ent_boxes)

        feature_map = torch.tensor(loader[image_id]).float().cuda()
        ent_embs = vision_model.infer_ent(feature_map, torch.tensor(ent_boxes).float().cuda())
        ent_embs = ent_embs.data.cpu().numpy()

        sbj_boxes, obj_boxes, rel_boxes = get_triple_boxes(ent_boxes)
        sbj_boxes = torch.tensor(sbj_boxes).float().cuda()
        obj_boxes = torch.tensor(obj_boxes).float().cuda()
        rel_boxes = torch.tensor(rel_boxes).float().cuda()

        n_boxes = len(rel_boxes)
        n_batches = int(math.ceil(n_boxes / args.batch_size))
        rel_embs = []
        for i in range(n_batches):
            batch_sbj = sbj_boxes[i * args.batch_size: (i + 1) * args.batch_size]
            batch_obj = obj_boxes[i * args.batch_size: (i + 1) * args.batch_size]
            batch_rel = rel_boxes[i * args.batch_size: (i + 1) * args.batch_size]
            batch_rel_embs = vision_model.infer_rel(feature_map, batch_sbj, batch_obj, batch_rel)
            rel_embs.append(batch_rel_embs.data.cpu().numpy())
        rel_embs = np.stack(rel_embs).reshape([n_ent, n_ent, cfg.vision_model.emb_dim])

        ent_embs_out = np.zeros([args.max_entities, cfg.vision_model.emb_dim])
        rel_embs_out = np.zeros([args.max_entities, args.max_entities, cfg.vision_model.emb_dim])
        ent_embs_out[:n_ent, :] = ent_embs
        rel_embs_out[:n_ent, :n_ent, :] = rel_embs

        writer.put(image_id, [ent_embs_out, rel_embs_out])