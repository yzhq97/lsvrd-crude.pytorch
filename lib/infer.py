import os
import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
from lib.data.dataset import box_union
from threading import Thread
from lib.module.similarity_model import PairwiseCosineSimilarity

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

class LoaderThread(Thread):
    def __init__(self, loader, image_id, output):
        super(LoaderThread, self).__init__()
        self.loader = loader
        self.image_id = image_id
        self.output = output
    def run(self):
        self.output.append(self.loader[self.image_id])

class WriterThread(Thread):
    def __init__(self, writer, image_id, h5s, info, max_ent, rel_emb, rel_mat):
        super(WriterThread, self).__init__()
        self.writer = writer
        self.image_id = image_id
        self.h5s = h5s
        self.info = info[image_id]
        self.max_ent = max_ent
        self.rel_emb = rel_emb
        self.rel_mat = rel_mat
    def run(self):
        file_idx = self.info["file"]
        array_idx = self.info["idx"]
        data = [
            self.h5s[file_idx][array_idx, :self.max_ent, :],
            self.rel_emb, self.rel_mat]
        self.writer.put(self.image_id, data)

def infer(vision_model, all_ent_boxes, pred_emb, loader, writer, h5s, info, args, cfg):

    tasks = list(all_ent_boxes.items())
    n_tasks = len(tasks)
    similarity = PairwiseCosineSimilarity()

    with torch.no_grad():

        loaded = []
        loader_thread = LoaderThread(loader, tasks[0][0], loaded)
        loader_thread.start()
        writer_thread = None

        for task_idx in trange(n_tasks):

            image_id, ent_boxes = tasks[task_idx]
            n_ent = len(ent_boxes)

            loader_thread.join()
            feature_map = torch.tensor(loaded[0]).float().cuda()
            if task_idx + 1 < n_tasks:
                loaded = []
                loader_thread = LoaderThread(loader, tasks[task_idx+1][0], loaded)
                loader_thread.start()

            # ent_emb = vision_model.infer_ent(feature_map, torch.tensor(ent_boxes).float().cuda())

            sbj_boxes, obj_boxes, rel_boxes = get_triple_boxes(ent_boxes)
            sbj_boxes = torch.tensor(sbj_boxes).float().cuda()
            obj_boxes = torch.tensor(obj_boxes).float().cuda()
            rel_boxes = torch.tensor(rel_boxes).float().cuda()

            n_boxes = len(rel_boxes)
            n_batches = int(math.ceil(n_boxes / args.batch_size))
            rel_emb = []
            for i in range(n_batches):
                batch_sbj = sbj_boxes[i * args.batch_size: (i + 1) * args.batch_size]
                batch_obj = obj_boxes[i * args.batch_size: (i + 1) * args.batch_size]
                batch_rel = rel_boxes[i * args.batch_size: (i + 1) * args.batch_size]
                batch_rel_emb = vision_model.infer_rel(feature_map, batch_sbj, batch_obj, batch_rel)
                rel_emb.append(batch_rel_emb)
            rel_emb = torch.cat(rel_emb, dim=0)

            s = similarity(rel_emb, pred_emb)
            _, labels = s.max(dim=1)
            rel_mat = labels.reshape([n_ent, n_ent])
            rel_emb = rel_emb.reshape([n_ent, n_ent, cfg.vision_model.emb_dim])

            # ent_emb_out = np.zeros([args.max_entities, cfg.vision_model.emb_dim])
            # ent_emb_out[:n_ent, :] = ent_emb

            rel_mat_out = -np.ones([args.max_entities, args.max_entities], dtype="int32")
            rel_mat_out[:n_ent, :n_ent] = rel_mat

            rel_emb_out = np.zeros([args.max_entities, args.max_entities, cfg.vision_model.emb_dim])
            rel_emb_out[:n_ent, :n_ent, :] = rel_emb

            if writer_thread is not None: writer_thread.join()
            writer_thread = WriterThread(writer, image_id, h5s, info, args.max_entities, rel_emb_out, rel_mat_out)
            writer_thread.start()