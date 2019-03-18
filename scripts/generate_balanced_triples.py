import sys, os
sys.path.insert(0, os.getcwd())
import json
import h5py
import random
import pickle
import numpy as np
from tqdm import tqdm, trange
from lib.data.sym_dict import SymbolDictionary

def get_entities(scene_graph, ent_dict: SymbolDictionary):

    ent_labels = []
    ent_boxes = []
    eid2idx = {}
    idx2eid = []

    for idx, (eid, entity) in enumerate(scene_graph["objects"].items()):

        eid2idx[eid] = idx
        idx2eid.append(eid)
        ent_labels.append(ent_dict.sym2idx[entity["name"]])
        ent_boxes.append((entity["x"], entity["y"], entity["x"]+entity["w"], entity["y"]+entity["h"]))

    return ent_labels, ent_boxes, eid2idx, idx2eid

def get_rel_mat(eid2idx, scene_graph, pred_dict: SymbolDictionary):

    n_entities = len(eid2idx)
    rel_mat = [ [ [] for j in range(n_entities) ] for i in range(n_entities) ]

    for sbj_id, entity in scene_graph["objects"].items():

        relations = entity["relations"]
        for relation in relations:

            obj_id = relation["object"]
            pred_label = pred_dict.sym2idx[relation["name"]]
            sbj_idx = eid2idx[sbj_id]
            obj_idx = eid2idx[obj_id]
            rel_mat[sbj_idx][obj_idx].append(pred_label)

    return rel_mat

# def compute_iou_mat(b1, b2):
#
#     x11, y11, x12, y12 = np.split(b1, 4, axis=1)
#     x21, y21, x22, y22 = np.split(b2, 4, axis=1)
#     xA = np.maximum(x11, np.transpose(x21))
#     yA = np.maximum(y11, np.transpose(y21))
#     xB = np.minimum(x12, np.transpose(x22))
#     yB = np.minimum(y12, np.transpose(y22))
#     intersection = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
#     b1_area = (x12 - x11) * (y12 - y11)
#     b2_area = (x22 - x21) * (y22 - y21)
#     union = b1_area + np.transpose(b2_area) - intersection
#     iou = intersection / union
#     return iou

# def get_roidb_from_proposals(proposals, gt_boxes, ent_labels, rel_mat, threshold = 0.5):
#
#     iou_mat = compute_iou_mat(proposals, gt_boxes)
#     max_idx = np.argmax(iou_mat, axis=1)
#     max_iou = iou_mat[ np.arange(len(max_idx)), max_idx ]
#
#     pos_proposals = np.where(max_iou >= threshold)
#     n_pos_props = len(pos_proposals)
#     pos2gt = max_idx[pos_proposals]
#
#     sbj_boxes = []
#     obj_boxes = []
#     rlp_labels = []
#
#     for i in range(n_pos_props):
#         for j in range(n_pos_props):
#             if i != j and pos2gt[i] != pos2gt[j]:
#                 sbj_gt = pos2gt[i]
#                 obj_gt = pos2gt[j]
#                 for pred_id in rel_mat[sbj_gt][obj_gt]:
#                     sbj_boxes.append(proposals[i])
#                     obj_boxes.append(proposals[j])
#                     rlp_labels.append([ ent_labels[sbj_gt], pred_id, ent_labels[obj_gt] ])
#
#     sbj_boxes = np.concatenate(sbj_boxes, axis=0)
#     obj_boxes = np.concatenate(obj_boxes, axis=0)
#     rlp_labels = np.array(rlp_labels, dtype="int32")
#
#     return sbj_boxes, obj_boxes, rlp_labels

def get_roidb_from_gt(gt_boxes, ent_labels, rel_mat, pred_use_prob, use_none_label):

    entries = []

    n_ent = len(ent_labels)
    for i in range(n_ent):
        for j in range(n_ent):
            if i != j:
                if use_none_label and len(rel_mat[i][j]) == 0:
                    if random.random() > pred_use_prob[0]: continue
                    entry = {
                        "sbj_box": gt_boxes[i],
                        "sbj_label": ent_labels[i],
                        "obj_box": gt_boxes[j],
                        "obj_label": ent_labels[j],
                        "pred_label": 0
                    }
                    entries.append(entry)
                for pred_id in rel_mat[i][j]:
                    if random.random() > pred_use_prob[pred_id]: continue
                    entry = {
                        "sbj_box": gt_boxes[i],
                        "sbj_label": ent_labels[i],
                        "obj_box": gt_boxes[j],
                        "obj_label": ent_labels[j],
                        "pred_label": pred_id
                    }
                    entries.append(entry)

    return entries

# def generate_batch_bal(labels, N_each):
#     N_total = len(labels)
#     num_batch = np.int32(N_total/N_each)
#     if N_total%N_each == 0:
#         index_box = range(N_total)
#     else:
#         index_box = np.empty(shape=[N_each*(num_batch+1)],dtype=np.int32)
#         index_box[0:N_total] = range(N_total)
#         N_rest = N_each*(num_batch+1) - N_total
#
#         unique_labels = np.unique(labels, axis = 0)
#         N_unique = len(unique_labels)
#         num_label = np.zeros([N_unique,])
#         for ii in range(N_unique):
#             num_label[ii]=np.sum(labels == unique_labels[ii])
#         prob_label = np.sum(num_label)/num_label
#         prob_label = prob_label/np.sum(prob_label)
#         index_rest = np.random.choice(N_unique, size=[N_rest,], p=prob_label)
#         for ii in range(N_rest):
#             ind = index_rest[ii]
#             ind2 = np.where(labels == unique_labels[ind])[0]
#             a = np.random.randint(len(ind2))
#             index_box[N_total+ii] = ind2[a]
#     return index_box

if __name__ == "__main__":

    image_dir = "data/gqa/images"
    box_source = "gt"
    out_dir = "cache"

    ent_dict_path = "cache/ent_dict.json"
    pred_dict_path = "cache/pred_dict_311.json"
    pred_use_prob_path = "cache/pred_use_prob_311.json"

    use_none_label = False
    random.seed(999)

    scene_graphs_dir = "data/gqa/scene_graphs"
    scene_graph_files = [ "train_sceneGraphs.json", "val_sceneGraphs.json" ]
    splits = [ "train", "val" ]
    triple_cnt = {}

    object_features_dir = "data/gqa/objects"
    n_h5s = 15

    # load dictionaries
    ent_dict = SymbolDictionary.load_from_file(ent_dict_path)
    pred_dict = SymbolDictionary.load_from_file(pred_dict_path)
    pred_use_prob = json.load(open(pred_use_prob_path))

    # load h5s
    h5_paths = [ os.path.join(object_features_dir, "gqa_objects_%d.h5" % i) for i in range(n_h5s)]
    h5_boxes = [ h5py.File(path, "r").get("bboxes") for path in h5_paths ]
    h5_lookup = json.load(open(os.path.join(object_features_dir, "gqa_objects_info.json")))

    # start

    for split, sg_file in tqdm(zip(splits, scene_graph_files)):

        entries = []
        sg_path = os.path.join(scene_graphs_dir, sg_file)
        scene_graphs = json.load(open(sg_path))
        triple_cnt[split] = 0

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            meta = h5_lookup[image_id]
            ent_labels, ent_boxes, eid2idx, idx2eid = get_entities(scene_graph, ent_dict)
            rel_mat = get_rel_mat(eid2idx, scene_graph, pred_dict)

            if box_source == "gt":
                graph_entries = get_roidb_from_gt(ent_boxes, ent_labels, rel_mat, pred_use_prob, use_none_label)
                for entry in graph_entries:
                    entry["image_id"] = image_id
                    entry["width"] = meta["width"]
                    entry["height"] = meta["height"]
            # elif box_source == "proposal":
            #     proposals = h5_boxes[meta["file"]][meta["idx"], :meta["objectsNum"], :]
            #     sbj_boxes, obj_boxes, rlp_labels = get_roidb_from_proposals(proposals, ent_boxes, ent_labels, rel_mat)
            else:
                raise Exception("invalid box source")

            entries.extend(graph_entries)
            triple_cnt[split] += len(graph_entries)

        # with open(os.path.join(out_dir, "%s_%s_triples.json" % (split, box_source)), "w") as f:
        #     json.dump(entries, f)

        if use_none_label:
            out_name = "%s_balanced_triples_use_none_label.pkl" % split
        else:
            out_name = "%s_balanced_triples.pkl" % split
        with open(os.path.join(out_dir, out_name), "wb") as f:
            pickle.dump(entries, f)

    print(triple_cnt)