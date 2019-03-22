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

if __name__ == "__main__":

    print("generating triples ...")

    image_dir = "data/gqa/images"
    box_source = "gt"
    out_dir = "cache"
    n_preds = 311
    n_rel_max = 100000

    ent_dict_path = "cache/ent_dict.json"
    pred_dict_path = "cache/pred_dict_%d.json" % n_preds
    pred_use_prob_path = "cache/pred_use_prob_%d_max_%d.json" % (n_preds, n_rel_max)
    balanced = False

    use_none_label = True
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

        out_name = split + "_triples"
        out_name = out_name + "_%s_boxes" % box_source
        if use_none_label: out_name = out_name + "_use_none"
        out_name = out_name + "_%d_max_%d" % (n_preds, n_rel_max)
        out_name = out_name + ".pkl"
        with open(os.path.join(out_dir, out_name), "wb") as f:
            pickle.dump(entries, f)
            print("dumped to %s" % out_name)

    print("triple count: %s" % str(triple_cnt))