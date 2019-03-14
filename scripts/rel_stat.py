import sys, os
sys.path.insert(0, os.getcwd())
import json
import h5py
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

    ent_labels = np.array(ent_labels, dtype="int32")
    ent_boxes = np.array(ent_boxes, dtype="float32")

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

if __name__ == "__main__":

    n_pred_use = 40
    n_rel_max = 50000

    ent_dict_path = "data/gqa/vrd/ent_dict.json"
    pred_dict_path = "data/gqa/vrd/pred_dict.json"

    scene_graphs_dir = "data/gqa/scene_graphs"
    scene_graph_files = [ "train_sceneGraphs.json", "val_sceneGraphs.json" ]

    object_features_dir = "data/gqa/objects"
    n_h5s = 15

    out_dir = "data/gqa/vrd/vtranse"

    # load dictionaries
    ent_dict = SymbolDictionary.load_from_file(ent_dict_path)
    pred_dict = SymbolDictionary.load_from_file(pred_dict_path)

    rel_cnt = [ 0 ] * len(pred_dict)
    ent_cnt = [ 0 ] * len(ent_dict)

    # start
    for sg_file in tqdm(scene_graph_files):

        sg_path = os.path.join(scene_graphs_dir, sg_file)
        scene_graphs = json.load(open(sg_path))

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            ent_labels, ent_boxes, eid2idx, idx2eid = get_entities(scene_graph, ent_dict)
            rel_mat = get_rel_mat(eid2idx, scene_graph, pred_dict)

            for label in ent_labels:
                ent_cnt[label] += 1

            n_ent = len(idx2eid)
            for i in range(n_ent):
                for j in range(n_ent):
                    if i != j:
                        if len(rel_mat[i][j]) == 0:
                            rel_cnt[0] += 1
                        else:
                            for pred_id in rel_mat[i][j]:
                                rel_cnt[pred_id] += 1

    pred_ids = [ _ for _ in range(len(pred_dict)) ]
    rel_sum = sum(rel_cnt) - rel_cnt[0]
    rel_portion = [ 1.0 * n / rel_sum for n in rel_cnt ]
    rel_use =  [ n if n <= n_rel_max else n_rel_max for n in rel_cnt ]
    rel_prob = [ 1.0*rel_use[i]/rel_cnt[i] for i in range(len(pred_dict)) ]
    rel_cnt = zip(pred_ids, rel_cnt, rel_portion, rel_use, rel_prob)
    rel_cnt = sorted(rel_cnt, key=lambda x: x[1], reverse=True)

    for i in range(len(pred_dict)):
        pred_id, cnt, portion, rel_use, rel_prob = rel_cnt[i]
        print("%3d | %3d %20s %10d %.4f" % (i+1, pred_id, pred_dict.idx2sym[pred_id], cnt, 100 * portion))

    pred_dict_use = SymbolDictionary()
    pred_use_prob = []
    for i in range(1, n_pred_use+1):
        pred_id, cnt, portion, rel_use, rel_prob = rel_cnt[i]
        pred_name = pred_dict.idx2sym[pred_id]
        pred_dict_use.add_sym(pred_name)
        pred_use_prob.append(rel_prob)
    pred_dict_use.dump_to_file("data/gqa/vrd/pred_dict_%d.json" % n_pred_use)

    with open("data/gqa/vrd/pred_use_prob.json", "w") as f:
        json.dump(pred_use_prob, f)