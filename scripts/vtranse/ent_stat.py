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

if __name__ == "__main__":

    n_ent_use = 400

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

    ent_cnt = [ 0 ] * len(ent_dict)

    # start
    for sg_file in tqdm(scene_graph_files):

        sg_path = os.path.join(scene_graphs_dir, sg_file)
        scene_graphs = json.load(open(sg_path))

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            ent_labels, ent_boxes, eid2idx, idx2eid = get_entities(scene_graph, ent_dict)

            for label in ent_labels:
                ent_cnt[label] += 1

    ent_ids = [ _ for _ in range(len(ent_dict)) ]
    ent_sum = sum(ent_cnt)
    ent_portion = [ 1.0 * n / ent_sum for n in ent_cnt ]
    ent_cnt = zip(ent_ids, ent_cnt, ent_portion)
    ent_cnt = sorted(ent_cnt, key=lambda x: x[1], reverse=True)

    print("entity stats:")
    for i in range(len(ent_dict)):
        ent_id, cnt, portion = ent_cnt[i]
        print("%4d | %4d %20s %10d %.4f" % (i+1, ent_id, ent_dict.idx2sym[ent_id], cnt, 100 * portion))

    ent_dict_use = SymbolDictionary()
    for i in range(n_ent_use):
        ent_id, cnt, portion = ent_cnt[i]
        ent_name = ent_dict.idx2sym[ent_id]
        ent_dict_use.add_sym(ent_name)
    ent_dict_use.dump_to_file("data/gqa/vrd/ent_dict_%d.json" % n_ent_use)