import sys, os
sys.path.insert(0, os.getcwd())
import json
import h5py
import time
import random
import numpy as np
from tqdm import tqdm, trange
from lib.data.sym_dict import SymbolDictionary

def get_entities(scene_graph, ent_dict: SymbolDictionary):

    ent_labels = []
    ent_boxes = []
    eid2idx = {}
    idx2eid = []

    for idx, (eid, entity) in enumerate(scene_graph["objects"].items()):
        if entity["name"] not in ent_dict: continue
        eid2idx[eid] = len(idx2eid)
        idx2eid.append(eid)
        ent_labels.append(ent_dict.sym2idx[entity["name"]])
        ent_boxes.append((entity["x"], entity["y"], entity["x"]+entity["w"], entity["y"]+entity["h"]))

    ent_labels = np.array(ent_labels, dtype="int32")
    ent_boxes = np.array(ent_boxes, dtype="float32")

    return ent_labels, ent_boxes, eid2idx, idx2eid

def get_flipped_entry(entry):
    flipped_entry = entry
    flipped_entry["flipped"] = True
    boxes_old = entry['boxes']
    width = entry['width']
    boxes_new = np.zeros(np.shape(boxes_old))
    boxes_new[:, 0] = width - boxes_old[:, 2]
    boxes_new[:, 1] = boxes_old[:, 1]
    boxes_new[:, 2] = width - boxes_old[:, 0]
    boxes_new[:, 3] = boxes_old[:, 3]
    flipped_entry['boxes'] = boxes_new
    return flipped_entry

if __name__ == "__main__":

    image_dir = "/db/data/gqa11/images"
    roidb_keys = ["train_roidb", "test_roidb"]

    ent_dict_path = "data/gqa/vrd/ent_dict_400.json"

    scene_graphs_dir = "data/gqa/scene_graphs"
    scene_graph_files = [ "train_sceneGraphs.json", "val_sceneGraphs.json" ]

    object_features_dir = "data/gqa/objects"
    h5_lookup = json.load(open(os.path.join(object_features_dir, "gqa_objects_info.json")))

    # load dictionaries
    ent_dict = SymbolDictionary.load_from_file(ent_dict_path)

    n_ent = len(ent_dict)

    out_path = "data/gqa/vrd/vtranse/input/gqa_detection_%d_roidb.npz" % (n_ent)

    # start

    roidb = {}
    n_boxes = {}

    for roidb_key, sg_file in tqdm(zip(roidb_keys, scene_graph_files)):

        entries = []
        sg_path = os.path.join(scene_graphs_dir, sg_file)
        scene_graphs = json.load(open(sg_path))
        n_boxes[roidb_key] = 0

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            meta = h5_lookup[image_id]
            ent_labels, ent_boxes, eid2idx, idx2eid = get_entities(scene_graph, ent_dict)
            n_boxes[roidb_key] += len(ent_labels)

            if len(ent_boxes) == 0: continue

            entry = {
                "image": os.path.join(image_dir, "%s.jpg" % image_id),
                "width": meta["width"],
                "height": meta["height"],
                "boxes": ent_boxes,
                "gt_classes": ent_labels,
                "max_overlaps": np.ones(np.shape(ent_labels)),
                "flipped": False
            }

            entries.append(entry)

        if roidb_key == "train_roidb":
            flipped_entries = []
            for entry in tqdm(entries):
                flipped_entries.append(get_flipped_entry(entry))
            entries.extend(flipped_entries)

        roidb[roidb_key] = entries

    np.savez(out_path, roidb=roidb)

    print(n_boxes)
    print("dumped to %s" % out_path)