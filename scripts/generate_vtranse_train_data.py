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
    rel_mat = np.zeros([n_entities, n_entities], dtype="int32")

    for sbj_id, entity in scene_graph["objects"].items():

        relations = entity["relations"]
        for rel_id, relation in relations.items():

            obj_id = relation["object"]
            pred_label = pred_dict.sym2idx[relation["name"]]
            sbj_idx = eid2idx[sbj_id]
            obj_idx = eid2idx[obj_id]
            rel_mat[sbj_idx][obj_idx] = pred_label

    return rel_mat

def compute_iou_mat(b1, b2):
    x11, y11, x12, y12 = np.split(b1, 4, axis=1)
    x21, y21, x22, y22 = np.split(b2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    intersection = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    b1_area = (x12 - x11) * (y12 - y11)
    b2_area = (x22 - x21) * (y22 - y21)
    union = b1_area + np.transpose(b2_area) - intersection
    iou = intersection / union
    return iou

def get_roidb(proposals, gt_boxes, ent_labels, rel_mat, threshold = 0.5):
    iou_mat = compute_iou_mat(proposals, gt_boxes)
    max_idx = np.argmax(iou_mat, axis=1)
    max_iou = iou_mat[ np.arange(len(max_idx)), max_idx ]
    proposal_gts = np.array(max_idx)
    proposal_gts[max_iou < threshold] = -1
    proposal_labels = ent_labels[max_idx]
    proposal_labels[max_iou < threshold] = 0
    # TODO: generate roidb


if __name__ == "__main__":

    ent_dict_path = "data/gqa/vrd/ent_dict.json"
    pred_dict_path = "data/gqa/vrd/pred_dict.json"

    scene_graphs_dir = "data/gqa/scene_graphs"
    scene_graph_files = [ "train_sceneGraphs.json", "val_sceneGraphs" ]

    object_features_dir = "data/gqa/objects"
    n_h5s = 15

    out_dir = "data/gqa/vrd/vtranse"

    # load dictionaries
    ent_dict = SymbolDictionary.load_from_file(ent_dict_path)
    pred_dict = SymbolDictionary.load_from_file(pred_dict_path)

    # load h5s
    h5_paths = [ os.path.join(object_features_dir, "gqa_objects_%d.h5" % i) for i in range(n_h5s)]
    h5_boxes = [ h5py.File(path, "r").get("bboxes") for path in h5_paths ]
    h5_lookup = json.load(open(os.path.join(object_features_dir, "gqa_objects_info.json")))

    # start
    for sg_file in tqdm(scene_graph_files):

        sg_path = os.path.join(scene_graphs_dir, sg_file)
        scene_graphs = json.load(open(sg_path))

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            ent_labels, ent_boxes, eid2idx, idx2eid = get_entities(scene_graph, ent_dict)
            rel_mat = get_rel_mat(eid2idx, scene_graph, pred_dict)
            h5_index = h5_lookup[image_id]
            proposals = h5_boxes[h5_index["file"]][h5_index["idx"], :h5_index["objectsNum"], :]

            # TODO: generate ROIDB

