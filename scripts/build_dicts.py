import sys, os
sys.path.insert(0, os.getcwd())
import json
import numpy as np
from tqdm import tqdm, trange
from lib.data.sym_dict import SymbolDictionary

def create_glove_emb(sym_dict, glove_txt_path, emb_dim):
    w_emb = np.zeros([len(sym_dict), emb_dim])
    with open(glove_txt_path) as f:
        for line in tqdm(f.readlines()):
            elems = line.split(' ')
            word = elems[0]
            if word in sym_dict:
                idx = sym_dict.sym2idx[word]
                vec = [float(_) for _ in elems[1:]]
                w_emb[idx] = vec
    return w_emb


scence_graphs_dir = "data/gqa/scene_graphs"
glove_txt_path = "data/glove/glove.6B.300d.txt"
emb_dim = 300
files = [
    "train_sceneGraphs.json",
    "val_sceneGraphs.json"
]

if __name__ == "__main__":

    # get glove words
    print("parsing glove words")
    glove_words = []
    with open(glove_txt_path) as f:
        for line in tqdm(f.readlines()):
            elems = line.split(' ')
            word = elems[0]
            glove_words.append(word)

    # get dictionary

    inst_dict = SymbolDictionary()
    inst_dict.add_sym("unknown")
    inst_cnt = {}

    pred_dict = SymbolDictionary()
    pred_dict.add_sym("none")
    pred_dict.add_sym("is")
    pred_cnt = {}

    print("running statistics ...")

    for file in tqdm(files):

        path = os.path.join(scence_graphs_dir, file)
        scene_graphs = json.load(open(path))

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            instances = scene_graph["objects"]

            for inst_id, inst in instances.items():

                if inst["name"] in inst_cnt:
                    inst_cnt[inst["name"]] += 1
                else:
                    inst_cnt[inst["name"]] = 1

                rels = inst["relations"]

                for rel in rels:

                    if rel["name"] in pred_cnt:
                        pred_cnt[rel["name"]] += 1
                    else:
                        pred_cnt[rel["name"]] = 1

    print("adding symbols ...")

    for inst_name, cnt in tqdm(inst_cnt.items()):
        if inst_name in glove_words:
            inst_dict.add_sym(inst_name)

    for pred_name, cnt in tqdm(pred_cnt.items()):
        if pred_name in glove_words:
            inst_dict.add_sym(pred_name)

    inst_dict.dump_to_file("data/gqa/inst_dict.json")
    pred_dict.dump_to_file("data/gqa/pred_dict.json")
    print("saving inst embeddings...")
    inst_emb = create_glove_emb(inst_dict, glove_txt_path, 300)
    np.save("data/gqa/inst_emb.npy", inst_emb)
    print("saving pred embeddings...")
    pred_emb = create_glove_emb(pred_dict, glove_txt_path, 300)
    np.save("data/gqa/pred_emb.npy", pred_emb)

    print("%d instance categories, %d initialized" % (len(inst_cnt), len(inst_dict)))
    print("%d predicates, %d initialized" % (len(pred_cnt), len(pred_dict)))

