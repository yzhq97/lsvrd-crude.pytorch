import sys, os
sys.path.insert(0, os.getcwd())
import json
import numpy as np
from tqdm import tqdm, trange
from lib.data.sym_dict import SymbolDictionary

def create_glove_emb(sym_dict, glove_words, glove_emb):
    _, emb_dim = glove_emb.shape
    uninit = []
    w_emb = np.zeros([len(sym_dict), emb_dim])
    for sym, idx in sym_dict.sym2idx.items():
        if sym in glove_words:
            glove_idx = glove_words.index(sym)
            w_emb[idx] = glove_emb[glove_idx]
        else:
            glove_idx = glove_words.index('unknown')
            uninit.append(sym)
            w_emb[idx] = glove_emb[glove_idx]

    return w_emb, uninit

if __name__ == "__main__":

    scence_graphs_dir = "data/gqa/scene_graphs"
    glove_txt_path = "data/glove/glove.6B.300d.txt"
    emb_dim = 300
    files = [
        "train_sceneGraphs.json",
        "val_sceneGraphs.json"
    ]

    # get dictionary

    word_dict = SymbolDictionary()
    word_dict.add_sym("unknown")
    len_cnt = [0] * 8

    ent_dict = SymbolDictionary()
    pred_dict = SymbolDictionary()
    ent_dict.add_sym("unknown")
    pred_dict.add_sym("unknown")
    ent_cnt = {}
    pred_cnt = {}

    print("running statistics ...")

    for file in tqdm(files):

        path = os.path.join(scence_graphs_dir, file)
        scene_graphs = json.load(open(path))

        for image_id, scene_graph in tqdm(scene_graphs.items()):

            entities = scene_graph["objects"]

            for ent_id, ent in entities.items():

                if ent["name"] in ent_cnt:
                    ent_cnt[ent["name"]] += 1
                else:
                    ent_cnt[ent["name"]] = 1

                rels = ent["relations"]

                for rel in rels:

                    if rel["name"] in pred_cnt:
                        pred_cnt[rel["name"]] += 1
                    else:
                        pred_cnt[rel["name"]] = 1

    print("adding symbols ...")


    for ent_name, cnt in tqdm(ent_cnt.items()):
        ent_dict.add_sym(ent_name)
        tokens = word_dict.tokenize(ent_name, add_sym=True)
        len_cnt[len(tokens)] += 1

    for pred_name, cnt in tqdm(pred_cnt.items()):
        pred_dict.add_sym(pred_name)
        tokens = word_dict.tokenize(pred_name, add_sym=True)
        len_cnt[len(tokens)] += 1

    word_dict.dump_to_file("cache/word_dict.json")
    ent_dict.dump_to_file("cache/ent_dict.json")
    pred_dict.dump_to_file("cache/pred_dict.json")

    # get glove words
    print("parsing glove words")
    glove_words = []
    with open(glove_txt_path) as f:
        for line in tqdm(f.readlines()):
            elems = line.split(' ')
            word = elems[0]
            glove_words.append(word)
    print("parsing glove embeddings")
    glove_emb = np.zeros([len(glove_words), emb_dim])
    with open(glove_txt_path) as f:
        for i, line in tqdm(enumerate(f.readlines())):
            elems = line.split(' ')
            vec = elems[1:]
            vec = [float(_) for _ in vec]
            glove_emb[i] = vec

    print("saving word embeddings...")
    word_emb, uninit = create_glove_emb(word_dict, glove_words, glove_emb)
    np.save("cache/word_emb_init.npy", word_emb)

    print()
    print("%d entity categories, %d predicates" % (len(ent_dict), len(pred_dict)))

    print()
    print("%d words in word_dict" % len(word_dict))
    print("uninitialized words (replaced with the embedding of 'unknown'):")
    print(uninit)

    print()
    print("token length count (by category):")
    for length, number in enumerate(len_cnt):
        print("%2d %d" % (length, number))
