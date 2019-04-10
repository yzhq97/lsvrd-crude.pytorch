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

    print("building dictionaries ... ")

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

    ent_dict = SymbolDictionary()
    ent_dict.add_sym("unknown")
    ent_cnt = {}
    ent_len_cnt = [0] * 8

    pred_dict = SymbolDictionary()
    pred_dict.add_sym("unknown")
    pred_cnt = {}
    pred_len_cnt = [0] * 8

    attr_dict = SymbolDictionary()
    attr_cnt = {}
    attr_len_cnt = [0] * 8

    print("    running statistics ...")

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

                attrs = ent["attributes"]
                for attr in attrs:
                    if attr in attr_cnt:
                        attr_cnt[attr] += 1
                    else:
                        attr_cnt[attr] = 1

                rels = ent["relations"]
                for rel in rels:
                    if rel["name"] in pred_cnt:
                        pred_cnt[rel["name"]] += 1
                    else:
                        pred_cnt[rel["name"]] = 1

    print("    adding symbols ...")

    for ent_name, cnt in tqdm(ent_cnt.items()):
        ent_dict.add_sym(ent_name)
        tokens = word_dict.tokenize(ent_name, add_sym=True)
        ent_len_cnt[len(tokens)] += 1

    for attr_name, cnt in tqdm(attr_cnt.items()):
        attr_dict.add_sym(attr_name)
        tokens = word_dict.tokenize(attr_name, add_sym=True)
        attr_len_cnt[len(tokens)] += 1

    for pred_name, cnt in tqdm(pred_cnt.items()):
        pred_dict.add_sym(pred_name)
        tokens = word_dict.tokenize(pred_name, add_sym=True)
        pred_len_cnt[len(tokens)] += 1

    word_dict.dump_to_file("cache/word_dict.json")
    ent_dict.dump_to_file("cache/ent_dict.json")
    attr_dict.dump_to_file("cache/attr_dict.json")
    pred_dict.dump_to_file("cache/pred_dict.json")

    # get glove words
    print("    parsing glove words and embeddings")
    glove_lines = open(glove_txt_path).readlines()
    glove_words = []
    glove_emb = np.zeros([len(glove_lines), emb_dim])
    for i in trange(len(glove_lines)):
        line = glove_lines[i]
        elems = line.split(' ')
        word = elems[0]
        glove_words.append(word)
        vec = elems[1:]
        vec = [float(_) for _ in vec]
        glove_emb[i] = vec

    print("    saving word embeddings...")
    word_emb, uninit = create_glove_emb(word_dict, glove_words, glove_emb)
    np.save("cache/word_emb_init_%dd.npy" % emb_dim, word_emb)

    print()
    print("    %d entity categories, %d attributes, %d predicates" % (len(ent_dict), len(attr_dict), len(pred_dict)))

    print()
    print("    %d words in word_dict" % len(word_dict))
    print("    uninitialized words (replaced with the embedding of 'unknown'):")
    print(uninit)

    print()
    print("    ent token length count (by category):")
    for length, number in enumerate(ent_len_cnt):
        print("%2d %d" % (length, number))

    print()
    print("    attr token length count (by category):")
    for length, number in enumerate(attr_len_cnt):
        print("%2d %d" % (length, number))

    print()
    print("    pred token length count (by category):")
    for length, number in enumerate(pred_len_cnt):
        print("%2d %d" % (length, number))
