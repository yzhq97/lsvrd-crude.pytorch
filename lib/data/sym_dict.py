import json

class SymbolDictionary:

    all_puct = ["?", "!", "\\", "/", ")", "(", ".", ",", ";", ":"]
    ignored_puncts = ["?", "!", "\\", "/", ")", "("]
    kept_puncts = [".", ",", ";", ":"]
    end_puncts = [">", "<", ":"]
    delim = " "

    def __init__(self, style="gqa", sym2idx=None, idx2sym=None):
        """
        :param style: "vqa2" or "gqa"
        :param sym2idx: a dict (str)sym -> (int)idx
        :param idx2sym: a list of syms ordered by idx
        """
        if sym2idx is None:
            sym2idx = {}
        if idx2sym is None:
            idx2sym = []
        self.sym2idx = sym2idx
        self.idx2sym = idx2sym
        self.style = style

    @property
    def n_tokens(self):
        return len(self.sym2idx)

    @property
    def padding_idx(self):
        return len(self.sym2idx)

    def __contains__(self, sym):
        return sym in self.sym2idx

    def __len__(self):
        return len(self.idx2sym)

    def add_sym(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
        return self.sym2idx[sym]

    def remove_puncts(self, string):
        for punct in self.kept_puncts:
            string = string.replace(punct, self.delim + punct + self.delim)
        for punct in self.ignored_puncts:
            string = string.replace(punct, "")
        return string

    def tokenize(self, sentence, add_sym=False):

        if self.style == "vqa2":
            sentence = sentence.lower()
            sentence = sentence.replace(',', '').replace('?', '').replace("'s", " \'s")
            syms = sentence.split()
        elif self.style == "gqa":
            for punct in self.kept_puncts:
                sentence = sentence.replace(punct, self.delim + punct + self.delim)
            for punct in self.ignored_puncts:
                sentence = sentence.replace(punct, "")
            syms = sentence.lower().split(self.delim)
            syms = [t for t in syms if t != ""]
        else:
            raise Exception("invalid style: %s" % str(self.style))

        if add_sym:
            tokens = [ self.add_sym(w) for w in syms ]
        else:
            tokens = [ self.sym2idx[w] for w in syms ]

        return tokens

    def dump_to_file(self, path):
        sym_dict = { "sym2idx": self.sym2idx, "idx2sym": self.idx2sym }
        json.dump(sym_dict, open(path, 'w'))
        print('sym_dict dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path, style="gqa"):
        sym_dict = json.load(open(path, 'r'))
        sym2idx = sym_dict["sym2idx"]
        idx2sym = sym_dict["idx2sym"]
        d = cls(style, sym2idx, idx2sym)
        return d

