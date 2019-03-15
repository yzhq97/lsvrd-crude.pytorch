import os
import json
import h5py
import math
import numpy as np
from tqdm import tqdm, trange
from lib.data.utils import digitize_dict_keys


class H5DataLoader:

    def __init__(self, info, h5s, fields, split_name="all"):
        """
        :param info: meta info loaded from "info.json"
        :param h5s: a list of h5s
        :param fields: specify which fields will be used and their order
                       as json array, e.g. [ {"name": "feature1", "preload": true}, ]
        :param split_name: the split to use, e.g. "train", "val", etc.
        """

        self.split_name = split_name

        self.name = info["name"]
        self.fields = [ self.get_field_info(info, field["name"]) for field in fields ]
        for i, field in enumerate(fields): self.fields[i]["preload"] = field["preload"]
        self.ids = info["splits"][split_name]
        indices = digitize_dict_keys(info["indices"])
        self.indices = { key: (val["block"], val["idx"]) for key, val in indices.items() }

        self.data = []
        for field in self.fields:
            if field["preload"]:
                field_arrays = self.preload_field(field, h5s)
                self.data.append(field_arrays)
            else:
                field_h5s = [ h5.get(field["name"]) for h5 in h5s ]
                self.data.append(field_h5s)

    def __len__(self):
        return len(self.ids)

    @property
    def n_fields(self):
        return len(self.fields)

    def __getitem__(self, id):
        block_idx, idx = self.indices[id]
        retrieved = [ self.data[field_idx][block_idx][idx] for field_idx in range(self.n_fields) ]
        return retrieved

    def preload_field(self, field, h5s):
        print("loading field '%s' into memory" % field["name"])
        blocks = [h5[field["name"]] for h5 in h5s]
        array_shape = [ len(self) ] + field["shape"]
        array = np.zeros(shape=array_shape, dtype=field["dtype"])
        for i in trange(len(self)):
            id = self.ids[i]
            block_idx, idx = self.indices[id]
            array[i] = blocks[block_idx][idx]
            self.indices[id] = (0, i)
        return [array]

    @staticmethod
    def get_field_info(info, field_name):
        for field in info["fields"]:
            if field["name"] == field_name:
                return field
        return None

    @classmethod
    def load_from_directory(cls, dir_path, fields, split_name="all"):
        info_path = os.path.join(dir_path, "info.json")
        info = json.load(open(info_path))
        h5s = [ h5py.File(os.path.join(dir_path, "data_%d.h5" % i), "r") for i in range(info["n_blocks"]) ]
        return cls(info, h5s, fields, split_name)


class H5DataWriter:

    def __init__(self, dir_path, name, n_entries, n_blocks, fields):
        """
        create formated partitioned h5 files
        :param dir_path: output directory path
        :param name: dataset name
        :param n_entries: total number of entries
        :param n_blocks: number of partitions
        :param fields: field definitions given in json array,
        e.g. [ { "name": "field1", "shape": [2048], "dtype": "float32"} ]
        """

        self.dir_path = dir_path

        self.name = name
        self.n_entries = n_entries
        self.n_blocks = n_blocks
        self.fields = fields

        self.block_size = int(math.ceil(1.0 * n_entries / n_blocks))
        self.array_lengths = [ self.block_size ] * (n_blocks - 1) + [ n_entries % self.block_size ]
        self.h5s = [ h5py.File(os.path.join(dir_path, "data_%d.h5" % i), "w")
                     for i in range(n_blocks) ]
        for i in range(n_blocks):
            h5 = self.h5s[i]
            length = self.array_lengths[i]
            for field in self.fields:
                array_shape = [ length ] + field["shape"]
                h5.create_dataset(name=field["name"], shape=array_shape, dtype=field["dtype"])
        self.h5_lookup_table = [ [h5.get(field["name"]) for field in self.fields]
                                 for h5 in self.h5s]

        self.ids = []
        self.indices = {}
        self.splits = {}

        self.current_idx = 0

    def convert_idx(self, idx):
        block_idx = idx // self.block_size
        ins_idx = idx % self.block_size
        return block_idx, ins_idx

    def put(self, id, data, idx=-1):
        id = int(id)
        if idx < 0:
            idx = self.current_idx
            self.ids.append(id)
            self.current_idx += 1
        else:
            self.ids[idx] = id
        block_idx, ins_idx = self.convert_idx(idx)
        arrays = self.h5_lookup_table[block_idx]
        for datum, array in zip(data, arrays):
            array[ins_idx] = datum
        self.indices[id] = { "block": block_idx, "idx": ins_idx }

    def add_split(self, split_name, ids):
        self.splits[split_name] = ids

    def close(self):
        self.splits["all"] = self.ids
        info = {
            "name": self.name,
            "n_entries": self.n_entries,
            "n_blocks": self.n_blocks,
            "fields": self.fields,
            "splits": self.splits,
            "indices": self.indices
        }
        with open(os.path.join(self.dir_path, "info.json"), "w") as fp:
            json.dump(info, fp)
        for h5 in self.h5s: h5.close()