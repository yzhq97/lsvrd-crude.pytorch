from __future__ import print_function
import os
import json
import h5py
import torch
from torch.utils.data import Dataset

class GQADataset:

    def __init__(self, spatial_dir, objects_dir, scene_graph_path):

        self.scene_graph = json.load(open(scene_graph_path))
        self.ids = self.scene_graph.keys()

        self.spatial_info = json.load(open(os.path.join(spatial_dir, 'gqa_spatial_info.json')))
        self.objects_info = json.load(open(os.path.join(objects_dir, 'gqa_objects_info.json')))

        self.spatial_datasets = [h5py.File(os.path.join(spatial_dir, 'gqa_spatial_%d.h5' % i), 'r').get('features')
                            for i in range(10)]
        self.roi_datasets = [h5py.File(os.path.join(objects_dir, 'gqa_objects_%d.h5' % i), 'r').get('bboxes')
                            for i in range(10)]

    def __getitem__(self, idx):

        image_id = self.ids[idx]

        sg = self.scene_graph[image_id]

        spatial_info = self.spatial_info[image_id]
        objects_info = self.objects_info[image_id]

        spatial = self.spatial_datasets[spatial_info['file']][spatial_info['idx']]
        rois = self.roi_datasets[objects_info['file']][objects_info['idx']]