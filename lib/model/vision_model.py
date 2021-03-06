import torch
import torch.nn as nn
from torch.autograd import gradcheck
from roi_align.crop_and_resize import CropAndResizeFunction
from lib.module.backbone import backbones
from lib.module.entity_net import EntityNet
from lib.module.relation_net import RelationNet

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x.detach()

    def freeze(self):
        pass

class VisionModel(nn.Module):

    def __init__(self, backbone,
                 ent_crop_and_resize: CropAndResizeFunction,
                 rel_crop_and_resize: CropAndResizeFunction,
                 ent_net: EntityNet,
                 rel_net: RelationNet):

        super(VisionModel, self).__init__()

        self.backbone = backbone
        self.ent_crop_and_resize = ent_crop_and_resize
        self.rel_crop_and_resize = rel_crop_and_resize
        self.ent_net = ent_net
        self.rel_net = rel_net

    def forward(self, images, sbj_boxes, obj_boxes, rel_boxes):

        N = images.size(0)
        feature_maps = self.backbone(images)

        box_ind = torch.arange(N, dtype=torch.int, device=images.device)
        ent_box_ind = box_ind.repeat(2)
        ent_boxes = torch.cat([sbj_boxes, obj_boxes], dim=0)
        ent_features = self.ent_crop_and_resize(feature_maps, ent_boxes, ent_box_ind)
        rel_features = self.rel_crop_and_resize(feature_maps, rel_boxes, box_ind)

        sbj_emb, sbj_inter = self.ent_net(ent_features[:N])
        obj_emb, obj_inter = self.ent_net(ent_features[N:])
        rel_emb = self.rel_net(rel_features, sbj_emb, sbj_inter, obj_emb, obj_inter)

        return sbj_emb, obj_emb, rel_emb

    def infer_rel(self, image, sbj_boxes, obj_boxes, rel_boxes):

        N = rel_boxes.size(0)
        feature_map = self.backbone(image) # [ 1, C, H, W ]

        box_ind = torch.zeros(N, dtype=torch.int, device=feature_map.device) # [ N ]
        ent_box_ind = box_ind.repeat(2) # [ 2N ]
        ent_boxes = torch.cat([sbj_boxes, obj_boxes], dim=0)
        ent_features = self.ent_crop_and_resize(feature_map, ent_boxes, ent_box_ind)
        rel_features = self.rel_crop_and_resize(feature_map, rel_boxes, box_ind)

        sbj_emb, sbj_inter = self.ent_net(ent_features[:N])
        obj_emb, obj_inter = self.ent_net(ent_features[N:])
        rel_emb = self.rel_net(rel_features, sbj_emb, sbj_inter, obj_emb, obj_inter)

        return rel_emb

    def infer_ent(self, image, boxes):

        N = boxes.size(0)
        feature_map = self.backbone(image)

        box_ind = torch.zeros(N, dtype=torch.int, device=feature_map.device)
        ent_features = self.ent_crop_and_resize(feature_map, boxes, box_ind)
        ent_embs, ent_inters = self.ent_net(ent_features)

        return ent_embs

    @classmethod
    def build_from_config(cls, cfg):

        backbone_cls = Identity if cfg.pre_extract else backbones[cfg.backbone]
        backbone = backbone_cls()
        backbone.freeze()

        ent_crop_and_resize = CropAndResizeFunction(cfg.ent_crop_size, cfg.ent_crop_size)
        rel_crop_and_resize = CropAndResizeFunction(cfg.rel_crop_size, cfg.rel_crop_size)
        ent_net = EntityNet(cfg.feature_dim, cfg.ent_crop_size, cfg.emb_dim, cfg.hid_dim)
        rel_net = RelationNet(cfg.feature_dim, cfg.rel_crop_size, cfg.emb_dim, cfg.hid_dim)

        model = cls(backbone, ent_crop_and_resize, rel_crop_and_resize, ent_net, rel_net)

        return model

