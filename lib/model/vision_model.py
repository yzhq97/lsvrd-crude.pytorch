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
                 crop_and_resize: CropAndResizeFunction,
                 ent_net: EntityNet,
                 rel_net: RelationNet):

        super(VisionModel, self).__init__()

        self.backbone = backbone
        self.crop_and_resize = crop_and_resize
        self.ent_net = ent_net
        self.rel_net = rel_net

    def forward(self, images, sbj_boxes, obj_boxes, rel_boxes):

        N = images.size(0)
        feature_maps = self.backbone(images)

        box_ind = torch.arange(N, dtype=torch.int).repeat(3).cuda()
        boxes = torch.cat([sbj_boxes, obj_boxes, rel_boxes], dim=0)
        roi_features = self.crop_and_resize(feature_maps, boxes, box_ind) # [ N, C, crop_size, crop_size ]

        # gradcheck(self.crop_and_resize, (feature_maps, boxes, box_ind), raise_exception=True)

        sbj_features = roi_features[:N]
        obj_features = roi_features[N:2*N]
        rel_features = roi_features[2*N:]

        sbj_emb, sbj_inter = self.ent_net(sbj_features)
        obj_emb, obj_inter = self.ent_net(obj_features)
        rel_emb = self.rel_net(rel_features, sbj_emb, sbj_inter, obj_emb, obj_inter)

        return sbj_emb, obj_emb, rel_emb

    @classmethod
    def build_from_config(cls, cfg):

        backbone_cls = Identity if cfg.pre_extract else backbones[cfg.backbone]
        backbone = backbone_cls()
        backbone.freeze()

        crop_and_resize = CropAndResizeFunction(cfg.crop_size, cfg.crop_size)
        ent_net = EntityNet(cfg.feature_dim, cfg.crop_size, cfg.emb_dim)
        rel_net = RelationNet(cfg.feature_dim, cfg.crop_size, cfg.emb_dim)

        model = cls(backbone, crop_and_resize, ent_net, rel_net)

        return model

