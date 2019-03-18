import torch
import torch.nn as nn
from torch.autograd import Variable
from roi_align.roi_align import RoIAlign
from lib.module.backbone import backbones
from lib.module.entity_net import EntityNet
from lib.module.relation_net import RelationNet

class VisionModel(nn.Module):

    def __init__(self, backbone,
                 roi_align: RoIAlign,
                 subject_net: EntityNet,
                 object_net: EntityNet,
                 relation_net: RelationNet):

        super(VisionModel, self).__init__()

        self.backbone = backbone
        self.roi_align = roi_align
        self.subject_net = subject_net
        self.object_net = object_net
        self.relation_net = relation_net

    def forward(self, images, sbj_boxes, obj_boxes, rel_boxes):

        N = images.size(0)
        feature_maps = self.backbone(images)

        box_ind = torch.arange(N, dtype=torch.int).repeat(3)
        boxes = torch.cat([sbj_boxes, obj_boxes, rel_boxes], dim=0)
        roi_features = self.roi_align(feature_maps, boxes, box_ind) # [ N, C, crop_size, crop_size ]

        sbj_emb, sbj_inter = self.subject_net(roi_features[:N])
        obj_emb, obj_inter = self.object_net(roi_features[N:2*N])
        rel_emb = self.relation_net(roi_features[2*N:], sbj_emb, sbj_inter, obj_emb, obj_inter)

        return sbj_emb, obj_emb, rel_emb

    @classmethod
    def build_from_config(cls, cfg):

        backbone_cls = backbones[cfg.backbone]
        backbone = backbone_cls()
        backbone.freeze()
        for layer in cfg.finetune_layers:
            backbone.defreeze(layer)

        roi_align = RoIAlign(cfg.crop_size, cfg.crop_size)
        entity_net = EntityNet(cfg.feature_dim, cfg.crop_size, cfg.emb_dim)
        relation_net = EntityNet(cfg.feature_dim, cfg.crop_size, cfg.emb_dim)

        return cls(backbone, roi_align, entity_net, entity_net, relation_net)

