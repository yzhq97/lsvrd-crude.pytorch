import torch
import torch.nn as nn
from roi_align.roi_align import RoIAlign
from lib.module.feature_net import FeatureNet
from lib.module.entity_net import EntityNet
from lib.module.relation_net import RelationNet

class VisionModel(nn.Module):

    def __init__(self,
                 roi_align: RoIAlign,
                 feature_net: FeatureNet,
                 subject_net: EntityNet,
                 object_net: EntityNet,
                 relation_net: RelationNet):

        super(VisionModel, self).__init__()

        self.roi_align = roi_align
        self.feature_net = feature_net
        self.subject_net = subject_net
        self.object_net = object_net
        self.relation_net = relation_net


    def forward(self, images, sbj_boxes, obj_boxes, rel_boxes):

        N = images.size(0)
        feature_maps = self.feature_net(images)

        box_ind = torch.arange(N, dtype=torch.int)

        sbj_features = self.roi_align(feature_maps, sbj_boxes, box_ind)
        obj_features = self.roi_align(feature_maps, obj_boxes, box_ind)
        rel_features = self.roi_align(feature_maps, rel_boxes, box_ind)

        sbj_emb, sbj_inter = self.subject_net(sbj_features)
        obj_emb, obj_inter = self.object_net(obj_features)
        rel_emb = self.relation_net(rel_features, sbj_emb, sbj_inter, obj_emb, obj_inter)

        return sbj_emb, obj_emb, rel_emb

    @classmethod
    def build_from_config(cls, cfg):
        roi_align = RoIAlign(cfg.crop_height, cfg.crop_width)
        if cfg.finetune:
            feature_net = FeatureNet()
        else:
            with torch.no_grad():
                feature_net = FeatureNet()
        entity_net = EntityNet(cfg.in_dim, cfg.emb_dim)
        relation_net = EntityNet(cfg.in_dim, cfg.emb_dim)
        return cls(roi_align, feature_net, entity_net, entity_net, relation_net)

