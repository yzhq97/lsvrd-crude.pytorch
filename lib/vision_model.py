import torch
import torch.nn as nn
from roi_align.roi_align import RoIAlign
from lib.module.language_model import TextEmbedding
from lib.module.feature_net import FeatureNet
from lib.module.entity_net import EntityNet
from lib.module.relation_net import RelationNet
from lib.module.similarity_model import PairwiseCosineSimilarity

class VisionModel(nn.Module):

    def __init__(self,
                 roi_align: RoIAlign,
                 feature_net: FeatureNet,
                 subject_net: EntityNet,
                 object_net: EntityNet,
                 relation_net: RelationNet,
                 pairwise_cosine: PairwiseCosineSimilarity):

        super(VisionModel, self).__init__()

        self.roi_align = roi_align
        self.feature_net = feature_net
        self.subject_net = subject_net
        self.object_net = object_net
        self.relation_net = relation_net
        self.pairwise_cosine = pairwise_cosine


    def forward(self, image_ids, images,
                sbj_boxes, obj_boxes, pred_boxes):

        N = image_ids.size(0)
        feature_maps = self.feature_net(images)

        box_ind = torch.arange(N, dtype=torch.int)

        sbj_features = self.roi_align(feature_maps, sbj_boxes, box_ind)
        obj_features = self.roi_align(feature_maps, obj_boxes, box_ind)
        pred_features = self.roi_align(feature_maps, pred_boxes, box_ind)

        sbj_emb, sbj_inter = self.subject_net(sbj_features)
        obj_emb, obj_inter = self.object_net(obj_features)
        pred_emb = self.relation_net(pred_features, sbj_emb, sbj_inter, obj_emb, obj_inter)

        return sbj_emb, obj_emb, pred_emb


