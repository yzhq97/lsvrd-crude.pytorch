import torch
import torch.nn as nn
from roi_align.roi_align import RoIAlign
from lib.module.language_model import TextEmbedding
from lib.module.feature_net import FeatureNet
from lib.module.entity_net import EntityNet
from lib.module.relation_net import RelationNet
from lib.module.similarity_model import PairwiseCosineSimilarity
from lib.loss.triplet_loss import TripletLoss
from lib.loss.triplet_softmax_loss import TripletSoftmaxLoss
from lib.loss.consistency_loss import ConsistencyLoss

class LSVRD(nn.Module):

    def __init__(self,
                 roi_align: RoIAlign,
                 feature_net: FeatureNet,
                 language_model: TextEmbedding,
                 subject_net: EntityNet,
                 object_net: EntityNet,
                 relation_net: RelationNet,
                 pairwise_cosine: PairwiseCosineSimilarity,
                 triplet_loss: TripletLoss,
                 triplet_softmax_loss: TripletSoftmaxLoss):

        super(LSVRD, self).__init__()

        self.roi_align = roi_align
        self.feature_net = feature_net
        self.language_model = language_model
        self.subject_net = subject_net
        self.object_net = object_net
        self.relation_net = relation_net
        self.pairwise_cosine = pairwise_cosine
        self.triplet_loss = triplet_loss
        self.triplet_softmax_loss = triplet_softmax_loss


    def forward(self, image_ids, im):
        pass