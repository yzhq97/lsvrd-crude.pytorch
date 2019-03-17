import torch
import torch.nn as nn
from lib.vision_model import VisionModel
from lib.language_model import LanguageModel
from lib.module.similarity_model import PairwiseCosineSimilarity
from lib.loss.triplet_loss import TripletLoss
from lib.loss.triplet_softmax_loss import TripletSoftmaxLoss
from lib.loss.consistency_loss import ConsistencyLoss

class LossModel(nn.Module):

    def __init__(self,
                 pairwise_cosine: PairwiseCosineSimilarity,
                 triplet_loss: TripletLoss,
                 triplet_softmax_loss: TripletSoftmaxLoss):

        super(LossModel, self).__init__()

        self.pairwise_cosine = pairwise_cosine
        self.triplet_loss = triplet_loss
        self.triplet_softmax_loss = triplet_softmax_loss

    def forward(self, v_emb, l_emb):

        s = self.pairwise_cosine(v_emb, l_emb)
        tr_loss = self.triplet_loss(s.t())
        trsm_loss = self.triplet_softmax_loss(s)
        loss = tr_loss + trsm_loss

        return loss

    @classmethod
    def build_from_config(cls, cfg):
        pairwise_cosine = PairwiseCosineSimilarity(cfg.norm_scale)
        triplet_loss = TripletLoss(cfg.loss.n_neg, cfg.margin)
        triplet_softmax_loss = TripletSoftmaxLoss(cfg.n_neg, cfg.margin)
        return cls(pairwise_cosine, triplet_loss, triplet_softmax_loss)

