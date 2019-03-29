import torch
import torch.nn as nn
from lib.module.similarity_model import PairwiseCosineSimilarity
from lib.loss.triplet_loss import TripletLoss
from lib.loss.triplet_softmax_loss import TripletSoftmaxLoss


class LossModel(nn.Module):

    def __init__(self,
                 similarity: PairwiseCosineSimilarity,
                 triplet_loss: TripletLoss,
                 triplet_softmax_loss: TripletSoftmaxLoss,
                 x_tr, x_trsm, y_tr, y_trsm,
                 similarity_norm = 5.0):

        super(LossModel, self).__init__()

        self.similarity = similarity
        self.triplet_loss = triplet_loss
        self.triplet_softmax_loss = triplet_softmax_loss
        self.similarity_norm = similarity_norm

        self.x_tr = x_tr
        self.x_trsm = x_trsm
        self.y_tr = y_tr
        self.y_trsm = y_trsm

    def forward(self, v_emb, l_emb):

        s = self.similarity(v_emb, l_emb)
        loss = torch.tensor(0.0)
        if self.x_tr: loss = loss + self.triplet_loss(s.t())
        if self.x_trsm: loss = loss + self.triplet_softmax_loss(s.t().mul(self.similarity_norm))
        if self.y_tr: loss = loss + self.triplet_loss(s)
        if self.y_trsm: loss = loss + self.triplet_softmax_loss(s.mul(self.similarity_norm))
        return loss

    @classmethod
    def build_from_config(cls, cfg):
        similarity = PairwiseCosineSimilarity()
        triplet_loss = TripletLoss(cfg.n_neg, cfg.margin)
        triplet_softmax_loss = TripletSoftmaxLoss(cfg.n_neg, cfg.margin)
        return cls(similarity, triplet_loss, triplet_softmax_loss,
                   cfg.x_tr, cfg.x_trsm, cfg.y_tr, cfg.y_trsm, cfg.similarity_norm)

