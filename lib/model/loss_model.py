import torch.nn as nn
from lib.module.similarity_model import PairwiseCosineSimilarity
from lib.loss.triplet_loss import TripletLoss
from lib.loss.triplet_softmax_loss import TripletSoftmaxLoss


class LossModel(nn.Module):

    def __init__(self,
                 similarity: PairwiseCosineSimilarity,
                 triplet_loss: TripletLoss,
                 triplet_softmax_loss: TripletSoftmaxLoss,
                 similarity_norm = 5.0):

        super(LossModel, self).__init__()

        self.similarity = similarity
        self.triplet_loss = triplet_loss
        self.triplet_softmax_loss = triplet_softmax_loss
        self.similarity_norm = similarity_norm

    def forward(self, v_emb, l_emb):

        s = self.similarity(v_emb, l_emb)
        trsm_loss = self.triplet_softmax_loss(s.mul(self.similarity_norm)) # Ly_trsm
        tr_loss = self.triplet_loss(s.t())  # Lx_tr
        loss = tr_loss + trsm_loss
        return loss

    @classmethod
    def build_from_config(cls, cfg):
        similarity = PairwiseCosineSimilarity()
        triplet_loss = TripletLoss(cfg.n_neg, cfg.margin)
        triplet_softmax_loss = TripletSoftmaxLoss(cfg.n_neg, cfg.margin)
        return cls(similarity, triplet_loss, triplet_softmax_loss, cfg.similarity_norm)

