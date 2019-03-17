import torch.nn as nn
from lib.module.similarity_model import PairwiseCosineSimilarity
from lib.loss.triplet_loss import TripletLoss
from lib.loss.triplet_softmax_loss import TripletSoftmaxLoss


class LossModel(nn.Module):

    def __init__(self,
                 similarity: PairwiseCosineSimilarity,
                 triplet_loss: TripletLoss,
                 triplet_softmax_loss: TripletSoftmaxLoss,
                 similarity_norm = 3.0):

        super(LossModel, self).__init__()

        self.similarity = similarity
        self.triplet_loss = triplet_loss
        self.triplet_softmax_loss = triplet_softmax_loss
        self.similarity_norm = similarity_norm

    def forward(self, v_emb, l_emb):

        s = self.similarity(v_emb, l_emb).mul_(self.similarity_norm)
        tr_loss = self.triplet_loss(s.t())
        trsm_loss = self.triplet_softmax_loss(s)
        loss = tr_loss + trsm_loss

        return loss

    @classmethod
    def build_from_config(cls, cfg):
        similarity = PairwiseCosineSimilarity()
        triplet_loss = TripletLoss(cfg.loss.n_neg, cfg.margin)
        triplet_softmax_loss = TripletSoftmaxLoss(cfg.n_neg, cfg.margin)
        return cls(similarity, triplet_loss, triplet_softmax_loss, cfg.similarity_norm)

