import torch
import torch.nn as nn

class PairwiseCosineSimilarity(nn.Module):

    def __init__(self, eps=1e-8):

        super(PairwiseCosineSimilarity, self).__init__()

        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=eps)

    def forward(self, x, y):

        Nx, Lx = x.size()
        Ny, Ly = y.size()
        assert Lx == Ly

        x_r = x.unsqueeze(1).repeat(1, Ny, 1)
        y_r = y.unsqueeze(0).repeat(Nx, 1, 1)

        p = self.cosine_similarity(x_r, y_r) # [B, Nx, Ny]

        return p


if __name__ == "__main__":
    """
    unit test
    """

    pcs = PairwiseCosineSimilarity()

    x = torch.rand(5, 3)
    y = torch.rand(5, 3)
    s = pcs(x, y)

    print(s.size())