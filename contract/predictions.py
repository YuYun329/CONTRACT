import torch
from torch import nn


class ATAC_Final(nn.Module):
    def __init__(self, latent_dim, cells_atac):
        super(ATAC_Final, self).__init__()
        self.linear = nn.Linear(latent_dim, cells_atac)

    def forward(self, atac_embed):
        return torch.sigmoid(self.linear(atac_embed))


class RNA_Final(nn.Module):
    def __init__(self, latent_dim, cells_rna):
        super(RNA_Final, self).__init__()
        self.linear = nn.Linear(latent_dim, cells_rna)

    def forward(self, rna_embed):
        return torch.sigmoid(self.linear(rna_embed))


class Concat_Final(nn.Module):
    def __init__(self, latent_dim, cells_total):
        super(Concat_Final, self).__init__()
        self.linear = nn.Linear(latent_dim, cells_total)

    def forward(self, concat_embed):
        return torch.sigmoid(self.linear(concat_embed))