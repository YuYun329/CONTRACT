import logging

import anndata
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import nn

logging.basicConfig(
    format="%(asctime)s - %(filename)s - %(name)s:   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.type_num = None
        self.args = args
        self.tau = args.info_nce_tau

    def info_nce(self, atac_prototypes, rna_prototypes, eps=1e-8):
        atac_prototypes = F.normalize(atac_prototypes, dim=1)
        rna_prototypes = F.normalize(rna_prototypes, dim=1)
        loss_res = None

        for k in range(self.type_num):
            idx = torch.ones((self.type_num,), dtype=torch.bool)
            idx[k] = False
            negative_atac = atac_prototypes[idx, :]
            negative_rna = rna_prototypes[idx, :]
            negative = torch.concat((negative_atac, negative_rna), dim=0)
            positive = rna_prototypes[k]

            Fp = torch.exp(atac_prototypes[k] * (positive / self.tau))
            Fn = (torch.exp(atac_prototypes[k] * (negative / self.tau)) + Fp).sum(dim=0)

            loss = (-1 * torch.log(Fp / (Fn + eps))).mean()
            if loss_res is None:
                loss_res = loss
            else:
                loss_res += loss
        return loss_res

    def forward(self, atac_feature, rna_feature, atac_label, rna_label):
        self.type_num = len(np.unique(rna_label))
        atac_prototypes = None
        rna_prototypes = None
        loss_cos = None

        for k in range(self.type_num):
            atac_proto = atac_feature[atac_label == k]
            rna_proto = rna_feature[rna_label == k]

            # prototype to info_nce
            if atac_prototypes is None:
                atac_prototypes = atac_proto.mean(dim=0).unsqueeze(dim=0)
            else:
                atac_prototypes = torch.concat(
                    (atac_prototypes, atac_proto.mean(dim=0).unsqueeze(dim=0)), dim=0)

            if rna_prototypes is None:
                rna_prototypes = rna_proto.mean(dim=0).unsqueeze(dim=0)
            else:
                rna_prototypes = torch.concat(
                    (rna_prototypes, rna_proto.mean(dim=0).unsqueeze(dim=0)), dim=0)

            # omics loss
            # cos = F.cosine_similarity(atac_proto, rna_proto, dim=0)
            # if loss_cos is None:
            #     loss_cos = -1 * cos.mean()
            # else:
            #     loss_cos -= cos.mean()

        return self.info_nce(atac_prototypes, rna_prototypes) # + loss_cos


def cosine_similarity(x1, x2, eps=1e-8):
    x1_norm = x1 / (torch.norm(x1, dim=1, keepdim=True) + eps)
    x2_norm = x2 / (torch.norm(x2, dim=1, keepdim=True) + eps)

    similarity = torch.matmul(x1_norm, x2_norm.T)
    return similarity

