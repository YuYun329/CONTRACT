from torch import nn

from contract.predictions import ATAC_Final, RNA_Final


class ModelBase(nn.Module):
    def __init__(self, atac_cells, rna_cells, bottleneck_size):
        super(ModelBase, self).__init__()
        self.atac_final = ATAC_Final(latent_dim=bottleneck_size, cells_atac=atac_cells)
        self.rna_final = RNA_Final(latent_dim=bottleneck_size, cells_rna=rna_cells)
        # self.concat_final = Concat_Final(latent_dim=bottleneck_size, cells_total=atac_cells + rna_cells)

    def get_embedding(self):
        return self.atac_final.linear.weight, self.rna_final.linear.weight