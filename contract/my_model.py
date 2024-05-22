import numpy as np
import torch
from torch import nn
from torch.nn import GELU

from contract.model_base import ModelBase
from contract.predictions import ATAC_Final, RNA_Final


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel: int = 5,
                 filters: int = 288,
                 kernel_size=1,
                 activation="gelu",
                 strides=1,
                 dilation_rate=1,
                 dropout=0,
                 residual=False,
                 pool_size=1,
                 batch_norm=True,
                 bn_momentum=0.90,
                 bn_gamma=None,
                 bn_type="standard",
                 padding="same",
                 ):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.residual = residual
        self.pool_size = pool_size
        self.batch_norm = batch_norm
        self.bn_momentum = bn_momentum
        self.bn_gamma = bn_gamma
        self.bn_type = bn_type
        self.padding = padding

        if self.activation == "gelu":
            self.act_func = GELU()
        else:
            self.act_func = nn.ReLU()

        self.conv_layer = nn.Conv1d(in_channels=self.in_channel,
                                    out_channels=self.filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.strides,
                                    padding=self.padding,
                                    dilation=self.dilation_rate,
                                    bias=False)
        nn.init.kaiming_normal_(self.conv_layer.weight, mode="fan_in", nonlinearity='relu')

        if self.batch_norm:
            self.bn_layer = nn.BatchNorm1d(num_features=self.filters,
                                           momentum=self.bn_momentum,
                                           affine=True,
                                           track_running_stats=True)
            if self.bn_gamma is None:
                nn.init.zeros_(self.bn_layer.weight) if self.residual else nn.init.ones_(self.bn_layer.weight)

        if self.dropout > 0:
            self.drop_func = nn.Dropout(dropout)

        if self.pool_size > 1:
            self.pool_func = nn.MaxPool1d(kernel_size=self.pool_size)

    def forward(self, inputs):
        # flow through variable current, shape(bs, 4, 1344)
        current = inputs

        # activation
        current = self.act_func(current)

        # convolution
        current = self.conv_layer(current)

        # batch norm
        if self.batch_norm:
            current = self.bn_layer(current)

        # dropout
        if self.dropout > 0:
            current = self.drop_func(current)

        # residual add
        if self.residual:
            current = inputs + current

        # Pool
        if self.pool_size > 1:
            current = self.pool_func(current)

        return current


class ConvTower(nn.Module):
    def __init__(
            self,
            filters_init,
            filters_end=None,
            filters_mult=None,
            divisible_by=1,
            repeat=1,
            **kwargs
    ):
        super(ConvTower, self).__init__()

        self.filters_init = filters_init
        self.filters_end = filters_end
        self.filters_mult = filters_mult
        self.divisible_by = divisible_by
        self.repeat = repeat
        self.kwargs = kwargs
        self.conv_inchannel = 288

        self.conv_blocks = nn.ModuleList()

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        # initialize filters
        rep_filters = filters_init

        # determine multiplier
        if filters_mult is None:
            assert filters_end is not None
            filters_mult = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

        for ri in range(repeat):
            self.conv_blocks.append(ConvBlock(in_channel=self.conv_inchannel, filters=_round(rep_filters), **kwargs))
            self.conv_inchannel = _round(rep_filters)
            # update filters
            rep_filters *= filters_mult

    def forward(self, inputs):
        current = inputs

        for conv_block in self.conv_blocks:
            current = conv_block(current)

        return current


class GateNet(nn.Module):
    def __init__(self):
        super(GateNet, self).__init__()
        self.linear_wx = nn.Linear(in_features=5, out_features=7)
        nn.init.kaiming_normal_(self.linear_wx.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.linear_sx = nn.Linear(in_features=512, out_features=256)
        nn.init.kaiming_normal_(self.linear_sx.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, conv_feature):
        x = x.permute(0, 2, 1)
        wx = self.linear_wx(x)
        wx = self.softmax(wx)

        sx = conv_feature.permute(0, 2, 1)
        sx = self.linear_sx(sx)
        return wx @ sx


class PLEContract(ModelBase):
    def __init__(self, atac_cells, rna_cells, bottleneck_size):
        super(PLEContract, self).__init__(atac_cells, rna_cells, bottleneck_size)
        self.share_A_conv_1 = ConvBlock(in_channel=5, filters=288, kernel_size=17, pool_size=3)
        self.share_A_tower = ConvTower(filters_init=288, filters_mult=1.122, repeat=6, kernel_size=5, pool_size=2)
        self.share_A_conv_2 = ConvBlock(in_channel=512, filters=256, kernel_size=1)

        self.share_B_conv_1 = ConvBlock(in_channel=10, filters=288, kernel_size=17, pool_size=3)
        self.share_B_tower = ConvTower(filters_init=288, filters_mult=1.122, repeat=6, kernel_size=5, pool_size=2)
        self.share_B_conv_2 = ConvBlock(in_channel=512, filters=256, kernel_size=1)

        self.share_C_conv_1 = ConvBlock(in_channel=5, filters=288, kernel_size=17, pool_size=3)
        self.share_C_tower = ConvTower(filters_init=288, filters_mult=1.122, repeat=6, kernel_size=5, pool_size=2)
        self.share_C_conv_2 = ConvBlock(in_channel=512, filters=256, kernel_size=1)

        self.gate_A_net = GateNet()
        self.gate_C_net = GateNet()

        self.gelu = GELU()

        self.gate_A_linear = nn.Linear(in_features=256, out_features=5)
        self.gate_C_linear = nn.Linear(in_features=256, out_features=5)
        nn.init.kaiming_normal_(self.gate_A_linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.gate_C_linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

        self.dense_A = nn.Linear(in_features=1344 * 5, out_features=64)
        self.dense_C = nn.Linear(in_features=1344 * 5, out_features=64)
        self.drop = nn.Dropout(0.2)

        self.atac_final = ATAC_Final(latent_dim=bottleneck_size, cells_atac=atac_cells)
        self.rna_final = RNA_Final(latent_dim=bottleneck_size, cells_rna=rna_cells)
        self.sigmoid = nn.Sigmoid()

    def forward(self, atac_seq, rna_seq):
        atac_seq = atac_seq.permute(0, 2, 1)
        rna_seq = rna_seq.permute(0, 2, 1)
        mix_seq = torch.concat([atac_seq, rna_seq], dim=1)

        # share_A to conv tower
        seq_share_A = self.share_A_conv_1(atac_seq)
        seq_share_A = self.share_A_tower(seq_share_A)
        seq_share_A = self.share_A_conv_2(seq_share_A)

        # share_B #########
        seq_share_B = self.share_B_conv_1(mix_seq)
        seq_share_B = self.share_B_tower(seq_share_B)
        seq_share_B = self.share_B_conv_2(seq_share_B)

        # share C #####
        seq_share_C = self.share_C_conv_1(rna_seq)
        seq_share_C = self.share_C_tower(seq_share_C)
        seq_share_C = self.share_C_conv_2(seq_share_C)

        # A dense block
        gate_A = self.gate_A_net(x=atac_seq, conv_feature=torch.concat([seq_share_A, seq_share_B], dim=1))
        gate_A = self.gelu(gate_A)
        gate_A = self.gate_A_linear(gate_A)
        gate_A = gate_A + atac_seq.permute(0, 2, 1)
        gate_A = self.dense_A(gate_A.flatten(start_dim=1))
        gate_A = self.drop(gate_A)

        # C dense block
        gate_C = self.gate_C_net(x=rna_seq, conv_feature=torch.concat([seq_share_C, seq_share_B], dim=1))
        gate_C = self.gelu(gate_C)
        gate_C = self.gate_C_linear(gate_C)
        gate_C = gate_C + rna_seq.permute(0, 2, 1)
        gate_C = self.dense_C(gate_C.flatten(start_dim=1))
        gate_C = self.drop(gate_C)

        current_atac = self.atac_final(gate_A)
        current_rna = self.rna_final(gate_C)

        return current_atac, current_rna


if __name__ == '__main__':
    model = PLEContract(atac_cells=1000, rna_cells=1000, bottleneck_size=64).cuda()

    data_atac = torch.rand(size=(2, 1344, 5)).float().cuda()
    data_rna = torch.rand(size=(2, 1344, 5)).float().cuda()

    pred_atac, pred_rna = model.forward(data_atac, data_rna)
    print(model)
