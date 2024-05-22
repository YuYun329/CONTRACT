import torch.nn as nn
import torch.nn.functional as F
import torch


class _conv_layer(nn.Module):
    def __init__(self, number_input_features, number_output_features, kernel_size, pool_size, drop_rate):
        super(_conv_layer, self).__init__()
        self.pool_size = pool_size
        self.drop_rate = drop_rate

        self.add_module('conv', nn.Conv1d(in_channels=number_input_features,
                                          out_channels=number_output_features,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          bias=True))
        self.add_module('norm', nn.BatchNorm1d(num_features=number_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

        if self.pool_size > 0:
            self.add_module('pool', nn.MaxPool1d(kernel_size=pool_size))

    def forward(self, input):
        new_features = self.relu(self.norm(self.conv(input)))
        if self.pool_size > 0:
            new_features = self.pool(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, self.drop_rate)
        return new_features


class DeepSEA(nn.Module):
    def __init__(self, n_cells=2034):
        super(DeepSEA, self).__init__()
        self.add_module('conv1', _conv_layer(number_input_features=4,
                                             number_output_features=320,
                                             kernel_size=8,
                                             pool_size=4,
                                             drop_rate=0.2))
        self.add_module('conv2', _conv_layer(number_input_features=320,
                                             number_output_features=480,
                                             kernel_size=8,
                                             pool_size=4,
                                             drop_rate=0.2))
        self.add_module('conv3', _conv_layer(number_input_features=480,
                                             number_output_features=960,
                                             kernel_size=8,
                                             pool_size=0,
                                             drop_rate=0.5))

        self.add_module('dense1', nn.Sequential(nn.Flatten(start_dim=1),
                                                nn.Linear(in_features=74 * 960,
                                                          out_features=925),
                                                nn.ReLU(inplace=True)))
        self.add_module('dense2', nn.Sequential(nn.Linear(in_features=925,
                                                          out_features=n_cells),
                                                nn.Sigmoid()))

    def forward(self, input_ids):
        """
        :param input_ids: [batch_size, 4, 1000]
        :return: pred: [batch_size, n_cells]
        """
        input_ids = input_ids.permute(0, 2, 1)
        new_features = self.conv3(self.conv2(self.conv1(input_ids)))
        pred = self.dense2(self.dense1(new_features))

        return pred


if __name__ == '__main__':
    # data_atac = torch.rand((4, 1344, 5)).float().cuda()
    # data_rna = torch.rand((4, 1344, 5)).float().cuda()
    # model = BassetIntegrate(atac_cells=1000, rna_cells=1000).cuda()
    # model.train()
    # output = model(data_atac, data_rna)
    # atac_feature, rna_feature = model.get_embedding()
    # print(model)
    model = DeepSEA(n_cells=2034).cuda()
    print(model)
    data_atac = torch.rand((4, 1344, 4)).float().cuda()
    output = model(data_atac)
