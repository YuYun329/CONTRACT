import torch.nn as nn
import torch.nn.functional as F
import torch


class _conv_layer(nn.Module):
    def __init__(self, number_input_features, filters_mul, kernel_size, pool_size):
        super(_conv_layer, self).__init__()
        self.pool_size = pool_size

        self.add_module('conv', nn.Conv1d(in_channels=number_input_features,
                                          out_channels=int(number_input_features * filters_mul),
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2,
                                          bias=True))
        self.add_module('norm', nn.BatchNorm1d(num_features=int(number_input_features * filters_mul)))
        self.add_module('relu', nn.ReLU(inplace=True))

        if self.pool_size > 0:
            self.add_module('pool', nn.MaxPool1d(kernel_size=pool_size))

    def forward(self, input):
        new_features = self.relu(self.norm(self.conv(input)))
        if self.pool_size > 0:
            new_features = self.pool(new_features)
        return new_features


class _conv_block(nn.ModuleDict):
    def __init__(self, num_layers, number_input_features, filters_mul, kernel_size, pool_size):
        super(_conv_block, self).__init__()
        for i in range(num_layers):
            layer = _conv_layer(number_input_features=number_input_features,
                                filters_mul=filters_mul,
                                kernel_size=kernel_size,
                                pool_size=pool_size)
            self.add_module('conv_layer%d' % (i + 1), layer)
            number_input_features = int(number_input_features * filters_mul)

    def forward(self, input):
        new_features = input
        for name, layer in self.items():
            new_features = layer(new_features)

        return new_features


class _dense_layer(nn.Module):
    def __init__(self, number_input_features, bn_size, growth_rate, drop_rate):
        super(_dense_layer, self).__init__()

        # 1*1 conv for reducing channel numbers
        self.add_module('norm1', nn.BatchNorm1d(num_features=number_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv1d(in_channels=number_input_features,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           padding=0,
                                           bias=False))

        # 3*3 conv for learning features
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv1d(in_channels=bn_size * growth_rate,
                                           out_channels=growth_rate,
                                           kernel_size=3,
                                           dilation=1,
                                           padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        # (List[Tensor] -> Tensor)
        prev_features = torch.cat(inputs, 1)
        bn_output = self.conv1(self.relu1(self.norm1(prev_features)))
        return bn_output

    def forward(self, input):
        bn_output = self.bn_function(input)
        new_features = self.conv2(self.relu2(self.norm2(bn_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate)

        return new_features


class _dense_block(nn.ModuleDict):
    def __init__(self, num_layers, number_input_features, bn_size, growth_rate, drop_rate):
        super(_dense_block, self).__init__()
        for i in range(num_layers):
            layer = _dense_layer(number_input_features=number_input_features + i * growth_rate,
                                 bn_size=bn_size,
                                 growth_rate=growth_rate,
                                 drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, 1)


class BasenjiV1(nn.Module):
    def __init__(self, crop_len=0, n_cells=2034):
        super(BasenjiV1, self).__init__()
        self.stem_conv = _conv_block(num_layers=1,
                                     number_input_features=4,
                                     filters_mul=16,
                                     kernel_size=15,
                                     pool_size=2)

        num_conv_layer = 3
        self.conv_layer = _conv_block(num_layers=num_conv_layer,
                                      number_input_features=64,
                                      filters_mul=1.125,
                                      kernel_size=5,
                                      pool_size=4)

        num_dense_layer = 7
        self.dense_layer = _dense_block(num_layers=num_dense_layer,
                                        number_input_features=91,
                                        bn_size=2,
                                        growth_rate=32,
                                        drop_rate=0.25)

        self.crop_len = crop_len

        # Basenji does not use pool
        self.pred_layer = nn.Sequential(nn.AdaptiveMaxPool1d(output_size=1),
                                        nn.Flatten(start_dim=1),
                                        nn.Linear(in_features=315,
                                                  out_features=n_cells),
                                        nn.Sigmoid())
        # self.layer_temp = nn.Sequential(nn.AdaptiveMaxPool1d(output_size=1))

    def forward(self, input_ids):
        """
        :param sequence: [batch_size, 4, 1344]
        :return: pred: [batch_size, n_cells]
        """
        input_ids = input_ids.permute(0, 2, 1)
        stem_conv = self.stem_conv(input_ids)
        conv_features = self.conv_layer(stem_conv)
        dense_features = self.dense_layer(conv_features)
        if self.crop_len > 0:
            _, _, seq_len = dense_features.size()
            dense_features = dense_features[:, :, self.crop_len:seq_len - self.crop_len]
        pred = self.pred_layer(dense_features)

        return pred
