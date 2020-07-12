import torch
import torch.utils.data
from torch import nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, conv_filts, res_filts,
                 skip_filts, dilation_rate, res=True, skip=True):
        super(ResidualBlock, self).__init__()
        self.res = res
        self.skip = skip

        padding = int((kernel_size + ((kernel_size-1)*(dilation_rate-1)) - 1)/2)

        self.conv_filt = nn.Sequential(
            nn.Conv1d(in_channels=conv_filts, out_channels=res_filts, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation_rate, bias=False),
            nn.Tanh())

        self.conv_gate = nn.Sequential(
            nn.Conv1d(in_channels=conv_filts, out_channels=res_filts, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation_rate, bias=False), nn.Sigmoid())

        self.conv_res = nn.Sequential(
            nn.Conv1d(in_channels=res_filts, out_channels=conv_filts, kernel_size=1, stride=1,
                      padding=0, dilation=dilation_rate, bias=False))

        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_channels=res_filts, out_channels=skip_filts, kernel_size=1, stride=1,
                      padding=0, dilation=dilation_rate, bias=False))

    def forward(self, x):
        outputs = dict()
        activation = self.conv_filt(x) * self.conv_gate(x)
        if self.res:
            outputs['res'] = self.conv_res(activation)
            outputs['res'] = outputs['res'] + x

        if self.skip:
            outputs['skip'] = self.conv_skip(activation)

        return outputs


class HaifaNetVPT(nn.Module):
    def __init__(self, num_classes, feature_opt='None', feature_len=0, gap_norm_opt='batch_norm'):
        super(HaifaNetVPT, self).__init__()
        self.num_classes = num_classes
        self.feature_opt = feature_opt
        self.feature_len = feature_len
        self.gap_norm_opt = gap_norm_opt

        conv_filts = 128
        res_filts = 128
        skip_filts = 128
        kernel_size = 3

        self.activation_size = 512
        padding = kernel_size//2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_filts, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False, dilation=1),
            nn.MaxPool1d(kernel_size=kernel_size, stride=2, padding=padding),
        )
        self.res_block2 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=2, res=True, skip=True)

        self.res_block3 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=4, res=True, skip=True)

        self.res_block4 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=8, res=True, skip=True)

        self.res_block5 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=16, res=True, skip=True)

        self.res_block6 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=32, res=True, skip=True)

        self.res_block7 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=64, res=True, skip=True)

        self.res_block8 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=128, res=True, skip=True)

        self.res_block9 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                        skip_filts=skip_filts, dilation_rate=256, res=True, skip=True)

        self.res_block10 = ResidualBlock(kernel_size=kernel_size, conv_filts=conv_filts, res_filts=res_filts,
                                         skip_filts=skip_filts, dilation_rate=512, res=False, skip=True)

        self.tail = nn.Sequential(
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=skip_filts, out_channels=256, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=padding),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False, dilation=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=padding),
            nn.Dropout(p=0.3)
        )

        if gap_norm_opt == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(num_features=self.activation_size)

        ext_rep_size = self.feature_len
        fc_layer_in_size = self.activation_size + ext_rep_size*int('concat' in self.feature_opt.lower())
        self.fc_layer = nn.Linear(in_features=fc_layer_in_size, out_features=self.num_classes, bias=False)

        self.relu = nn.ReLU()

        self.gradients = None

    def forward(self, x, features=None):
        skips = list()

        out = self.conv1(x)

        outputs = self.res_block2(out)
        skips.append(outputs['skip'])

        outputs = self.res_block3(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block4(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block5(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block6(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block7(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block8(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block9(outputs['res'])
        skips.append(outputs['skip'])

        outputs = self.res_block10(outputs['res'])
        skips.append(outputs['skip'])

        output = 0
        for skip in skips:
            output += skip

        output = self.tail(output)

        gap = torch.mean(output, dim=2)

        if self.gap_norm_opt == 'batch_norm':
            gap = self.batch_norm(gap)

        if ('concat' in self.feature_opt.lower()) and (self.feature_len > 0):
            gap = torch.cat([gap, features], dim=1)

        logits = self.fc_layer(gap)
        weight_fc = list(self.fc_layer.parameters())[0][:, :output.shape[1]]
        weight_fc_tile = weight_fc.repeat(output.shape[0], 1, 1)

        cam = torch.bmm(weight_fc_tile, output)
        return logits, cam, gap


from torch import nn
import torch


class FC_FeatureNet(nn.Module):
    def __init__(self, num_classes, feature_len=0):
        super(FC_FeatureNet, self).__init__()
        hidden_layer_size = 128

        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=feature_len, out_features=hidden_layer_size, bias=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_layer_size, out_features=num_classes, bias=True)

    def forward(self, x, features=None):
        x = self.fc_layer1(x)
        rep = self.relu(x)
        x = self.fc_layer2(rep)
        return x, rep


class OLD_multi_FC_FeatureNet(nn.Module):
    def __init__(self, rep_size=512, out_size=12):
        super(multi_FC_FeatureNet, self).__init__()
        hidden_layer_size = 128

        self.out_size = out_size
        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=rep_size, out_features=hidden_layer_size, bias=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size, bias=True)

        # per feature
        self.fc_layer_f1 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f2 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f3 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f4 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f5 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f6 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f7 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f8 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f9 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f10 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f11= nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)
        self.fc_layer_f12 = nn.Linear(in_features=hidden_layer_size, out_features=1, bias=True)

    def forward(self, x, features=None):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        x = self.relu(x)

        # per feature
        x1 = self.fc_layer_f1(x)
        x2 = self.fc_layer_f2(x)
        x3 = self.fc_layer_f3(x)
        x4 = self.fc_layer_f4(x)
        x5 = self.fc_layer_f5(x)
        x6 = self.fc_layer_f6(x)
        x7 = self.fc_layer_f7(x)
        x8 = self.fc_layer_f8(x)
        x9 = self.fc_layer_f9(x)
        x10 = self.fc_layer_f10(x)
        x11 = self.fc_layer_f11(x)
        x12 = self.fc_layer_f12(x)

        x = torch.cat([x1, x2, x3, x4, x5, x6,x7,x8, x9, x10, x11, x12], dim=1)

        return x


class multi_FC_FeatureNet(nn.Module):
    def __init__(self, rep_size=512, out_size=17):
        super(multi_FC_FeatureNet, self).__init__()
        hidden_layer_size = 128

        self.out_size = out_size
        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=rep_size, out_features=hidden_layer_size, bias=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size, bias=True)
        self.fc_layer3 = nn.Linear(in_features=hidden_layer_size, out_features=out_size, bias=True)


    def forward(self, x, features=None):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        rep = self.relu(x)
        x = self.fc_layer3(rep)

        return x


class External_to_class(nn.Module):
    def __init__(self, feature_len=12, num_classes=3):
        super(External_to_class, self).__init__()
        hidden_layer_size1 = 36
        hidden_layer_size2 = 128
        hidden_layer_size3 =64

        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=feature_len, out_features=hidden_layer_size1, bias=True)
        self.fc_layer2 = nn.Linear(in_features=hidden_layer_size1, out_features=hidden_layer_size2, bias=True)
        self.fc_layer3 = nn.Linear(in_features=hidden_layer_size2, out_features=hidden_layer_size1, bias=True)
        self.fc_layer4 = nn.Linear(in_features=hidden_layer_size1, out_features=hidden_layer_size3, bias=True)
        self.fc_layer5 = nn.Linear(in_features=hidden_layer_size3, out_features=num_classes, bias=True)

    def forward(self, x, features=None):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        x = self.relu(x)
        x = self.fc_layer3(x)
        x = self.relu(x)
        rep = self.fc_layer4(x)
        x = self.fc_layer5(rep)

        return x, rep


class OneFC(nn.Module):
    def __init__(self, rep_size=12, num_classes=3):
        super(OneFC, self).__init__()

        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=rep_size, out_features=num_classes, bias=True)
        self.fc_layer0 = nn.Linear(in_features=rep_size, out_features=rep_size, bias=True)

    def forward(self, x, features=None):
        x = self.fc_layer0(x)
        x = self.relu(x)
        x = self.fc_layer1(x)

        return x


class Rep2Label(nn.Module):
    def __init__(self, in_size=512, num_classes=3):
        super(Rep2Label, self).__init__()
        self.relu = nn.ReLU()
        self.fc_layer1 = nn.Linear(in_features=in_size, out_features=256, bias=True)
        self.fc_layer2 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.fc_layer3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)

    def forward(self, x, features=None):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        x = self.relu(x)
        x = self.fc_layer3(x)
        return x
