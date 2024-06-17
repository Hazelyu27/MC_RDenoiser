import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

__all__ = ['Single_Model']
device = torch.device("cuda:0")

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, input_channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(input_channel, input_channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(input_channel // reduction_ratio, input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channel)
        y = self.excitation(y).view(batch_size, channel, 1, 1)
        return x * y



class Ea_block(nn.Module):
    def __init__(self, in_channels, out_channels, se_reduction_ratio=16,leaky_relu_slope=0.1):
        # se_reduction_ratio 是SE的参数，se_reduction_ratio越大，越轻量，牺牲表达能力
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.ln1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        # self.relu = nn.Hardswish(inplace=True)
        self.se = SEBlock(out_channels, reduction_ratio=se_reduction_ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.se(out)
        return out

class Eb_block(nn.Module):
    def __init__(self, in_channels, out_channels, se_reduction_ratio=16,leaky_relu_slope=0.1):
        # se_reduction_ratio 是SE的参数，se_reduction_ratio越大，越轻量，牺牲表达能力
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.ln1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        # self.relu = nn.Hardswish(inplace=True)
        self.se = SEBlock(out_channels, reduction_ratio=se_reduction_ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.se(out)
        return out
class CFM(nn.Module):
    def __init__(self, in_channels1,in_channels2, out_channels1, out_channels2, dilation_rate=1,leaky_relu_slope=0.1):
        super(CFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels1, out_channels1, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate)
        self.conv2 = nn.Conv2d(in_channels2, out_channels2, kernel_size=3, padding=dilation_rate,
                               dilation=dilation_rate)
        self.bn1 = nn.InstanceNorm2d(out_channels1)
        self.bn2 = nn.InstanceNorm2d(out_channels2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        out1_1 = self.bn1(out1)
        out2 = self.conv2(x2)
        out2_1 = self.bn2(out2)
        out = torch.cat([out1_1, out2_1], dim=1)
        out = self.leaky_relu(out)

        return out


class Decoder_student(nn.Module):
    def __init__(self, in_channels, out_channels, se_reduction_ratio=16,leaky_relu_slope=0.1):
        # se_reduction_ratio 是SE的参数，se_reduction_ratio越大，越轻量，牺牲表达能力
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.ln1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        # self.relu = nn.Hardswish(inplace=True)
        self.se = SEBlock(out_channels, reduction_ratio=se_reduction_ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.se(out)
        return out



class Single_Model(nn.Module):
    def __init__(self, output_channels=3, input_channels=3, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.ea1 = Ea_block(input_channels, nb_filter[0])
        self.ea2 = Ea_block(nb_filter[0], nb_filter[1])
        self.ea3 = Ea_block(nb_filter[1], nb_filter[2]) # ea3输出128
        self.eb1 = Eb_block(input_channels, nb_filter[0])
        self.eb2 = Eb_block(nb_filter[0]+nb_filter[0], 2*(nb_filter[0]+nb_filter[0]))
        self.eb3 = Eb_block(2*(nb_filter[0]+nb_filter[0])+nb_filter[1], 8*nb_filter[0]+2*nb_filter[1])

        # fusion输入和输出通道相同
        self.fusion = CFM(128,384,128,384)
        self.de1 = Decoder_student(512,256)
        self.de2 = Decoder_student(384, 192)
        self.de3 = Decoder_student(224, 112)
        self.de4 = Decoder_student(112, 56)
        self.convfinal2 = nn.Conv2d(56, output_channels, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x1,x2):

        ea1_out = self.ea1(self.pool(x1))
        ea2_out = self.ea2(self.pool(ea1_out))
        ea3_out = self.ea3(self.pool(ea2_out))

        eb1_out = self.eb1(self.pool(x2))
        eb2_out = self.eb2(self.pool(torch.cat([ea1_out, eb1_out], 1)))
        eb3_out = self.eb3(self.pool(torch.cat([ea2_out, eb2_out], 1)))
        fusion_out = self.fusion(ea3_out,eb3_out)

        de1out = self.de1(self.up(fusion_out))
        de2out = self.de2(self.up(torch.cat([eb2_out, de1out], 1)))
        de3out = self.de3(self.up(torch.cat([eb1_out, de2out], 1)))
        d4out = self.de4(de3out)
        result = self.convfinal2(d4out)
        return result


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



class Teacher_Model(nn.Module):
    def __init__(self, output_channels=3, input_channels=3, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.ea1 = Ea_block(input_channels, nb_filter[0])
        self.ea2 = Ea_block(nb_filter[0], nb_filter[1])
        self.ea3 = Ea_block(nb_filter[1], nb_filter[2]) # ea3输出128
        self.eb1 = Eb_block(input_channels, nb_filter[0])
        self.eb2 = Eb_block(nb_filter[0]+nb_filter[0], 2*(nb_filter[0]+nb_filter[0]))
        self.eb3 = Eb_block(2*(nb_filter[0]+nb_filter[0])+nb_filter[1], 8*nb_filter[0]+2*nb_filter[1])

        # fusion输入和输出通道相同
        self.fusion = CFM(128,384,128,384)
        self.de1 = Decoder_student(512,256)
        self.de2 = Decoder_student(384, 192)
        self.de3 = Decoder_student(224, 112)
        self.de4 = Decoder_student(112, 56)
        self.convfinal2 = nn.Conv2d(56, output_channels, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.mlp = ThreeLayerMLP(input_size=56*1024*1024, hidden_size=1024, output_size=56*1024*1024)

    def forward(self, x1,x2):

        ea1_out = self.ea1(self.pool(x1))
        ea2_out = self.ea2(self.pool(ea1_out))
        ea3_out = self.ea3(self.pool(ea2_out))

        eb1_out = self.eb1(self.pool(x2))
        eb2_out = self.eb2(self.pool(torch.cat([ea1_out, eb1_out], 1)))
        eb3_out = self.eb3(self.pool(torch.cat([ea2_out, eb2_out], 1)))
        fusion_out = self.fusion(ea3_out,eb3_out)
        # up_f = self.up(fusion_out)
        # print(up_f.shape)
        # print(outx1_3.shape)

        de1out = self.de1(self.up(fusion_out))
        # print(eb2_out.shape)
        # print(de1out.shape)
        de2out = self.de2(self.up(torch.cat([eb2_out, de1out], 1)))
        de3out = self.de3(self.up(torch.cat([eb1_out, de2out], 1)))
        d4out = self.de4(de3out)
        result = self.convfinal2(d4out)
        # print(d4out.shape)
        # d4_flatten = d4out.view(32, -1)

        # mlp_out = self.mlp(d4out)
        # print(mlp_out.shape)

        # result = self.convfinal2(d4out)
        return result




