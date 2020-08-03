import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift


class SubPixelBackProjection(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(SubPixelBackProjection, self).__init__()
        stride = upscale_factor
        padding = 1
        kernel_size = 3

        self.num_groups = num_groups



        self.upBlocks = nn.ModuleList()
        self.upPac = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upPac.append(ConvBlock(num_features, num_features * (upscale_factor ** 2),
                                             kernel_size=3, stride=1, padding=1, act_type=None, valid_padding=False))
            self.upBlocks.append(nn.PixelShuffle(upscale_factor))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))


        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.prelu = nn.PReLU(num_parameters=1, init=0.2)



    def forward(self, x):


        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_L = self.upPac[idx](LD_L)
            LD_H = self.prelu(self.upBlocks[idx](LD_L))

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)
        output = self.compress_out(output)
        return output



class SPBP(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_groups, upscale_factor, act_type = 'prelu', norm_type = None):
        super(SPBP, self).__init__()

        self.num_features = num_features
        self.upscale_factor = upscale_factor

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.prelu1 = nn.PReLU(num_parameters=1, init=0.2)
        self.feat_in = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.2)

        # Non-Linear block
        self.block = SubPixelBackProjection(num_features, num_groups, upscale_factor, act_type, norm_type)
        self.prelu3 = nn.PReLU(num_parameters=1, init=0.2)

        # reconstruction block
        self.conv4 = nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        
    def forward(self, x):
        x = self.sub_mean(x)
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        x = self.conv_in(x)
        x = self.prelu1(x)
        x = self.feat_in(x)
        x = self.prelu2(x)

        h = self.block(x)
        h = self.prelu3(h)

        h = self.pixel_shuffle(self.conv4(h))
        h = self.prelu(h)

        h = torch.add(inter_res, self.conv_out(h))
        h = self.add_mean(h)
        return h

