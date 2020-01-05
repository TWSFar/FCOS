import math
import torch.nn as nn


# Channel-level attention (SE module)
class GateLayer_ca(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GateLayer_ca, self).__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sq(x).view(b, c)
        y = self.ex(y).view(b, c, 1, 1)
        return y.expand_as(x)


# Global-level attention
class GateLayer_ga(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GateLayer_ga, self).__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sq(x).view(b, c)
        y = self.ex(y).view(b, 1, 1, 1)
        return y.expand_as(x)


class FPN_GATE(nn.Module):
    def __init__(self, in_channels, feature_size=256):
        super(FPN_GATE, self).__init__()
        C3_size, C4_size, C5_size = in_channels

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_ca = GateLayer_ca(feature_size)
        self.P5_ga = GateLayer_ga(feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_ca = GateLayer_ca(feature_size)
        self.P4_ga = GateLayer_ga(feature_size)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        _U = self.P5_ca(P5_upsampled_x)
        _e = self.P5_ga(P5_upsampled_x)
        _V = _e * _U
        P5_upsampled_x = P5_upsampled_x + _V
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        _U = self.P4_ca(P4_upsampled_x)
        _e = self.P4_ga(P4_upsampled_x)
        _V = _e * _U
        P4_upsampled_x = P4_upsampled_x + _V
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]