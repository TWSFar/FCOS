import math
import torch.nn as nn


class HRNET_NECK(nn.Module):
    def __init__(self, in_channels, feature_size=256):
        super(HRNET_NECK, self).__init__()
        C2_size, C3_size, C4_size, C5_size = in_channels

        # P2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # P3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # P4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # P5
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P6 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
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

        C2, C3, C4, C5 = inputs

        P2_x = self.P2_1(C2)
        P2_downsample = self.P2_2(P2_x)

        P3_x = self.P3_1(C3)
        P3_x = P2_downsample + P3_x
        P3_downsample = self.P3_2(P3_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P3_downsample
        P4_downsample = self.P4_2(P4_x)

        P5_x = self.P5_1(C5)
        P5_x = P5_x + P4_downsample

        P6_x = self.P6(P5_x)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
