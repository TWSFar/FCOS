import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_feaures_layer=5, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, 4, kernel_size=3, padding=1)

        # learnable parameter for scales
        self.scale_div = nn.Parameter(torch.ones(num_feaures_layer))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)
                nn.init.normal_(m.weight, mean=0, std=0.01)

        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)

    def forward(self, x, i):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = (self.output(out) * self.scale_div[i]).exp()

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)
