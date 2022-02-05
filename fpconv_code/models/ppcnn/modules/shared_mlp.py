import torch.nn as nn

__all__ = ['SharedMLP']


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1, last_relu=True, last_bn=True):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for i, oc in enumerate(out_channels):
            layers.extend([conv(in_channels, oc, 1)])
            if i < len(out_channels) - 1:
                layers.extend([bn(oc), nn.ReLU(True)])
            else:
                if last_bn: layers.extend([bn(oc)])
                if last_relu: layers.extend([nn.ReLU(True)])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)

