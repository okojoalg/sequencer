import math

import torch
from torch import nn

from erf.save_output import SaveOutput
from utils.helpers import rgetattr


class ERFNet(nn.Module):
    def __init__(self, model, attrs, channels_last=False):
        super().__init__()
        self.model = model
        self.save_output = SaveOutput()
        self.channels_last = channels_last
        hook_handles = []
        for module in attrs:
            layer = rgetattr(self.model, module)
            handle = layer.register_forward_hook(self.save_output)
            hook_handles.append(handle)

    def forward(self, x):
        features = []
        _ = self.model(x)
        for feature in self.save_output.outputs:
            if len(feature.shape) == 3:
                feature = feature[:, feature.shape[1] // 2 - int(math.sqrt(feature.shape[1])) // 2, :]
            elif self.channels_last:
                feature = feature[:, feature.shape[1] // 2 - 1, feature.shape[2] // 2 - 1, :]
            else:
                feature = feature[:, :, feature.shape[2] // 2 - 1, feature.shape[3] // 2 - 1]
            features.append(torch.sum(feature))
        self.save_output.clear()

        return features
