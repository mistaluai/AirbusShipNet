import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsMixup(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, target):
        return F.binary_cross_entropy_with_logits(inputs, target, pos_weight=self.pos_weight)