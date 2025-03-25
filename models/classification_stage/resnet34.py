import torch
import torch.nn as nn
import torchvision.models as models 

class ResNet34(nn.Module):
    def __init__(self, fc_layer=None) -> None:
        super(ResNet34, self).__init__()
        self.model = self._init_backbone(
            models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1, progress=False),
            fc_layer
        )

    def _init_backbone(self, backbone: nn.Module, fc_layer) -> nn.Module:
        in_features = backbone.fc.in_features

        if fc_layer is None:
            fc_layer = nn.Linear(in_features, 1)

        backbone.fc = nn.Sequential(
           *fc_layer
        )
        return backbone

    def get_layer(self, layer_name: str):
        try:
            return getattr(self.model, layer_name)
        except AttributeError as e:
            raise AttributeError(f"Layer '{layer_name}' not found in the model.") from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)