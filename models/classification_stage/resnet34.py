import torch
import torch.nn as nn
import torchvision.models as models 

class ResNet34(nn.Module):
    def __init__(self, hidden_units: int = 256, dropout: float= 0.5) -> None:
        super(ResNet34, self).__init__()
        self.dropout = dropout
        self.hidden_units = hidden_units
        self.model = self._init_backbone(
            models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        )

    def _init_backbone(self, backbone: nn.Module) -> nn.Module:
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, self.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_units, 1)
        )
        return backbone

    def get_layer(self, layer_name: str):
        try:
            return getattr(self.model, layer_name)
        except AttributeError as e:
            raise AttributeError(f"Layer '{layer_name}' not found in the model.") from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)