# torch library
import torch
import torch.nn as nn


class SharedMLP1D(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
        /,
        activation_fn: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        # save arguments
        self.in_features = in_features
        self.out_features = out_features

        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1
        )

        self.bn = nn.BatchNorm1d(out_features)

        self.activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, in_features, point_num]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class PointNetClassification1D(nn.Module):
    def __init__(
        self, in_features: int, predicted_cls: int,
        with_feature_transform: bool = False
    ) -> None:
        super().__init__()
        if not with_feature_transform:
            raise NotImplemented
        
        self.shared_mlp1 = SharedMLP1D(in_features=3, out_features=64)
        self.shared_mlp2 = SharedMLP1D(in_features=64, out_features=128)
        self.shared_mlp3 = SharedMLP1D(in_features=128, out_features=1024)

        
