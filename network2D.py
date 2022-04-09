# torch library
import h5py
import numpy as np
import torch
import torch.nn as nn


class SharedMLP2D(nn.Module):
    """
    SharedMLP implementation with 2D convolution and 2D batch normalization
    Notes:
        input should be [batch, channel, point_num, feature_num], channel should be kept as 1
    """

    def __init__(
            self, in_features: int, out_features: int,
            /,
            activation_fn: nn.Module = nn.ReLU
    ):
        super(SharedMLP2D, self).__init__()
        # save arguments
        self.in_features = in_features
        self.out_features = out_features

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_features,
            kernel_size=(1, in_features),
            stride=1,
            padding=0
        )

        self.bn = nn.BatchNorm2d(out_features)
        # 最后的参数的问题
        self.bn = nn.BatchNorm2d(out_features, eps=0, momentum=0, affine=False, track_running_stats=False)

        self.activation = activation_fn()

    def forward(self, x: torch.Tensor):
        """
        forward pass of SharedMLP
        Args:
            x: torch.Tensor, should be [batch, channel, point_num, in_feature_num]
        Returns:
            y: torch.Tensor, will be [batch, channel, point_num, out_feature_num]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 3, 2, 1)
        x = self.activation(x)
        return x


class PointNetSegmentation2D(nn.Module):
    def __init__(
            self, in_features: int, predicted_cls: int,
            with_feature_transform: bool = False
    ):
        super(PointNetSegmentation2D, self).__init__()
        if with_feature_transform:
            raise NotImplementedError

        # Encoder
        self.shared_mlp1 = SharedMLP2D(in_features, 64)
        self.shared_mlp11 = SharedMLP2D(64, 64)
        self.shared_mlp2 = SharedMLP2D(64, 128)
        self.shared_mlp3 = SharedMLP2D(128, 1024)

        # Decoder
        self.shared_mlp4 = SharedMLP2D(1088, 512)
        self.shared_mlp5 = SharedMLP2D(512, 256)
        self.shared_mlp6 = SharedMLP2D(256, 128)

        self.shared_mlp_prediction = SharedMLP2D(128, predicted_cls)

        for module in self.modules():
            if (s:=getattr(module, "weight", None)) is not None:
                self.dtype = s.dtype
                break

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: [batch, channel, point_num, in_feature_num]
        point_num = x.size(2)
        x = self.shared_mlp1(x)
        pointwise_feature = self.shared_mlp11(x)
        x = self.shared_mlp2(pointwise_feature)
        x = self.shared_mlp3(x)
        global_feature: torch.Tensor = x.max(dim=2, keepdim=True)[0].repeat(1, 1, point_num, 1)

        # segmentation part
        global_feature = torch.cat((pointwise_feature, global_feature), dim=-1)
        x = self.shared_mlp4(global_feature)
        x = self.shared_mlp5(x)
        x = self.shared_mlp6(x)

        y_predicted = self.shared_mlp_prediction(x)
        return y_predicted.squeeze().permute(0, 2, 1)


if __name__ == "__main__":
    pn = PointNetSegmentation2D(in_features=3, predicted_cls=14)
    print(pn.dtype)

    # 128 batch, 1024 points with 3 features
    points = torch.randn(128, 1, 1024, 3)
    result = pn(points)
    print(result.shape)
