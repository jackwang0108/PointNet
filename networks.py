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
        raise NotImplementedError
        if not with_feature_transform:
            raise NotImplemented

        self.shared_mlp1 = SharedMLP1D(in_features=in_features, out_features=64)
        self.shared_mlp2 = SharedMLP1D(in_features=64, out_features=128)
        self.shared_mlp3 = SharedMLP1D(in_features=128, out_features=1024)


class PointNetSegmentation1D(nn.Module):
    def __init__(
            self, in_features: int, predicted_cls: int,
            with_feature_transform: bool = False
    ):
        super().__init__()
        if with_feature_transform:
            raise NotImplementedError

        # Encoder
        self.shared_mlp1 = SharedMLP1D(in_features, 64)
        self.shared_mlp11 = SharedMLP1D(64, 64)
        self.shared_mlp2 = SharedMLP1D(64, 128)
        self.shared_mlp3 = SharedMLP1D(128, 1024)

        self.shared_mlp4 = SharedMLP1D(1088, 512)
        self.shared_mlp5 = SharedMLP1D(512, 256)
        self.shared_mlp6 = SharedMLP1D(256, 128)

        self.shared_mlp_prediction = SharedMLP1D(128, predicted_cls)

        for module in self.modules():
            if (s:=getattr(module, "weight", None)) is not None:
                self.dtype = s.dtype
                break

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: [batch, feature_num, point_num)
        point_num = x.size(2)
        x = self.shared_mlp1(x)
        pointwise_feature = self.shared_mlp11(x)
        x = self.shared_mlp2(pointwise_feature)
        x = self.shared_mlp3(x)
        global_feature: torch.Tensor = x.max(dim=2, keepdim=True)[0].repeat(1, 1, point_num)

        # segmentation part
        global_feature = torch.cat((pointwise_feature, global_feature), dim=1)
        x = self.shared_mlp4(global_feature)
        x = self.shared_mlp5(x)
        x = self.shared_mlp6(x)

        y_predicted = self.shared_mlp_prediction(x)
        return y_predicted


if __name__ == "__main__":
    pn = PointNetSegmentation1D(in_features=3, predicted_cls=14)
    print(pn.dtype)

    # 128 batch, 1024 points with 3 features
    points = torch.randn(128, 3, 1024)
    result = pn(points)
    print(result.shape)




