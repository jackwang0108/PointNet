# torch library
import h5py
import numpy as np
import torch
import torch.nn as nn


class SharedMLP1D(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 batch_norm: bool = True,
                 activation_fn: nn.Module = nn.ReLU):
        super(SharedMLP1D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.Conv1d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d


class SharedMLP2D(nn.Module):
    """
    SharedMLP implementation with 2D convolution and 2D batch normalization
    Notes:
        input should be [batch, channel, point_num, feature_num], channel should be kept as 1
    """

    def __init__(self, in_features: int, out_features: int,
                 batch_norm: bool = True,
                 activation_fn: nn.Module = nn.ReLU):
        super(SharedMLP2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=out_features,
                              kernel_size=(1, in_features),
                              stride=(1, 1),
                              padding=0)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_features=1)
        self.activation = activation_fn()

        self.dtype = self.weight.dtype

    def forward(self, x: torch.Tensor):
        """
        forward pass of SharedMLP
        Args:
            x: torch.Tensor, should be [batch, channel, point_num, in_feature_num]
        Returns:
            y: torch.Tensor, will be [batch, channel, point_num, out_feature_num]
        """
        x = self.conv(x)
        x = x.permute(0, 3, 2, 1)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class PointNet2D(nn.Module):
    def __init__(self, in_features: int, output: str = "segmentation"):
        super(PointNet2D, self).__init__()

        # self.SharedMLP = nn.Conv2d(in_channels=)

    def forward(self, x):
        # sharedMLP = nn.Conv2d()
        pass


if __name__ == "__main__":
    from helper import load_hdf5, visualize_xyz_label, visualize_xyz_rgb, DatasetPaths, voxelize, load_txt

    # with load_hdf5(DatasetPaths.S3DIS.s3dis_my_processed_h5_data["Area_1"]) as f:
    #     for room in f.keys():
    #         for batch_name in f[room].keys():
    #             batch = torch.from_numpy(np.asarray(f[room][batch_name]))
    #             x, y = torch.hsplit(batch, indices=[6])
    #             break

    # data = load_txt(DatasetPaths.S3DIS.s3dis_original_xyzrgb_data["Area_1"]["WC_1"]["data"], with_label=True)
    # split for further process
    # xyz, rgb, label = np.hsplit(data, indices_or_sections=[3, 6])
    # rgb /= 255
    # # voxelization
    # voxelization_index = voxelize(xyz=xyz)
    # for voxel_grid_idx in np.unique(voxelization_index):
    #     # process each voxel grid
    #     idx = (voxelization_index == voxel_grid_idx)
    #     point, color, l = xyz[idx], rgb[idx], label[idx]

    with h5py.File(DatasetPaths.S3DIS.s3dis_processed_h5_data["ply_data_all_1"], mode="r") as f:
        data, label = np.asarray(f["data"], dtype=float), np.asarray(f["label"], dtype=float)
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        # mlp = SharedMLP(in_features=6, out_features=16).double()
        mlp = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=3, padding=0)
        for batch_x, batch_y in zip(data, label):
            splited = torch.hsplit(batch_x, [3, 6])
            # batch_x = torch.concat((splited[2], splited[1]), dim=1).unsqueeze(dim=0).unsqueeze(dim=0)
            batch_x = torch.concat((splited[2], splited[1]), dim=1)
            mlp(batch_x.T.to(mlp.dtype))
            print(batch_y)
            break
