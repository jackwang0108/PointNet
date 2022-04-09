# torch library
import torch
import torch.utils.data as data

# standard library
from typing import *
from numbers import Number
from pathlib import Path

# third-party library
import numpy as np

# my library
from helper import DatasetPaths, visualize_xyz_rgb, visualize_xyz_label, load_hdf5


class S3DISDataset(data.Dataset):
    def __init__(self, split: str, verbose: bool = False):
        super(S3DISDataset, self).__init__()
        assert split in ["train", "test", "validation", "val"]
        self.split = split
        self.all_data = []
        self.all_label = []
        for file_name, file_path in DatasetPaths.S3DIS.s3dis_processed_h5_data.items():
            if verbose:
                print(f"Reading {file_name}")
            with load_hdf5(file_path) as f:
                self.all_data.append(np.asarray(f["data"]))
                self.all_label.append(np.asarray(f["label"]))
        self.all_data = [torch.from_numpy(i) for i in self.all_data]
        self.all_label = [torch.from_numpy(i) for i in self.all_label]

        self.all_data = torch.concat(self.all_data, dim=0)
        self.all_label = torch.concat(self.all_label, dim=0)

        xyz_origin, rgb_normalized, xyz_normalized = torch.split(self.all_data, dim=2, split_size_or_sections=3)
        self.all_data = torch.concat((xyz_normalized, rgb_normalized), dim=2).permute(0, 2, 1)

        if (p := Path(__file__).resolve().parent.joinpath("trainval.npz")).exists():
            f = np.load(p)
            train_idx, val_idx, test_idx = f["train_idx"], f["val_idx"], f["test_idx"]
        else:
            idx = np.arange(len(self))
            np.random.shuffle(idx)
            l1, l2 = int(len(self) * 0.8), int(len(self) * 0.1)
            train_idx, val_idx, test_idx = np.hsplit(idx, [l1, l1 + l2])
            np.savez(p, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        if split == "train":
            idx = train_idx
        elif split == "val":
            idx = val_idx
        else:
            idx = test_idx

        self.all_data, self.all_label = self.all_data[idx], self.all_label[idx]

    def __len__(self):
        return len(self.all_label)

    def __getitem__(self, item) -> torch.Tensor:
        return self.all_data[item], self.all_label[item]


if __name__ == "__main__":
    # with load_hdf5(DatasetPaths.S3DIS.s3dis_processed_h5_data["ply_data_all_0"]) as f:
    #     print(f.keys())
    import torch.nn as nn
    from network1D import PointNetSegmentation1D

    ds = S3DISDataset(split="train")
    pn = PointNetSegmentation1D(in_features=6, predicted_cls=14)
    loss_func = nn.CrossEntropyLoss()
    loader = data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    for x, y in loader:
        # treat point cloud segmentation as image, then cross entropy on image
        y_pred = pn(x)
        # loss = loss_func(y_pred.unsqueeze(dim=-1), y.long().unsqueeze(dim=-1))
        # however, the upper is the same as
        loss = loss_func(y_pred, y.long())
        break
