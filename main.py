# torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# standary library
import datetime
from typing import *
from pathlib import Path

# third-party library
import tqdm
import h5py
import numpy as np
from colorama import Fore, Style, init

# my library
from dataset import S3DISDataset
from network1D import PointNetSegmentation1D
from helper import load_hdf5, visualize_xyz_label, visualize_xyz_rgb, num2label, convert_leagal_path, PathConfig

init(autoreset=True)


class NetworkTrainer(object):
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(self, network: nn.Module):
        super(NetworkTrainer, self).__init__()
        self.train_ds = S3DISDataset(split="train")
        self.val_ds = S3DISDataset(split="val")
        self.test_ds = S3DISDataset(split="test")

        self.net = network
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=1e-4)

        self.dtype = self.net.dtype

        # saving paths
        name = self.net.__class__.__name__
        start_time = convert_leagal_path(str(datetime.datetime.now()))
        writer_path = PathConfig.runs / f"{name}" / f"{start_time}"
        checkpoint_path = PathConfig.checkpoints / f"{name}" / f"{start_time}-best.pt"
        self.writer = SummaryWriter(log_dir=writer_path)
    
    def __del__(self):
        self.writer.close()

    def train(self,n_epoch: int = 1000, early_stop: int = 200):
        train_loader = data.DataLoader(self.train_ds, batch_size=64, shuffle=True, num_workers=2)
        val_loader = data.DataLoader(self.val_ds, batch_size=64, shuffle=False, num_workers=2)

        early_stop_cnt: int = 0
        for epoch in (tt := tqdm.trange(n_epoch)):
            x: torch.Tensor
            y: torch.Tensor
            y_pred: torch.Tensor
            loss: torch.Tensor

            # set information
            tt.set_description(desc=f"Epoch [{Fore.GREEN}{epoch}{Fore.RESET}/{n_epoch}]")
            # train
            self.net.train()
            for step, (x, y) in train_loader:
                self.net.zero_grad()
                x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=np.double,
                                                                                  device=self.available_device)
                y_pred = self.net(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optim.step()
                self.writer.add_scalar(tag="/loss/training", scalar_value=loss.item(), global_step=step + epoch * len(train_loader))

            # val
            self.net.eval()
            with torch.no_grad():
                for step, (x, y) in val_loader:
                    self.net.zero_grad()
                    x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=np.double,
                                                                                      device=self.available_device)
                    y_pred = self.net(x)
                    loss = self.loss_func(y_pred, y)
                    self.writer.add_scalar(tag="loss/train", scalar_value=loss.item(), global_step=step + epoch * len(val_loader))

    @torch.no_grad()
    def test(self):
        test_loader = data.DataLoader(self.test_ds, batch_size=64, shuffle=False, num_workers=2)
        for step, (x, y) in test_loader:
            self.net.zero_grad()
            x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=np.double,
                                                                              device=self.available_device)
            y_pred = self.net(x)
            loss = self.loss_func(y_pred, y)

if __name__ == "__main__":
    # todo: 写完训练代码, 包括: early stop, summary writer, save checkpoints
    # todo: helper中写一个PathConfig, 规划项目的路径, 具体包括: runs, checkpoints, base
    # todo: helper中需要评价性能指标, 语义分割的性能指标, 包括: mIOU, IOU, AP, mAP, Dice coefficient
    # todo: 训练代码中规范化输出格式
    # todo: 有时间的话改改preprocess, 修改的内容就是z不限制, x, y限制, 最后得到一样的数据, 还有降采样问题
    # todo: 写完 network2d
    trainer = NetworkTrainer(PointNetSegmentation1D(in_features=3, predicted_cls=14))
