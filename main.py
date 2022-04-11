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
from network2D import PointNetSegmentation2D
from helper import labels, convert_legal_path, PathConfig, Evaluator

init(autoreset=True)


class Network1DTrainer(object):
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(self, network: nn.Module):
        super(Network1DTrainer, self).__init__()
        self.train_ds = S3DISDataset(split="train", verbose=True)
        self.val_ds = S3DISDataset(split="val", verbose=True)
        # self.test_ds = S3DISDataset(split="test", verbose=True)

        self.net = network.to(device=self.available_device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=1e-4)

        self.dtype = self.net.dtype

        # saving paths
        name = self.net.__class__.__name__
        start_time = convert_legal_path(str(datetime.datetime.now()))
        writer_path = PathConfig.runs / f"{name}" / f"{start_time}"
        self.checkpoint_path = PathConfig.checkpoints / f"{name}" / f"{start_time}-best.pt"

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # summary writer
        self.writer = SummaryWriter(log_dir=writer_path)

        # evaluator
        self.evaluator = Evaluator()

    def __del__(self):
        self.writer.close()

    def train(self, n_epoch: int = 1000, early_stop: int = 200):
        # other setting
        global labels
        digits = len(str(n_epoch))

        # training setting
        # on windows, num_workers should be set to 0! Otherwise, this will cause: RuntimeError: DataLoader worker
        # (pid(s) 3376, 18232) exited unexpectedly
        train_loader = data.DataLoader(self.train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = data.DataLoader(self.val_ds, batch_size=32, shuffle=False, num_workers=0)

        max_miou = 0
        early_stop_cnt: int = 0
        for epoch in (tt := tqdm.trange(n_epoch)):
            x: torch.Tensor
            y: torch.Tensor
            y_pred: torch.Tensor
            loss: torch.Tensor

            # set information
            tt.set_description(desc=f"Epoch [{Fore.GREEN}{epoch:>{digits}d}{Fore.RESET}/{n_epoch:>{digits}d}]")

            # train
            self.net.train()
            self.evaluator.reset()
            for step, (x, y) in enumerate(train_loader):
                self.net.zero_grad()
                x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=torch.long,
                                                                                  device=self.available_device)
                y_pred = self.net(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optim.step()
                # log
                self.writer.add_scalar(tag="loss/train", scalar_value=loss.item(),
                                       global_step=step + epoch * len(train_loader))

                self.evaluator.add_batch(pred=y_pred.argmax(dim=1), label=y)

            # add to tensorboard
            self.writer.add_scalar(tag="point accuracy/train", scalar_value=self.evaluator.Piont_Accuracy(),
                                   global_step=epoch)
            iou = self.evaluator.IOU(epcoh=epoch)[0]
            self.writer.add_scalar(tag="mIOU/train", scalar_value=iou[-1], global_step=epoch)
            acc = self.evaluator.Point_Accuracy_Class(epoch=epoch)[0]
            self.writer.add_scalar(tag="mAcc/train", scalar_value=acc[-1], global_step=epoch)

            # val
            self.net.eval()
            self.evaluator.reset()
            with torch.no_grad():
                for step, (x, y) in enumerate(val_loader):
                    self.net.zero_grad()
                    x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=torch.long,
                                                                                      device=self.available_device)
                    y_pred = self.net(x)
                    loss = self.loss_func(y_pred, y)

                    self.writer.add_scalar(tag="loss/validation", scalar_value=loss.item(),
                                           global_step=step + epoch * len(val_loader))

                    self.evaluator.add_batch(y_pred.argmax(dim=1), y)

                # add to tensorboard
                self.writer.add_scalar(tag="point accuracy/validation",
                                       scalar_value=self.evaluator.Piont_Accuracy(), global_step=epoch)

                # make table
                l: List = labels.copy()
                l.append("mIOU")
                iou, iou_t = self.evaluator.IOU(epcoh=epoch)
                self.writer.add_scalar(tag="mIOU/validation", scalar_value=iou[-1], global_step=epoch)
                l.pop(-1)
                l.append("mAcc")
                acc, acc_t = self.evaluator.Point_Accuracy_Class(epoch=epoch)
                self.writer.add_scalar(tag="mAcc/validation", scalar_value=acc[-1], global_step=epoch)

            if iou[-1] > max_miou:
                max_miou = iou[-1]
                early_stop_cnt = 0
                tt.write(f"{Fore.BLUE}Saving checkpoint at epcoch {epoch}, maximum mIOU: {max_miou:>.5f}")
                torch.save(self.net.state_dict(), self.checkpoint_path)
            else:
                early_stop_cnt += 1

            # log info
            tt.write(f"Epoch[{Fore.GREEN}{epoch}{Style.RESET_ALL}|{n_epoch}], early_stop_cnt: {early_stop_cnt}")
            tt.write(acc_t.table)
            tt.write("\n")
            tt.write(iou_t.table)
            tt.write("\n"*3)

            if early_stop_cnt >= early_stop:
                tt.write(f"{Fore.YELLOW}Early stopped at epoch: {epoch}")
                break

    @torch.no_grad()
    def test(self):
        test_loader = data.DataLoader(self.test_ds, batch_size=64, shuffle=False, num_workers=2)
        for step, (x, y) in test_loader:
            self.net.zero_grad()
            x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=np.double,
                                                                              device=self.available_device)
            y_pred = self.net(x)
            loss = self.loss_func(y_pred, y)

class Network2DTrainer(object):
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(self, network: nn.Module):
        super(Network2DTrainer, self).__init__()
        self.train_ds = S3DISDataset(split="train", verbose=True)
        self.val_ds = S3DISDataset(split="val", verbose=True)
        # self.test_ds = S3DISDataset(split="test", verbose=True)

        self.net = network.to(device=self.available_device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=1e-4)

        self.dtype = self.net.dtype

        # saving paths
        name = self.net.__class__.__name__
        start_time = convert_legal_path(str(datetime.datetime.now()))
        writer_path = PathConfig.runs / f"{name}" / f"{start_time}"
        self.checkpoint_path = PathConfig.checkpoints / f"{name}" / f"{start_time}-best.pt"

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # summary writer
        self.writer = SummaryWriter(log_dir=writer_path)

        # evaluator
        self.evaluator = Evaluator()

    def __del__(self):
        self.writer.close()

    def train(self, n_epoch: int = 1000, early_stop: int = 200):
        # other setting
        global labels
        digits = len(str(n_epoch))

        # training setting
        # on windows, num_workers should be set to 0! Otherwise, this will cause: RuntimeError: DataLoader worker
        # (pid(s) 3376, 18232) exited unexpectedly
        train_loader = data.DataLoader(self.train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = data.DataLoader(self.val_ds, batch_size=32, shuffle=False, num_workers=0)

        max_miou = 0
        early_stop_cnt: int = 0
        for epoch in (tt := tqdm.trange(n_epoch)):
            x: torch.Tensor
            y: torch.Tensor
            y_pred: torch.Tensor
            loss: torch.Tensor

            # set information
            tt.set_description(desc=f"Epoch [{Fore.GREEN}{epoch:>{digits}d}{Fore.RESET}/{n_epoch:>{digits}d}]")

            # train
            self.net.train()
            self.evaluator.reset()
            for step, (x, y) in enumerate(train_loader):
                self.net.zero_grad()
                x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=torch.long,
                                                                                  device=self.available_device)
                y_pred = self.net(x.unsqueeze(dim=1).permute(0, 1, 3, 2))
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optim.step()
                # log, this is very slow for 2D
                self.writer.add_scalar(tag="loss/train", scalar_value=loss.item(),
                                       global_step=step + epoch * len(train_loader))

                self.evaluator.add_batch(pred=y_pred.argmax(dim=1), label=y)

            # add to tensorboard
            self.writer.add_scalar(tag="point accuracy/train", scalar_value=self.evaluator.Piont_Accuracy(),
                                   global_step=epoch)
            iou = self.evaluator.IOU(epcoh=epoch)[0]
            self.writer.add_scalar(tag="mIOU/train", scalar_value=iou[-1], global_step=epoch)
            acc = self.evaluator.Point_Accuracy_Class(epoch=epoch)[0]
            self.writer.add_scalar(tag="mAcc/train", scalar_value=acc[-1], global_step=epoch)

            # val
            self.net.eval()
            self.evaluator.reset()
            with torch.no_grad():
                for step, (x, y) in enumerate(val_loader):
                    self.net.zero_grad()
                    x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=torch.long,
                                                                                      device=self.available_device)
                    y_pred = self.net(x.unsqueeze(dim=1).permute(0, 1, 3, 2))
                    loss = self.loss_func(y_pred, y)

                    # This is very slow
                    self.writer.add_scalar(tag="loss/validation", scalar_value=loss.item(),
                                           global_step=step + epoch * len(val_loader))

                    self.evaluator.add_batch(y_pred.argmax(dim=1), y)

                # add to tensorboard
                self.writer.add_scalar(tag="point accuracy/validation",
                                       scalar_value=self.evaluator.Piont_Accuracy(), global_step=epoch)

                # make table
                l: List = labels.copy()
                l.append("mIOU")
                iou, iou_t = self.evaluator.IOU(epcoh=epoch)
                self.writer.add_scalar(tag="mIOU/validation", scalar_value=iou[-1], global_step=epoch)
                l.pop(-1)
                l.append("mAcc")
                acc, acc_t = self.evaluator.Point_Accuracy_Class(epoch=epoch)
                self.writer.add_scalar(tag="mAcc/validation", scalar_value=acc[-1], global_step=epoch)

            if iou[-1] > max_miou:
                max_miou = iou[-1]
                early_stop_cnt = 0
                tt.write(f"{Fore.BLUE}Saving checkpoint at epcoch {epoch}, maximum mIOU: {max_miou:>.5f}")
                torch.save(self.net.state_dict(), self.checkpoint_path)
            else:
                early_stop_cnt += 1

            # log info
            tt.write(f"Epoch[{Fore.GREEN}{epoch}{Style.RESET_ALL}|{n_epoch}], early_stop_cnt: {early_stop_cnt}")
            tt.write(acc_t.table)
            tt.write("\n")
            tt.write(iou_t.table)
            tt.write("\n"*3)

            if early_stop_cnt >= early_stop:
                tt.write(f"{Fore.YELLOW}Early stopped at epoch: {epoch}")
                break

    @torch.no_grad()
    def test(self):
        test_loader = data.DataLoader(self.test_ds, batch_size=64, shuffle=False, num_workers=2)
        for step, (x, y) in test_loader:
            self.net.zero_grad()
            x, y = x.to(dtype=self.dtype, device=self.available_device), y.to(dtype=torch.long,
                                                                                device=self.available_device)
            y_pred = self.net(x.unsqueeze(dim=1).permute(0, 1, 3, 2))
            loss = self.loss_func(y_pred, y)
            print(loss)


if __name__ == "__main__":
    # todo: 有时间的话改改preprocess, 修改的内容就是z不限制, x, y限制, 最后得到一样的数据, 还有降采样问题
    trainer = Network1DTrainer(PointNetSegmentation1D(in_features=6, predicted_cls=14)).train()
    # trainer = Network2DTrainer(PointNetSegmentation2D(in_features=6, predicted_cls=14)).train(n_epoch=1000)