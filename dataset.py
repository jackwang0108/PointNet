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
from helper import DatasetPaths, visualize_xyz_rgb




class S3DIS(data.Dataset):
    def __init__(self):
        super(S3DIS, self).__init__()
        pass




if __name__ == "__main__":
    pc = np.loadtxt(DatasetPaths.S3DIS.s3dis_original_xyzrgb_data["Area_1"]["office_1"]["data"], dtype=np.float64)
    xyz, rgb = np.hsplit(pc, indices_or_sections=2)
    rgb /= 255

