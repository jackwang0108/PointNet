# Standard Library
from typing import *
from pathlib import Path

# Third Party Library
import h5py
import tqdm
import numpy as np
from colorama import Fore, Style, init
from sklearn.preprocessing import MaxAbsScaler

init(autoreset=True)

# my Library
from helper import DatasetPaths, voxelize, visualize_xyz_rgb, load_txt


def centerization(xyz: np.ndarray):
    return xyz - xyz.mean(axis=0)


def normalization(xyz: np.ndarray):
    return MaxAbsScaler().fit_transform(xyz)


def preprocess_s3dis():
    s3dis = DatasetPaths.S3DIS()
    total_num = sum(len(i.keys()) for i in s3dis.s3dis_original_xyzrgb_data.values())
    tqdm_bar = tqdm.trange(total_num)
    # save folder
    my_processed = DatasetPaths.S3DIS.s3dis_base.joinpath("my_processed")
    my_processed.mkdir(parents=True, exist_ok=True)
    for area_name, area in s3dis.s3dis_original_xyzrgb_data.items():
        # file to save
        # file = h5py.File()
        area_data = {area_name:[]}
        for room_name, room in area.items():
            room_data = {"from": room_name, "data": []}
            # preprocess each room
            tqdm_bar.set_description(f"Preprocessing[{Fore.GREEN}{area_name}/{room_name}{Fore.RESET}]")
            # load data
            data = load_txt(room["data"])

            # split for fuether process
            xyz, rgb, label = np.hsplit(data, indices_or_sections=[3, 6])
            rgb /= 255
            # voxelization
            voxelization_index = voxelize(xyz=xyz)
            for voxel_grid_idx in np.unique(voxelization_index):
                # process each voxel grid
                idx = (voxelization_index == voxel_grid_idx)
                point, color, l = xyz[idx], rgb[idx], label[idx]
                point_centered = centerization(point)
                point_normalized = normalization(point_centered)
                room_data["data"].append(
                    {"grid_idx": voxel_grid_idx, "points": np.hstack((point_normalized, color)), "label": l})
            area_data[area_name].append(room_data)
            tqdm_bar.write(f"Finished {area_name}/{room_name}")
            # update
            tqdm_bar.update(1)
        tqdm_bar.write(f"Saving: {Fore.BLUE}{area_name}{Fore.RESET}")
        np.save(my_processed / f"{area_name}_all.npy", area_data)


if __name__ == "__main__":
    preprocess_s3dis()
