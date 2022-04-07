# Standary Library
from typing import *
from pathlib import Path
from numbers import Number

# Third Party Library
import h5py
import numpy as np
import open3d as o3d

labels = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board",
          "clutter", "stairs"]
num2label: Dict[int, str] = dict(enumerate(labels))
label2num: Dict[str, int] = {name: i for i, name in enumerate(labels)}
_colors = np.array([
    [192, 57, 43], [155, 89, 182], [41, 128, 185], [133, 193, 233], [118, 215, 196], [14, 102, 85],
    [244, 208, 63], [235, 152, 78], [93, 109, 126], [233, 30, 99], [52, 73, 94], [109, 76, 65], [218, 247, 166],
    [110, 44, 0]
]) / 255
num2color: Dict[int, np.ndarray] = dict(zip(range(_colors.shape[0]), _colors))


class DatasetPaths(object):
    """
    Paths of datasets
    Attributes:
        S3DIS: Paths of S3DIS dataset
    """
    base: Path = Path(__file__).resolve().parent
    dataset = base.joinpath("datasets")

    class S3DIS:
        """
        Paths of S3DIS dataset
        Attributes:
            s3dis_original_xyzrgb_data: Dict[str, Dict[str, Dict[str, Union[Path, List[Path]]]]], dictionary of original
                data. For example, s3dis_original_xyzrgb_data["Area_1"]["office_1"]["data"] is a pathlib.Path object,
                which points to S3DIS/original/Area_1/office_1/office_1.txt and s3dis_original_xyzrgb_data["Area_2"]["
                hallway_1" ][" annotations"] is a list of pathlib.Path. s3dis_original_xyzrgb_data["Area_2"]["hallway_1"
                ][" annotations"][0] points to S3DIS/original/Area_2/hallway_1/Annotations/bookcase_1.txt
            s3dis_processed_h5_data: Dict[str, Path], s3dis_processed_h5_data["ply_data_all_0"] points to S3DIS/preporce
                ssed/ply_data_all_0.
            s3dis_processed_npy_data: Dict[str, Path], s3dis_processed_npy_data["Area_1_all"] points to S3DIS/my_preproc
                essed/Area_1_all.npy
        """
        _base: Path = Path(__file__).resolve().parent.joinpath("datasets")
        # s3dis paths
        s3dis_base: Path = _base.joinpath("s3dis")

        # s3dis my processed path
        _s3dis_my_processed: Path = s3dis_base.joinpath("my_processed").resolve()
        s3dis_processed_npy_data: Dict[str, Path] = {
            _p.stem: _p for _p in _s3dis_my_processed.glob("*.npy")
        }


        # s3dis preprocessed path
        _s3dis_processed: Path = s3dis_base.joinpath("processed").resolve()
        # data: {file_name: file_path}
        s3dis_processed_h5_data: Dict[str, Path] = {
            _p.stem: _p for _p in _s3dis_processed.glob("*.h5")
        }

        # s3dis original path
        _s3dis_original: Path = s3dis_base.joinpath("original").resolve()
        _s3dis_original_areas: Dict[str, Path] = {
            _p.stem: _p for _p in _s3dis_original.iterdir() if _p.is_dir()
        }

        # s3dis_original_xyzrgb_data["Area_1"]["office_1"]["data"]
        s3dis_original_xyzrgb_data: Dict[str, Dict[str, Dict[str, Union[Path, List[Path]]]]] = {}
        for _area_name, _area_path in _s3dis_original_areas.items():
            _room_data = {}
            for _room_path in _area_path.iterdir():
                try:
                    _room_data[_room_path.name] = {
                        "data": next(_room_path.glob("*.txt")),
                        "annotations": list(_room_path.joinpath("Annotations").glob("*.txt"))
                    }
                except StopIteration:
                    pass
            s3dis_original_xyzrgb_data[_area_name] = _room_data


def voxelize(xyz: np.ndarray, voxel_grid_size: Number = 0.2) -> np.ndarray:
    """
    used to voxel downsample point cloud
    Args:
        xyz: point cloud, represented by a N*3 np.ndarray
        voxel_grid_size: size of each grid in the unit of meter
    Returns:
        per_point_indices: grid indices the point belongs to, represented by a np.ndarray of shape N
    Examples:
        >>> pc: np.ndarray = np.loadtxt(s3dis_area1_room1, dtype=np.float64).reshape(-1, 6)
        >>> xyz, rgb = np.hsplit(pc, indices_or_sections=2)
        >>> indices = voxelize(xyz)
    Notes:
        voxelize will build grids uniformly, but returned index is the grid index which the point belongs to.
        Since not all grids in the space have points, the distribution of per_point_indices is not an uniform
        distrubution (you can visualize with plt.hist(per_point_indices, bins=int(Ds[0]*Ds[1]*Ds[2]))
    """
    xyz_max, xyz_min = xyz.max(axis=0), xyz.min(axis=0)
    # Ds = [Dx, Dy, Dz]
    Ds = (xyz_max - xyz_min) / voxel_grid_size
    # find extra grid
    if not (where_extra_grid := (Ds - np.floor(Ds) == 0)).all():
        Ds = np.floor(Ds)
        Ds[~where_extra_grid] += 1
    # F = [1, Dx, Dx*Dy]
    F = np.array([1, Ds[0], Ds[0] * Ds[1]])
    # Hs = [hx, hy, hz] of shape N * 3
    Hs = np.floor((xyz - xyz_min) / voxel_grid_size)
    per_point_indices = Hs @ F.T
    return per_point_indices


def load_txt(point_path: Path, with_label: bool = True) -> np.ndarray:
    """
    load_txt is used to load point cloud from .txt file (i.e., original s3dis data format)
    Args:
        point_path: Path of point cloud to be loaded, should be either folder of rooms or all points.
        with_label: bool, if return with label
    Returns:
        room_points: point cloud of a room, represented by a N*6 (without label) or N*7 (with label) np.ndarray
    Examples:
        >>> from pathlib import Path
        >>> points = load_txt(point_path=Path("./datasets/s3dis/original/Area_1/office_1"))
        >>> # they are the same
        >>> points = load_txt(point_path=Path("./datasets/s3dis/original/Area_1/office_1/office_1.txt"))
        >>> print(points.shape)
        (884955, 7)
        >>> points = load_txt(point_path=Path("./datasets/s3dis/original/Area_1/office_1"))
        >>> print(points.sa)
        (884955, 6)
    Notes:
        label name to label number is show in helper.label2num
    """
    global label2num
    # load instances
    instance_folder = point_path.parent.joinpath("Annotations") if point_path.is_file() else point_path.joinpath(
        "Annotations")
    per_object_instance: Dict[str, List[np.ndarray]] = {}
    # load from each file
    for instance_file in instance_folder.glob("*.txt"):
        name = instance_file.stem.split("_")[0]
        # Some files are actually cannot be loaded due to error columes
        # for example, Area_5/hallway_6/Annotations/ceiling_1, row 180390.
        data = np.loadtxt(instance_file, dtype=str)

        # detect corrupted data
        # Some rows in the dataset are corrupted, for example, in Area_3/hallway_2/hallway_2.txt, row 5303 has
        # a control symbol which cannot be convert to float, so just drop those corrupted points.
        # this method is effective but not efficient, takes about 30% time to run next 6 lines
        temp = []
        for line in data:
            try:
                temp.append(line.reshape(1, -1).astype(np.float64, copy=False))
            except ValueError:
                continue

        # add to dict
        if (l := per_object_instance.get(name, None)) is not None:
            l.append(np.vstack(temp))
        else:
            per_object_instance[name] = [np.vstack(temp)]

    # concate each instance
    for name, instances in per_object_instance.items():
        points = np.vstack(instances)
        instance_idx = np.full(shape=(len(points), 1), fill_value=label2num[name])
        if with_label:
            points = np.hstack((points, instance_idx))
        per_object_instance[name] = points

    # concate all instance and shuffle
    room_points: np.ndarray = np.vstack(tuple(per_object_instance.values()))
    np.random.shuffle(room_points)
    return room_points

def load_hdf5(hdf5_path: Path):
    assert hdf5_path.exists() and hdf5_path.suffix == ".hdf5"
    return h5py.File(hdf5_path, mode="r")



def visualize_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray = None) -> None:
    """
    visualize the point cloud witt/without rgb color
    Args:
        xyz: point cloud with N point, represented by a N*3 np.ndarray.
             xyz[i, :] is the coordinate ([xi, yi, zi]) of the point pi.
        rgb: corresponding color of the given point cloud, represented by a N*3 np.ndarray.
             rgb[i, :] is the color ([ri, gi, bi]) of the point pi.
    Returns:
        None
    Examples:
        >>> pc: np.ndarray = load_txt(s3dis_area1_room1, with_label=False)
        >>> xyz, rgb = np.hsplit(pc, indices_or_sections=2)
        >>> visualize_xyz_rgb(xyz) # visualize pure point cloud with automatically color.
        >>> visualize_xyz_rgb(xyz, rgb) # visualize point cloud with given color.
    """
    assert xyz.shape[1] == 3 or rgb.shape[1] == 3
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None:
        if rgb.max() > 1:
            rgb = rgb / 255
        pointcloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.visualization.draw_geometries([pointcloud], width=800, height=600)


def visualize_xyz_label(xyz: np.ndarray, label: np.ndarray = None, lookup_table: np.ndarray = num2label):
    """
    visualize the point cloud witt/without label color
    Args:
        xyz: point cloud with N point, represented by a N*3 np.ndarray.
             xyz[i, :] is the coordinate ([xi, yi, zi]) of the point pi.
        label: corresponding label of the given point cloud, represented by a N*1 np.ndarray.
             rgb[i, 0] is the label (li) of the point pi.
    Returns:
        None
    Examples:
        >>> pc: np.ndarray = load_txt(s3dis_area1_room1, with_label=True)
        >>> xyz, rgb = np.hsplit(pc, indices_or_sections=[3, 6])
        >>> visualize_xyz_label(xyz) # visualize pure point cloud with automatically color.
        >>> visualize_xyz_label(xyz, rgb) # visualize point cloud with given label color.
    """
    assert xyz.shape[1] == 3 and xyz.shape[0] == label.shape[0]
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if label is not None:
        colors = np.vstack([num2color[i] for i in label[:, 0]])
        pointcloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.visualization.draw_geometries([pointcloud], width=800, height=600)


if __name__ == "__main__":
    # import pprint
    # pprint.pprint(p := s3dis.s3dis_original_xyzrgb_data["Area_1"]["office_1"]["data"])

    # pc: np.ndarray = np.loadtxt(DatasetPaths.S3DIS.s3dis_original_xyzrgb_data["Area_1"]["office_1"]["data"],
    #                             dtype=float).reshape(-1, 6)
    # xyz, rgb = np.hsplit(pc, indices_or_sections=2)
    # visualize(xyz=xyz, rgb=rgb)

    # voxelize
    # index = voxelize(xyz, voxel_grid_size=3)
    # print(index, index.max())
    # color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    # rgb[index == 0] = color[0, :]
    # rgb[index == 1] = color[1, :]
    # rgb[index == 2] = color[2, :]
    # rgb[index == 3] = color[3, :]
    # visualize(xyz, rgb)

    # index = voxelize(xyz, voxel_grid_size=1)
    # print(index.max())
    # visualize_xyz_rgb(xyz[index == 0], rgb[index == 0])

    # pc = load_txt(DatasetPaths.S3DIS.s3dis_original_xyzrgb_data["Area_1"]["office_1"]["data"])
    # print(pc.shape)
    # xyz, rgb, label = np.hsplit(pc[:, :], indices_or_sections=[3, 6])
    # # visualize_xyz_rgb(xyz, rgb)
    # visualize_xyz_label(xyz, label)

    # all_class = {}
    # a = DatasetPaths.S3DIS.s3dis_original_xyzrgb_data
    # for area in a.keys():
    #     for room in a[area].keys():
    #         p: Path = a[area][room]
    #         for i in p["annotations"]:
    #             name = i.stem.split("_")[0]
    #             if all_class.get(name, None) is not None:
    #                 all_class[name] += 1
    #             else:
    #                 all_class[name] = 1
    # print(all_class)

    # my_process = DatasetPaths.S3DIS.s3dis_processed_npy_data
    # import pprint
    # pprint.pprint(my_process)
    # pprint.pprint(DatasetPaths.S3DIS.s3dis_processed_h5_data)

    with load_hdf5(DatasetPaths.S3DIS.s3dis_processed_npy_data["Area_1_all"].parent.joinpath("Area_3.hdf5")) as f:
        print("Done")
