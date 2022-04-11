# import numpy as np


# class Node:
#     def __init__(self, key, value=-1):
#         self.left = None
#         self.right = None
#         self.key = key
#         self.value = value

#     def __repr__(self):
#         return f"Node({self.left}, {self.key}, {self.right})"


# def insert(root: Node, key, value=-1):
#     if root is None:
#         root = Node(key, value)
#     else:
#         if key < root.key:
#             root.left = insert(root.left, key, value)
#         elif key > root.key:
#             root.right = insert(root.right, key, value)
#         else:
#             pass
#     return root


# def inorder(root: Node):
#     if root is not None:
#         inorder(root.left)
#         print(root.key)
#         inorder(root.right)


# db_size = 10
# data = np.random.permutation(10).tolist()
# print(data)

# root = None
# for i, point in enumerate(data):
#     root = insert(root, point, i)

# inorder(root)


# from terminaltables import AsciiTable

# data = []
# data.append(['Row one column one', 'Row one column two'])
# data.append(['Row two column one', 'Row two column two'])
# data.append(['Row three column one', 'Row three column two'])

# table = AsciiTable(data)

# # print(table.table)

# import numpy as np
# import pandas as pd

# a = pd.DataFrame(np.ones(shape=(3, 5)), columns=["PPAcc", "IOU", "ROU", "AP", "mAP"], index=["train", "val", "test"])

# d = a.to_numpy().tolist()
# c = a.columns.to_list()
# i = a.index.to_list()

# c.insert(0, "evaluator")

# for ii, dd in zip(i, d):
#     dd.insert(0, ii)

# data = [c, *d]
# data = AsciiTable(data)

# epoch = 100
# data.title = f"Epoch[{epoch:>5d}] Evaluations"

# print(data.table)


# def bincount2d(arr, bins=None):
#     if bins is None:
#         bins = np.max(arr) + 1
#     count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
#     indexing = np.arange(len(arr))
#     for col in arr.T:
#         count[indexing, col] += 1
#     return count


# t = np.array([[1,2,3],[4,5,6],[3,2,2]], dtype=np.int64)
# print(bincount2d(t))

import torch
import numpy as np
from helper import load_hdf5, visualize_xyz_label, visualize_xyz_rgb, DatasetPaths
from network1D import PointNetSegmentation1D


with load_hdf5(DatasetPaths.S3DIS.s3dis_processed_h5_data["ply_data_all_0"]) as f:
    data, label = np.asarray(f["data"]), np.asarray(f["label"])
    room1_batch, label1 = data[:80], label[:80]
    xyz_norm, rgb, xyz_origin = np.hsplit(np.concatenate(room1_batch, axis=0), (3, 6))
    label2 = np.concatenate(label1, axis=0)
    visualize_xyz_label(xyz_origin, label=label2[:, np.newaxis])
    room1_batch, label = torch.from_numpy(room1_batch), torch.from_numpy(label1)
    result = []
    net = PointNetSegmentation1D(in_features=6, predicted_cls=14)
    net.load_state_dict(torch.load("./checkpoints/PointNetSegmentation1D/2022-04-10 02_53_40.508248-best.pt"))
    for patch in room1_batch:
        y_pred = net(patch[:, :6].unsqueeze(dim=0).permute(0, 2, 1))
        y_pred = y_pred.argmax(dim=1)
        result.append(y_pred.squeeze().numpy())
    
    pred_label = np.concatenate(result, axis=0)
    a = room1_batch[:, :, 6:]
    pc = np.concatenate(tuple(a),  axis=0)
    visualize_xyz_label(pc, pred_label)

    print("Done")