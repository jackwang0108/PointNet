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


from terminaltables import AsciiTable

data = []
data.append(['Row one column one', 'Row one column two'])
data.append(['Row two column one', 'Row two column two'])
data.append(['Row three column one', 'Row three column two'])

table = AsciiTable(data)

# print(table.table)

import numpy as np
import pandas as pd

a = pd.DataFrame(np.ones(shape=(3, 5)), columns=["PPAcc", "IOU", "ROU", "AP", "mAP"], index=["train", "val", "test"])

d = a.to_numpy().tolist()
c = a.columns.to_list()
i = a.index.to_list()

c.insert(0, "evaluator")

for ii, dd in zip(i, d):
    dd.insert(0, ii)

data = [c, *d]
data = AsciiTable(data)

epoch = 100
data.title = f"Epoch[{epoch:>5d}] Evaluations"

print(data.table)
