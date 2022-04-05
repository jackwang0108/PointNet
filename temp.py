import numpy as np


class Node:
    def __init__(self, key, value=-1):
        self.left = None
        self.right = None
        self.key = key
        self.value = value

    def __repr__(self):
        return f"Node({self.left}, {self.key}, {self.right})"


def insert(root: Node, key, value=-1):
    if root is None:
        root = Node(key, value)
    else:
        if key < root.key:
            root.left = insert(root.left, key, value)
        elif key > root.key:
            root.right = insert(root.right, key, value)
        else:
            pass
    return root


def inorder(root: Node):
    if root is not None:
        inorder(root.left)
        print(root.key)
        inorder(root.right)


db_size = 10
data = np.random.permutation(10).tolist()
print(data)

root = None
for i, point in enumerate(data):
    root = insert(root, point, i)

inorder(root)
