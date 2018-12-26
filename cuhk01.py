import os
import  shutil
import glob
import numpy as np

"""
   Image name format: 0001001.png, where first four digits represent identity
   and last four digits represent cameras. Camera 1&2 are considered the same
   view (i.e 1) and camera 3&4  are considered the same view (i.e 2).
"""

campus_dir = '/home/paul/datasets/cuhk01/campus'
img_paths = sorted(glob.glob(os.path.join(campus_dir, '*.png')))
img_list = []
pid_container = set()
for img_path in img_paths:
    img_name = os.path.basename(img_path)
    pid = int(img_name[:4]) - 1
    camid = int(img_name[4:7])
    img_list.append((img_path, pid, camid))
    pid_container.add(pid)

num_pids = len(pid_container)
num_train_pids = num_pids // 2

splits = []

order = np.arange(num_pids)
np.random.shuffle(order)
train_idxs = order[:num_train_pids]
train_idxs = np.sort(train_idxs)
idx2label = {idx: label for label, idx in enumerate(train_idxs)}

train, test_a, test_b = [], [], []

for img_path, pid, camid in img_list:
    if pid in train_idxs:
        train.append((img_path, idx2label[pid], camid))
    else:
        if camid == 1 or camid == 3:
            test_a.append((img_path, pid, camid))
        else:
            test_b.append((img_path, pid, camid))

train_path = '/home/paul/datasets/cuhk01/bounding_box_train'
gallery_path = '/home/paul/datasets/cuhk01/gallery'
test_path = '/home/paul/datasets/cuhk01/bounding_box_test'
for subset, folder in zip((train, test_a, test_b), (train_path, gallery_path, test_path)):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for imgpath, pid, camid in subset:
        if camid == 1 or camid == 2:
            camid = "c1_s{}".format(camid)
        else:
            camid = "c2_s{}".format(camid // 2)
        dst = os.path.join(folder, "{:04d}_{}.png".format(pid, camid))
        shutil.copy(imgpath, dst)
