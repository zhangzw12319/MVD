"""
Note: this file is a demo version of pre_process_sysu.py,
to prepare a demo dataset(save as .npy file) with
a small number of identities for debugging neural network.
"""
import os
import numpy as np
from PIL import Image


# todo_change your own path
data_path = "./data/SYSU-MM01"

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# load id info
file_path_train = os.path.join(data_path, 'exp/train_id_demo.txt')

with open(file_path_train, 'r', encoding="utf-8") as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["{x:04d}" for x in ids]

files_rgb = []
files_ir = []
for id_ in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, id_)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id_)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel
pid_container = set()
for img_path in files_ir:
    # print(img_path)
    # print(img_path[-13:-9])
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image_):
    """
    reading imgs
    """
    train_img_ = []
    train_label_ = []
    for img_path_ in train_image_:
        # img
        img = Image.open(img_path_)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img_.append(pix_array)

        # label
        pid_ = int(img_path_[-13:-9])
        pid_ = pid2label[pid_]
        train_label_.append(pid_)
    return np.array(train_img_), np.array(train_label_)

# rgb imges
train_img, train_label = read_imgs(files_rgb)
np.save(os.path.join(data_path, 'demo_train_rgb_resized_img.npy'), train_img)
np.save(os.path.join(data_path, 'demo_train_rgb_resized_label.npy'), train_label)

# ir imges
train_img, train_label = read_imgs(files_ir)
np.save(os.path.join(data_path, 'demo_train_ir_resized_img.npy'), train_img)
np.save(os.path.join(data_path, 'demo_train_ir_resized_label.npy'), train_label)
