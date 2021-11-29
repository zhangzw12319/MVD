"""
Different from data_loader.py, some preprocessing or specific
operations on SYSU * RegDB datasets.
"""
from __future__ import print_function, absolute_import
import os
import random
import numpy as np


def process_query_sysu(data_path, mode='all'):
    """
    preprocess sysu query
    """
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_ir = []

    with open(file_path, 'r', encoding="utf-8") as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = [f"{x}:0>4d"for x in ids]

    for id_ in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id_)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if i[0] != '.'])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode='all', random_seed=0):
    """
    preprocess sysu gallery
    """
    random.seed(random_seed)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r', encoding="utf-8") as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = [f"{x:04d}" for x in ids]

    for id_ in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id_)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if i[0] != '.'])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)

def process_test_regdb(img_dir, trial=1, modal='visible'):
    """
    preprocess regdb test
    """
    if modal == 'visible':
        input_data_path = os.path.join(img_dir, f'idx/test_visible_{trial}' + '.txt')
    elif modal == 'thermal':
        input_data_path = os.path.join(img_dir, f'idx/test_thermal_{trial}' + '.txt')

    with open(input_data_path, encoding="utf-8"):
        with open(input_data_path, 'rt', encoding="utf-8") as path_str:
            data_file_list = path_str.read().splitlines()
            # Get full list of image and labels
            file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)
