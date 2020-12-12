# -*- coding:utf8 -*-
# @TIME     : 2020/12/8 10:39
# @Author   : SuHao
# @File     : random_cut.py

import cv2
import os
import numpy as np

def RandomCrop(dataset_path, save_path, size, num):
    ''' 将数据集中的每张图片随机裁剪为num个子图
    :param dataset_path: 数据集路径
    :param save_path: 保存路径
    :param size: 子图大小(a, b)分别对应子图的宽和高
    :param num: m每张图裁剪的子图的数量
    :return:
    '''
    filename = os.listdir(dataset_path)
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(filename)):
        filepath = dataset_path +'\\' + filename[i]
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        img = cv2.resize(img, (256, 256))
        x = np.random.randint(0, img.shape[0]-size[1], (num, 1))
        # 此处需要注意矩阵的行数与图片的宽和高的对应方式
        y = np.random.randint(0, img.shape[1]-size[0], (num, 1))
        strart_point = np.concatenate([x, y], axis=1)
        if len(img.shape) == 2:         # 1 channels
            for j in range(num):
                sub_imgs = img[strart_point[j, 0]:strart_point[j, 0]+size[1],
                           strart_point[j, 1]:strart_point[j, 1] + size[0]]
                name = os.path.splitext(filename[i])[0] + "_" + str(j) + ".jpg"
                cv2.imencode('.jpg', sub_imgs)[1].tofile(save_path + '\\' + name)
        elif len(img.shape) == 3:         # 3 channels
            for j in range(num):
                sub_imgs = img[strart_point[j, 0]:strart_point[j, 0]+size[1],
                           strart_point[j, 1]:strart_point[j, 1] + size[0], :]
                name = os.path.splitext(filename[i])[0] + "_" + str(j) + ".jpg"
                cv2.imencode('.jpg', sub_imgs)[1].tofile(save_path + '\\' + name)
                # 将文件保存到中文路径


dataset_path = r"D:\硕士\mvtec - 副本\grid\train\good"
save_path = r"D:\硕士\mvtec\grid\train\good"

# dataset_path = r"F:\硕士\图像瑕疵\mvtec\mvtec\grid\grid\train\good"
# save_path = r"F:\硕士\图像瑕疵\mvtec\mvtec\grid\grid\crop\crop"
RandomCrop(dataset_path, save_path, (128, 128), 50)