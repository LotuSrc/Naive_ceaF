# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 23:04
# @Author  : Ruichen Shao
# @File    : seeta_face_prep.py

from src.feature_extractor.seeta_face_extractor import SeetaFaceExtractor
import os
import numpy as np

if __name__ == '__main__':
    imgdir = '/home/chuangke6/chuangke/diyi/no1'
    extractor = SeetaFaceExtractor()
    files = os.listdir(imgdir)
    length = len(files)
    print('There are {} images in total'.format(length))
    features = []
    count = 0
    index = 1
    for file in files:
        image_path = os.path.join(imgdir, file)
        feature = extractor.extact(image_path)
        if feature is None:
            print('{} extract feature failed.'.format(file))
            continue
        features.append(feature)
        count += 1
        if count % 10000 == 0:
            features_np = np.array(features)
            features_np = np.squeeze(features_np)
            np.save('/home/chuangke6/app/Naive_ceaF/resources/seeta_face_no1_' + str(index) + '.npy', features_np)
            print('finish {} images of {} images.'.format(count, length))
            features = []