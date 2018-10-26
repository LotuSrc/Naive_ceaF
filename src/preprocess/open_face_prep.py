# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 23:04
# @Author  : Ruichen Shao
# @File    : seeta_face_prep.py

from src.feature_extractor.seeta_face_extractor import SeetaFaceExtractor
import os
import numpy as np
import json

if __name__ == '__main__':
    root = '/home/chuangke6/chuangke/diyi'
    imgdir = ['no1', 'no2']
    extractor = SeetaFaceExtractor()
    errors = []
    for t in imgdir:
        tt = os.path.join(root, t)
        files = os.listdir(tt)
        length = len(files)
        print('There are {} images in {} in total'.format(t, length))
        features = {}
        count = 0
        for file in files:
            image_path = os.path.join(tt, file)
            feature = extractor.extact(image_path)
            if feature is None:
                print('{} extract feature failed.'.format(file))
                errors.append(file)
                continue
            features[file] = feature
            count += 1
            if count % 10000 == 0:
                print('finish {} images of {} images.'.format(count, length))

        if len(features.keys()) > 0:
            json.dump(features, open('/home/chuangke6/features/seeta_face_' + t + '.json', 'w'))

    json.dump(errors, open('/home/chuangke6/erros/seeta_face_errors.json', 'w'))