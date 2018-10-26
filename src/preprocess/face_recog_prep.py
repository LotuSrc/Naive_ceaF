# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 23:04
# @Author  : Ruichen Shao
# @File    : seeta_face_prep.py

import json
import os

from src.feature_extractor.face_recog_extractor import FaceRecogExtractor


erros = []

for sub_dir in ['no1', 'no2']:
    imgdir = '/home/chuangke6/chuangke/diyi/%s' %sub_dir
    extractor = FaceRecogExtractor()
    files = os.listdir(imgdir)
    length = len(files)
    print('There are {} images in total'.format(length))

    features = {}
    count = 0
    for file in files:
        image_path = os.path.join(imgdir, file)
        try:
            feature = extractor.extact(image_path)[0]
            if feature is None:
                # print("can't extract ", image_path)
                erros.append(image_path)
                continue
        except:
            # print("can't extract ", image_path)
            erros.append(image_path)
            continue
        features[file] = feature
        count += 1

        if count % 1000 == 0:
            print('finish {} images of {} images.'.format(count, length))
            print("erros: ", len(erros))

    print("finish %s and %d images" % (sub_dir, count))
    json.dump(features, open("/home/chuangke6/features/hog_%s.json" % sub_dir, "w"))


json.dump(erros, open("/home/chuangke6/erros/hog_erros.json", "w"))

