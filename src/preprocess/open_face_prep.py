# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 23:04
# @Author  : Ruichen Shao
# @File    : seeta_face_prep.py

import json
import os

from src.feature_extractor.open_face_extractor import OpenFaceExtractor

sub_dir = 'no2'
imgdir = '/home/chuangke6/chuangke/diyi/%s' % sub_dir


def func(process_id, files):
    print(process_id, len(files))

    extractor = OpenFaceExtractor()
    return
    erros = []
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
            print('{} finish {} images of {} images.'.format(process_id, count, length))
            print("%d erros: " % process_id, len(erros))

    print("finish %s and %d images" % (sub_dir, count))
    json.dump(features, open("/home/chuangke6/features/openface_%s_%d.json" % (sub_dir, process_id), "w"))

    json.dump(erros, open("/home/chuangke6/erros/openface_erros_%d.json" % process_id, "w"))


if __name__ == "__main__":
    imgdir = '/home/chuangke6/chuangke/diyi/%s' % sub_dir
    files = os.listdir(imgdir)
    import random

    random.shuffle(files)
    length = len(files)
    step = int(length / 3)
    for i in range(4):
        file_name = "/home/chuangke6/partition/%s/%i.idx" % (sub_dir, i)
        f = open(file_name, "w")
        f.write(('\n'.join(files[i * step: (i + 1) * step])))
        print(i)
