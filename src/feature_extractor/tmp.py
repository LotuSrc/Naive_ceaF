# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 12:19
# @Author  : Ruichen Shao
# @File    : tmp.py

import time

start = time.time()

import cv2
import os

import numpy as np

np.set_printoptions(precision=2)

import openface

modelDir = os.path.join("/home/chuangke6/tmp/download/lib-src/openface/models")
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

start = time.time()
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))

def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))

    start = time.time()
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))

    rep = net.forward(alignedFace)
    return rep

rep = getRep('/home/chuangke6/app/Naive_ceaF/resources/face_image/0/huangbo1.jpg')
print(rep)