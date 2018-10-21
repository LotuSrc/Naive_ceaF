# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 14:34
# @Author  : Ruichen Shao
# @File    : cv_detector_test.py

import cv2

from global_config import ROOT_PATH
from src.detection.cv_detector import OpencvDector
from src.detection.util import show_detected_face

image_path = ROOT_PATH + '/resources/face_image/6.jpg'
image = cv2.imread(image_path)

# OpencvDector 单元测试
opencv_detector = OpencvDector()
opencv_faces = opencv_detector.detect(image_path)

show_detected_face(image_path, opencv_faces)
