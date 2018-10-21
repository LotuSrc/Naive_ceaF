# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 14:34
# @Author  : Ruichen Shao
# @File    : detector_test.py

from src.detection.cv_detector import OpencvDector
import cv2
from src.detection.cv_config import Config

image_path = Config.ROOT + '/resources/face_image/1.jpg'
image = cv2.imread(image_path)

# OpencvDector 单元测试
opencv_detector = OpencvDector()
opencv_faces = opencv_detector.detect(image_path)

for (x, y, w, h) in opencv_faces:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('opencv_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

from src.detection.face_reg_detector import FaceRegDetector

detector = FaceRegDetector()


# wd = os.getcwd()
# index = wd.find("src")
# os.chdir(wd[:index - 1])
# print(os.getcwd())
# def test():
#     face_locations = detector.detect('resources/face_image/1.jpg')
#     print(face_locations)
#
#
# # wd = os.chdir
#
# test()

print("hi")
