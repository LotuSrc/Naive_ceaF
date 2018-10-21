# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 14:32
# @Author  : Ruichen Shao
# @File    : cv_detector.py

from src.detection.base_detector import BaseDetetctor
from src.detection.cv_config import Config
import cv2

class OpencvDector(BaseDetetctor):
    def detect(self, image_path):
        casc_path = Config.CASCPATH

        # 创建 haar cascade
        face_cascade = cv2.CascadeClassifier(casc_path)

        # 读取图片
        print(image_path)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测脸部
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print('Opencv Detected ', len(faces), ' face')

        return faces
