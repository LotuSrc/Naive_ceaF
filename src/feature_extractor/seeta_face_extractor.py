# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 11:06
# @Author  : Ruichen Shao
# @File    : seeta_face_extractor.py

import cv2
from pyseeta import Detector, Aligner, Identifier

from src.feature_extractor.base_feature_extractor import BaseFeatureExtractor


class SeetaFaceExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.detector = Detector()
        self.detector.set_min_face_size(30)
        self.aligner = Aligner()
        self.identifier = Identifier()

    def extact(self, image_path, is_one_face=False):
        image = cv2.imread(image_path)
        if image is None:
            print('Unable to load image: {}'.format(image_path))
            return None
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if is_one_face:
            bb1 = self.detector.detect(image_gray)
            bbs = [bb1]
        else:
            bbs = self.detector.detect(image_gray)

        if len(bbs) > 1:
            print('More than two faces in {}'.format(image_path))
            return None
        if len(bbs) == 0 or (is_one_face and bb1 is None):
            return None

        reps = []
        for bb in bbs:
            aligned_face = self.aligner.align(image_gray, bb)
            if aligned_face is None:
                return None
            rep = self.identifier.extract_feature_with_crop(image, aligned_face)
            return rep
            reps.append(rep)

        # 因为已经明确一张图只有一个人
        if is_one_face:
            return reps[0]
        else:
            return reps


if __name__ == "__main__":
    extractor = SeetaFaceExtractor()
    for i in range(0, 1000):
        feature = extractor.extact(
            "/home/chuangke6/app/Naive_ceaF/resources/face_image/0/huangbo1.jpg")
        print(i)
    print(feature)
