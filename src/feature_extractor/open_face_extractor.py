# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 22:18
# @Author  : Ruichen Shao
# @File    : open_face_extractor.py

from src.feature_extractor.base_feature_extractor import BaseFeatureExtractor
import openface
import cv2
import os


class OpenFaceExtractor(BaseFeatureExtractor):
    def __init__(self, model='nn4.small2.v1.t7'):
        self.align = openface.AlignDlib(
            '/home/chuangke6/tmp/download/lib-src/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
        self.net = openface.TorchNeuralNet(
            os.path.join('/home/chuangke6/tmp/download/lib-src/openface/models/openface', model))

    def extact(self, image_path, is_one_face=False):
        image = cv2.imread(image_path)
        if image is None:
            raise Exception('Unable to load image: {}'.format(image_path))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if is_one_face:
            bb1 = self.align.getLargestFaceBoundingBox(image)
            bbs = [bb1]
        else:
            bbs = self.align.getAllFaceBoundingBoxes(image)
        if len(bbs) == 0 or (is_one_face and bb1 is None):
            raise Exception('Unable to find a face: {}'.format(image_path))

        reps = []
        for bb in bbs:
            aligned_face = self.align.align(imgDim=96, rgbImg=image, bb=bb,
                                            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                raise Exception('Unable to align image: {}'.format(image_path))
            rep = self.net.forward(aligned_face)
            return rep
            reps.append(rep)

        # 因为已经明确一张图只有一个人
        if is_one_face:
            return reps[0]
        else:
            return reps


if __name__ == "__main__":
    extractor = OpenFaceExtractor(model='nn4.v1.t7')
    feature = extractor.extact(
        "/home/chuangke6/app/Naive_ceaF/resources/face_image/0/huangbo1.jpg")
    print(feature)
