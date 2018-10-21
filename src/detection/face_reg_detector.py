# coding=utf-8

import face_recognition

from src.detection.base_detector import BaseDetetctor


class FaceRegDetector(BaseDetetctor):
    def detect(self, image_path):
        image = face_recognition.load_image_file(image_path)
        # 这里的model可以设置成cnn，会有更高的准确率，但需要GPU环境
        face_locations = face_recognition.face_locations(image, model='cnn')
        face_locations = [(left, top, right - left, bottom - top) for (top, right, bottom, left) in face_locations]
        return face_locations
