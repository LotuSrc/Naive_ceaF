# coding=utf-8

from src.detection.base_detector import BaseDetetctor
import face_recognition

class FaceRegDetector(BaseDetetctor):
    def detect(self, image_path):
        pass


if __name__ == "__main__":
    detector = FaceRegDetector()
    detector.detect('resources/face_image/1.jpg')
