# coding=utf-8

from src.detection.face_reg_detector import FaceRegDetector

detector = FaceRegDetector()


# wd = os.getcwd()
# index = wd.find("src")
# os.chdir(wd[:index - 1])
# print(os.getcwd())
def test():
    face_locations = detector.detect('resources/face_image/1.jpg')
    print(face_locations)


# wd = os.chdir

test()
