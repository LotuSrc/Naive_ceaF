from global_config import ROOT_PATH
from src.detection.face_reg_detector import FaceRegDetector

image_path = ROOT_PATH + '/resources/face_image/1.jpg'
detector = FaceRegDetector()
face_locations = detector.detect(image_path)
print(face_locations)
