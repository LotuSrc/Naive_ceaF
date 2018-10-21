from global_config import ROOT_PATH
from src.detection.face_reg_detector import FaceRegDetector
from src.detection.util import show_detected_face

image_path = ROOT_PATH + '/resources/face_image/6.jpg'
detector = FaceRegDetector()
face_locations = detector.detect(image_path)
print('Face reg detect %d face' % len(face_locations))
show_detected_face(image_path, face_locations)
