# coding=utf-8
import face_recognition
from src.feature_extractor.base_feature_extractor import BaseFeatureExtractor


class FaceRecogExtractor(BaseFeatureExtractor):
    def extact(self, image_path, is_one_face=False):
        image = face_recognition.load_image_file(image_path)
        # TODO model到时候可以改成cnn
        face_locations = face_recognition.face_locations(image, model='hog')
        # num_jitters越大越准确，但是会消耗更多
        features = face_recognition.face_encodings(
            image, face_locations, num_jitters=1)
        # 因为已经明确一张图只有一个人
        if is_one_face:
            return features[0]
        else:
            return features


if __name__ == "__main__":
    extractor = FaceRecogExtractor()
    feature = extractor.extact(
        "/home/fanghao/Desktop/workstation/Py/Naive_ceaF/resources/face_image/5.jpg")
    print(feature)
