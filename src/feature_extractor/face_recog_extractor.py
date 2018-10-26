# coding=utf-8
import face_recognition
from src.feature_extractor.base_feature_extractor import BaseFeatureExtractor


class FaceRecogExtractor(BaseFeatureExtractor):
    def extact(self, image_path, model='hog'):
        image = face_recognition.load_image_file(image_path)
        if image is None:
            print('Unable to load image: {}'.format(image_path))
            return None
        # TODO model到时候可以改成cnn
        face_locations = face_recognition.face_locations(image, model=model)
        # num_jitters越大越准确，但是会消耗更多
        features = face_recognition.face_encodings(
            image, face_locations, num_jitters=1)
        # 因为已经明确一张图只有一个人
        if len(features) > 1:
            print("multi-face", image_path)
            return None
        return list(features.reshape((1, -1)))


if __name__ == "__main__":
    extractor = FaceRecogExtractor()
    feature = extractor.extact(
        "/home/fanghao/Desktop/workstation/Py/Naive_ceaF/resources/face_image/5.jpg")
    print(feature)
