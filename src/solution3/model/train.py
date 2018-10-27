import os

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from src.feature_extractor.face_recog_extractor import FaceRecogExtractor
from src.feature_extractor.seeta_face_extractor import SeetaFaceExtractor

labels = [
    'Fuzhenyu',
    'Guokui',
    'Hihuibin',
    'Jiangjianxin',
    'Lichaolin',
    'Linquan',
    'Sunqingfeng',
    'Tanyayun',
    'Xiangzhongwei',
    'Xuehongyu',
    'Yangyixin',
    'Yingbin'
]


def name2id(name):
    global labels
    return labels.index(name)


seeta_extractor = SeetaFaceExtractor()
face_recognization = FaceRecogExtractor()

X = []
y = []
X_test = []
y_test = []

for label in labels:
    for img in os.listdir('/home/chuangke6/app2/Naive_ceaF/resources/important/%s/output' % label):
        file_path = '/home/chuangke6/app2/Naive_ceaF/resources/important/%s/output/%s' % (label, img)
        try:
            feature1 = seeta_extractor.extact(image_path=file_path)
            feature2 = face_recognization.extact(image_path=file_path)
            if feature1 is None or feature2 is None:
                print(file_path)
                os.remove(file_path)
                continue
            feature1.extend(feature2)
            X.append(feature1)
            y.append(name2id(label))
        except:
            print(file_path)
            os.remove(file_path)

    img_path = "/home/chuangke6/app2/Naive_ceaF/resources/important/%s/%s.JPG" % (label, label)
    feature1 = seeta_extractor.extact(image_path=img_path)
    feature2 = face_recognization.extact(image_path=img_path)
    if feature1 is None or feature2 is None:
        print(img_path, "exception")
        continue

    feature1.extend(feature2)
    X_test.append(feature1)
    y_test.append(name2id(label))

print("data set constructed")

clf = RandomForestClassifier()
acc = cross_val_score(clf, X, y, cv=10).mean()
print("train acc", acc)

clf.fit(X, y)
print("test score", clf.score(X_test, y_test))
