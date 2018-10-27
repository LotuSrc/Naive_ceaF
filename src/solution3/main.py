# coding=utf-8
import os

from hyperopt import hp
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale, normalize

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

extractor = SeetaFaceExtractor()


def name2id(name):
    global labels
    return labels.index(name)


X = []
y = []
X_test = []
y_test = []

for label in labels:
    for img in os.listdir('/home/chuangke6/app2/Naive_ceaF/resources/important/%s/output' % label):
        file_path = '/home/chuangke6/app2/Naive_ceaF/resources/important/%s/output/%s' % (label, img)
        try:
            feature = extractor.extact(image_path=file_path)
            if feature is None:
                print(file_path)
                os.remove(file_path)
                continue
            X.append(feature[0])
            y.append(name2id(label))
        except:
            print(file_path)
            os.remove(file_path)
    feature = extractor.extact("/home/chuangke6/app2/Naive_ceaF/resources/important/%s/%s.JPG" % (label, label))
    X_test.append(feature[0])
    y_test.append(name2id(label))


# 待优化目标函数
def func(params):
    global X, y
    X_ = X[:]
    # 这里可以自定义一些操作
    if params['normalize']:
        X_ = normalize(X_)
    if params['scale']:
        X_ = scale(X_)

    del params['normalize']
    del params['scale']

    # 模型
    clf = RandomForestClassifier(**params)
    # 交叉验证
    acc = cross_val_score(clf, X, y, cv=10).mean()
    return -acc


# 搜索空间
# 如果是连续值，用hp.uniform(x_min, x_max)的形式
space = {
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 5)),
    'n_estimators': hp.choice('n_estimators', range(1, 20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [True, False]),
    'normalize': hp.choice('normalize', [True, False])
}

# hyper_opt.search(func=func,
#                  space=space,
#                  algo="tpe",  # 策略：tpe:贝叶斯，　rand:随机，　annel:退火，　mix: 前三种混合
#                  max_iter=50
#                  )
print('R-F')
clf = RandomForestClassifier(
    max_depth=7,
    max_features=3,
    n_estimators=18,
)
acc = cross_val_score(clf, X, y, cv=10).mean()
print("train acc", acc)

clf.fit(X, y)
print("test score", clf.score(X_test, y_test))
