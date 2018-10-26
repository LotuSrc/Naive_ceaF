# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 17:30
# @Author  : Ruichen Shao
# @File    : test.py

# 测试数据增强及自动调参模块
from src.feature_extractor.face_recog_extractor import FaceRecogExtractor
from global_config import ROOT_PATH
import src.offline_tools.hyper_opt.opt_util as hyper_opt
import src.offline_tools.data_augment.main as aug_wrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale, normalize
from hyperopt import hp
import numpy as np
import os

# 数据增强
# for i in range(0,3):
#     aug_wrapper.augment(ROOT_PATH + '/resources/face_image/' + str(i), 20)

# 数据准备
extractor = FaceRecogExtractor()
print('Start loading data.')
if not os.path.isfile(ROOT_PATH + '/resources/face_image/data.npy'):
    X = []
    y = []
    for i in range(0,3):
        imgdir = ROOT_PATH + '/resources/face_image/' + str(i) + '/output'
        flist = os.listdir(imgdir)
        for j in range(0, len(flist)):
            path = os.path.join(imgdir, flist[j])
            if os.path.isfile(path):
                feature = np.asarray(extractor.extact(path), dtype=float)
                if (feature.shape != (1, 128)):
                    print(path)
                    continue
                X.append(np.squeeze(feature))
                y.append(i)

    X = np.array(X)
    y = np.array(y)
    np.save(ROOT_PATH + '/resources/face_image/data.npy', X)
    np.save(ROOT_PATH + '/resources/face_image/label.npy', y)

else:
    X = np.load(ROOT_PATH + '/resources/face_image/data.npy')
    y = np.load(ROOT_PATH + '/resources/face_image/label.npy')

print('Finish loading data.')

print(X.shape)

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

hyper_opt.search(func=func,
                 space=space,
                 algo="tpe",  # 策略：tpe:贝叶斯，　rand:随机，　annel:退火，　mix: 前三种混合
                 max_iter=50
                 )
