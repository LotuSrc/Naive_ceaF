# coding=utf-8
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale, normalize
from hyperopt import hp
import opt_util as hyper_opt

# 数据准备
iris = datasets.load_iris()
X = iris.data
y = iris.target


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
