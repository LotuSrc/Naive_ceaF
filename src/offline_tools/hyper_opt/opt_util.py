# coding=utf-8
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe, rand, mix, anneal
import pprint

# 优化策略
algo_dict = {
    "tpe": tpe.suggest,
    "rand": rand.suggest,
    "anneal": anneal.suggest,
    "mix": mix.suggest
}
anneal.suggest


def search(func, space, algo="tpe", max_iter=50):
    trials = Trials()
    best = fmin(func,  # 待最小化函数
                space=space,  # 参数所搜索空间
                algo=algo_dict[algo],  # 算法选择，这里选择了TPE，也可以用rand.suggest等
                max_evals=max_iter,  # 　迭代次数
                trials=trials,  # 可以用trials数组记录中间结果
                )
    # best是loss最小的参数组合
    # 对于离散值，如criterion，会返回选择的元素索引
    loss = trials.best_trial['result']['loss']
    pprint.pprint(best)
    print("------------------------------------")
    print("loss: %.3f" % loss)
