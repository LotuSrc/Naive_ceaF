# coding=utf-8

import Augmentor


# 同一个imgdir下是同一类型的图片
# 通过一些变换，生成图片
def augment(imgdir, sample_n):
    p = Augmentor.Pipeline(imgdir)

    # 加入变换操作和概率

    p.sample(sample_n)
    pass


# 离线工具的测试可以简单点
if __name__ == "__main__":
    pass
