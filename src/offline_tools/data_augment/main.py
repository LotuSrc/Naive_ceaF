# coding=utf-8

import Augmentor
from global_config import ROOT_PATH

# 同一个imgdir下是同一类型的图片
# 通过一些变换，生成图片
def augment(imgdir, sample_n):
    p = Augmentor.Pipeline(imgdir)

    # 加入变换操作和概率
    # 旋转 & 放大 & 镜像
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)

    # 生成数据在 imgdir/output/ 下
    p.sample(sample_n)

# 离线工具的测试可以简单点
if __name__ == "__main__":
    augment(ROOT_PATH + '/resources/face_image', 50)
