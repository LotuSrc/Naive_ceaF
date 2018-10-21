# coding=utf-8
class BaseDetetctor(object):
    # 输入图片地址，返回bbox数组
    def detect(self, image):
        raise RuntimeError("Method has not been implemented")
