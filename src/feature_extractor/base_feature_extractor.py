# coding=utf-8
class BaseFeatureExtractor(object):
    # 输入图片地址，feature数组
    def extact(self, image):
        raise RuntimeError("Method has not been implemented")
