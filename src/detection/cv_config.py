# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 14:43
# @Author  : Ruichen Shao
# @File    : cv_config.py

import get_root

class Config():
    ROOT = get_root.getRoot()
    CASCPATH = ROOT + "/resources/model/haarcascade_frontalface_default.xml"
