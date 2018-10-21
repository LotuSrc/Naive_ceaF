# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 15:12
# @Author  : Ruichen Shao
# @File    : get_root.py

import os

def getRoot():
    root = os.path.dirname(os.path.abspath(__file__))
    return root