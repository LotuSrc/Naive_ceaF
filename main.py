# coding=utf-8

import sys
# 添加第三方库路径
sys.path.append('/home/chuangke6/tmp/download/lib-src/openface')
sys.path.append('/home/chuangke6/tmp/download/lib-src/pyseeta')

file_path = "/home/chuangke6/app2/Naive_ceaF/src/preprocess/face_recog_prep.py"

with open(file_path, 'r', encoding='UTF-8') as f:
    code = f.read()
exec(code)
