# coding=utf-8

import sys
# 添加第三方库路径
sys.path.append('/home/chuangke6/tmp/download/lib-src/openface')
sys.path.append('/home/chuangke6/tmp/download/lib-src/pyseeta')

file_path = "src/feature_extractor/open_face_extractor.py"

with open(file_path, encoding='utf-8') as f:
    code = f.read()
exec(code)
