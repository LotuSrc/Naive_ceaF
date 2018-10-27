# import os
# import random
#
# imgdir = '/home/chuangke6/chuangke/diyi/no2'
# files = os.listdir(imgdir)
#
# sub_dir = 'no2'
#
#
# random.shuffle(files)
# length = len(files)
# step = int(length / 5)
# for i in range(6):
#     file_name = "/home/chuangke6/partition/%s/%i.idx" % (sub_dir, i)
#     f = open(file_name, "w")
#     f.write(('\n'.join(files[i * step: (i + 1) * step])))
#     print(len(files[i * step: (i + 1) * step]))
#     print(i)


import json

import numpy as np

english = json.load(open('/home/chuangke6/face_lib/english.json'))

# all_dict = json.load(open('/home/chuangke6/face_lib/all_iter_3200.json'))
all_dict = english
# for key in english:
#     all_dict[key] = english[key]

face_rep = json.load(open('/home/chuangke6/face_lib/no2_final_5.json'))

order = list(face_rep.keys())

counter = 0
a = []
for img in order:
    counter += 1
    if counter % 2000 == 0:
        print(counter)
    a.append(face_rep[img])

a = np.array(a)

order2 = list(all_dict.keys())

b = np.array([0] * 128)
for img in order2:
    b = np.row_stack((b, all_dict[img]))

b = b[1:, :]


print(a.shape)
print(b.shape)
out = np.dot(a, b.T)

txt = ''

for img_id, all_id in enumerate(np.argmax(out, 1)):
    txt += order[img_id] + '\t' + order2[all_id] + '\n'
print(txt)
f = open('no2_5.txt', 'w')
f.write(txt)
f.close()
