# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 17:47
# @Author  : Ruichen Shao
# @File    : frame_capture.py

import os

import cv2


class FrameCapturer(object):
    def capture(self, video_path, save_path):
        if not os.path.exists(save_path):
            print('creating save directory.')
            os.mkdir(save_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        index = 1
        success = True
        while success:
            success, frame = cap.read()
            if frame_count % 30 == 0:
                cv2.imwrite(os.path.join(save_path, str(index) + '.jpg'), frame)
                index += 1
            frame_count += 1
        if index == 1:
            print(video_path)
        # print('finish processing {}'.format(video_path))


if __name__ == '__main__':
    videos = os.listdir('/home/chuangke6/chuangke/diertishiping')
    cap = FrameCapturer()
    total = len(videos)

    for i, video in enumerate(videos):
        src = '/home/chuangke6/chuangke/diertishiping/' + video
        dest = '/home/chuangke6/video_image/' + video
        cap.capture(src, dest)
        # if i % 500 == 0:
        #     print(i, ' / ', total)
