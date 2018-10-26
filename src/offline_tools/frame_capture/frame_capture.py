# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 17:47
# @Author  : Ruichen Shao
# @File    : frame_capture.py

import cv2
import os

class FrameCapturer(object):
    def capture(self, video_path, save_path):
        if not os.path.exists(save_path):
            print('creating save directory.')
            os.mkdir(save_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = 1
        index = 1
        success = True
        while success:
            success, frame = cap.read()
            if frame_count % 20 == 0:
                cv2.imwrite(os.path.join(save_path, str(index) + '.jpg'), frame)
                index += 1
            frame_count += 1
        print('finish processing {}'.format(video_path))

if __name__ == '__main__':
    cap = FrameCapturer()
    cap.capture('../../../resources/video/planet.mp4', '../../../resources/video_capture')