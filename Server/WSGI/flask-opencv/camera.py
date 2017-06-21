# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:10:06 2017

@author: admin
"""

import cv2

video1 = cv2.VideoCapture(0)
class VideoCamera(object):
    
#    # 定义静态变量实例
#    __singleton = None
    
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = video1
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
#    @staticmethod
#    def get_instance():
#        if VideoCamera.__singleton is None:
#            VideoCamera.__singleton = VideoCamera()
#        return VideoCamera.__singleton

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        # 对于 python2.7 或者低版本的 numpy 请使用 jpeg.tostring()
        return jpeg.tobytes()