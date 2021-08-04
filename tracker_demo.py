
import sys


from pyimagesearch.centroidtracker import CentroidTracker
from trt_yolo import DetectTensorRT
from tracker import IDetectionMetadata



import pandas as pd
import math 

import os
import cv2
import numpy as np
import time

from threading import Thread, enumerate
from queue import LifoQueue, Queue
import datetime 
ALLOWED_CLASSES = [0.]



class Detection(IDetectionMetadata):
	def __init__(self, darknet_det):
		#self._class = darknet_det[0].decode('ascii')
		self._class = darknet_det[0]

		x, y, width, height = darknet_det[2]
		self.bbox = [x,y,width-x,height-y]
		#self.bbox = [int(round(x - (width / 2))), int(round(y - (height / 2))), int(width), int(height)]
		self._confidence = float(darknet_det[1])
	
	def tlbr_a(self):
		return self.bbox

	def confidence(self):
		return self._confidence
	
	def class_(self):
		return self._class


class ArgsHelper:

    __slots__ = 'image', 'video', 'video_looping', 'rtsp', 'rtsp_latency', 'usb', 'onboard', 'copy_frame', 'do_resize', 'width', 'height', 'category_num', 'model'
    
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)





if __name__ == "__main__":
    

    args = ArgsHelper(image=None, video=None, video_looping=False,rtsp=None, rtsp_latency=200, usb=0, onboard=None, copy_frame=False, do_resize=False, width=416, height=416, category_num=1, model='yolov4-tiny-head-416')

    trt = DetectTensorRT(args)
    trt.load_tensorRT()
    ct = CentroidTracker(maxDisappeared=10,maxDistance=60)


    cap = cv2.VideoCapture('demo.avi')

    count = 0
    fskip=0
    trakeable = {}

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
                
        frame = cv2.resize(frame,(416,416)) 

        boxes, confs, clss =  trt.process_img(frame)
        
        img = trt.vis.draw_bboxes(frame, boxes, confs, clss)

        del_list = []
        objects = ct.update(boxes)
        try:
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)

                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except:
            continue

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    cap.release()
    cv2.destroyAllWindows()
