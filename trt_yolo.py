import numpy as np
import argparse
from threading import Thread
import cv2
import os
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
import time

WINDOW_NAME = 'TrtYOLODemo'


class DetectTensorRT():
    def __init__(self, args, parent=None):

        self.args = args
        self.trt_yolo = None
        self.conf_th = 0.3
        self.vis = None
        self.w = None
        self.h = None
    
    def load_tensorRT(self):
        if self.args.category_num <= 0:
            raise SystemExit('ERROR: bad category_num (%d)!' % self.args.category_num)
        if not os.path.isfile('yolo/%s.trt' % self.args.model):
            raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % self.args.model)

        cls_dict = get_cls_dict(self.args.category_num)
        yolo_dim = self.args.model.split('-')[-1]
        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            w, h = int(dim_split[0]), int(dim_split[1])
        else:
            h = w = int(yolo_dim)
        if h % 32 != 0 or w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        self.w = w
        self.h = h
        self.trt_yolo = TrtYOLO(self.args.model, (h, w), self.args.category_num,cuda_ctx=pycuda.autoinit.context)
        self.vis = BBoxVisualization(cls_dict)

    def process_img(self,img):
        boxes, confs, clss = self.trt_yolo.detect(img, self.conf_th)
        size = [self.w,self.h]
        # b_ = []
        # for box in boxes:
        #     b_.append(self.convert(size,box))
        detections = list(zip(clss,confs,boxes))
        return boxes, confs, clss
        #return detections

    def convert(self,size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

class ArgsHelper:

    __slots__ = 'image', 'video', 'video_looping', 'rtsp', 'rtsp_latency', 'usb', 'onboard', 'copy_frame', 'do_resize', 'width', 'height', 'category_num', 'model'
    
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
         

if __name__ == "__main__":
    args = ArgsHelper(image=None, video=None, video_looping=False,rtsp=None, rtsp_latency=200, usb=0, onboard=None, copy_frame=False, do_resize=False, width=640, height=480, category_num=80, model='yolov4-tiny-416')
    trt = DetectTensorRT(args)
    trt.load_tensorRT()

    cap = cv2.VideoCapture('demo.avi')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        boxes, confs, clss =  trt.process_img(frame)

        img = trt.vis.draw_bboxes(frame, boxes, confs, clss)
        
        detections = list(zip(clss,confs,boxes))


        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
