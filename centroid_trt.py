import sys

from pyimagesearch.centroidtracker import CentroidTracker
from trt_yolo import DetectTensorRT
#from tracker.backends.deepsort import DeepSortTracker
from tracker import IDetectionMetadata

import pandas as pd
from scipy.spatial import distance
import math 

import os
import cv2
import numpy as np
import time

from threading import Thread, enumerate
from queue import LifoQueue, Queue
import datetime 
import argparse
import sys
ALLOWED_CLASSES = [0.]

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


"""
    Clase linetrack, la cual lleva seguimiento de todas las ids de 
    las personas que han cruzado una linea, para el contexto
    de este problema, se manteiene seguimiento de 
    la direccion de las personas con tal de llevar el conteo
    de gente que sube y que baja.   
"""
class line_track():


	def __init__(self, pline1, pline2):
		self.pline1 = pline1
		self.pline2 = pline2
		self.countup = 0
		self.countdown = 0
		self.counted = {}
		self.frame_id = {}
		self.counters = {}
		columns = ['time_video','n_frame','id_crossed','total_count_up','total_count_down']
		self.df = pd.DataFrame(columns = columns)
	def resize(self, wratio, hratio):
		pline1 = np.array((int(self.pline1[0] * wratio), int(self.pline1[1] * hratio)))
		pline2 = np.array((int(self.pline2[0] * wratio), int(self.pline2[1] * hratio)))
		return line_track(pline1, pline2)

	def has_cossed(self,p0, p1):
		return intersect(p0, p1, self.pline1, self.pline2)
	def count_up(self):
		self.countup+=1
	def count_down(self):
		self.countdown+=1

	def _count(self, class_, in_, out):
		if class_ in self.counters:
			counter = self.counters[class_]
		else:
			counter = ClassCounter(class_)
			self.counters[class_] = counter
		counter.in_ += in_
		counter.out += out
	
	def count_in(self, class_):
		self._count(class_, 1, 0)

	def count_out(self, class_):
		self._count(class_, 0, 1)



"""
Clase Detection para mantener orden dentro de la metadata
que se envia al sistema centralizado de conteo.
"""
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

"""
clase para mantener  ordenados los agurmentos para TensorRT
"""
class ArgsHelper:

    __slots__ = 'image', 'video', 'video_looping', 'rtsp', 'rtsp_latency', 'usb', 'onboard', 'copy_frame', 'do_resize', 'width', 'height', 'category_num', 'model'
    
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='camshift tensorrt tracking')
    parser.add_argument('-i', "--input", dest='input', help='full path to input video that will be processed')
    parser.add_argument('-f', "--fskip", dest='fskip', help='frameskip to be used')
    parser.add_argument('-o', "--output", dest='output', help='full path for saving processed video output')
    args = parser.parse_args()
    if args.input is None or args.output is None or args.fskip is None:
        sys.exit("Please provide path to input or output video files! See --help")

    #instanciacion de los argumentos
    args2 = ArgsHelper(image=None, video=None, video_looping=False,rtsp=None, rtsp_latency=200, usb=0, onboard=None, copy_frame=False, do_resize=False, width=416, height=416, category_num=1, model='yolov4-tiny-head-416')
    trt = DetectTensorRT(args2)
    trt.load_tensorRT()

    #init centroid tracker
    ct = CentroidTracker(maxDisappeared=10,maxDistance=60)



    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(args.input)

    out = cv2.VideoWriter(
            args.output, fourcc, 25,
            (704, 576))
    count = int(args.fskip)

    start_time = datetime.datetime(100,1,1,10,26,47)
    fps = cap.get(cv2.CAP_PROP_FPS)
    


    #Esta lista contendra todas las lineas instanciadas para cada video
    #este argumento en un futuro sera automatizado desde el
    #sistema centralizado que manejará todos los buses.


    lines = []
    lines.append(line_track(np.array((125,110)), np.array((550,110))))
    lines.append(line_track(np.array((125,120)), np.array((550,120))))




    trakeable = {}

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        #Frame skip para ver comportamiento ante menos frames



        #resize y deteccion de cabezas
        if count == int(args.fskip):      

            #frame = cv2.resize(frame,(416,416)) 
            boxes, confs, clss =  trt.process_img(frame)
            img = trt.vis.draw_bboxes(frame, boxes, confs, clss)
        tim = time.time()

        # Creacion y update de IDS
        del_list = []
        objects = ct.update(boxes)
        try:
            for (objectID, centroid) in objects.items():

                    #verifica si una id de alguna cabeza ya cruzo
                    #por alguna de las lineas

                    for line in lines:
                        if objectID not in line.counted.keys():
                            line.counted[objectID] = False  

                    if objectID not in trakeable :
                        trakeable[objectID] = [centroid]
                    else: 

                        #si no ha cruzado aun, se espera que la id
                        #exista al menos un minimo de 2 frames
                        #para poder calcular su direccion.
                        if len(trakeable[objectID]) > 2:
                            trakeable[objectID].append(centroid)
                            del trakeable[objectID][0]

                            for from_, to in zip(trakeable[objectID][-3:], trakeable[objectID][-2:]):
                                
                                #se dibuja los centroides, en caso de que la distancia entre centroides
                                #de la misma id sea mayor a 50, se dibuja una linea roja, 
                                #solo por temas visuales para ver los saltos de las perdidas
                                #de id

                                a = from_
                                b = to
                                x1,y1 = a
                                x2,y2 = b
                                dst = distance.euclidean(a, b)
                                if dst > 50:
                                    cv2.line(frame, tuple(a),tuple(b), (0, 0, 255), 2)
                                else:
                                    cv2.line(frame, tuple(a),tuple(b), (0, 255, 0), 2)

                                for line in lines:
                                        if line.counted[objectID] == False:
                                            #Calculo de la direccion del centroide, 
                                            #un ejemplo visual de lo que esta ocurriendo
                                            #Se puede ver en este geogebra
                                            #https://www.geogebra.org/classic/evayn9xw
                                            up = 0
                                            down = 0
                                            if intersect((x1,y1), (x2,y2), line.pline1, line.pline2):  
                                    
                                                v1 = (x2-x1,y2-y1)
                                                v2 = line.pline2-line.pline1
                                                a = math.atan2(v1[1],v1[0])
                                                b = math.atan2(v2[1],v2[0])
                                                if a<0:
                                                    a = a + 2*math.pi
                                                if b<0:
                                                    b = b + 2*math.pi
                                                va = b-a
                                                if va < 0:
                                                    va = va + 2*math.pi
                                                if va > math.pi:
                                                    #sube
                                                    line.count_up()
                                                else:
                                                    #baja
                                                    line.count_down()
                                                del_list.append(objectID)
                                                line.counted[objectID] = True



                                    

                        else:
                            trakeable[objectID].append(centroid)

                    text = "ID {}".format(objectID)

                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        except:
            continue


        #finalmente, se dibujan en el frame la metadata obtenida.
        print(time.time()-tim)

        #fskip
        if count != int(args.fskip): 
            count+=1 
        else: 
            count = 0   


        cv2.rectangle(frame, (0, 0), (130, 40), (0,0,0), -1)
        cv2.putText(frame,'in: '+str(lines[0].countup)+ ' out: '+str(lines[0].countdown), (3,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
        cv2.line(frame, tuple(lines[0].pline1), tuple(lines[0].pline2), (0, 255, 255),2)

        cv2.putText(frame,'in: '+str(lines[1].countup)+ ' out: '+str(lines[1].countdown), (3,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
        cv2.line(frame, tuple(lines[1].pline1), tuple(lines[1].pline2), (255, 255, 0),2) 
        #out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
2