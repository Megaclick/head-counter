import numpy as np
import cv2
import sys
sys.path.append("./thirdparty/darknet")
from darknet_api import * 
import json
from sort import *
import pandas as pd
from tracker import IDetectionMetadata
import math 
from scipy.spatial import distance
import argparse
import sys

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
def output_to_original_tlbr(dets,orig_frame):
    height, width, channels = orig_frame.shape
    new_dets = []
    for det in dets:
        x,y,w,h = det[2]
        nx = x*width/416
        ny = y*height/416
        nw = w*width/416
        nh = h*height/416


        x1 = int(nx- nw/2)
        y1 = int(ny -nh/2)
        x2 = int(x1+nw)
        y2 = int(y1+nh)
        new_dets.append((det[0],det[1],(x1,y1,x2,y2)))

    return new_dets



def main():
    
    parser = argparse.ArgumentParser(description='Sort tracking')
    parser.add_argument('-i', "--input", dest='input', help='full path to input video that will be processed')
    parser.add_argument('-f', "--fskip", dest='fskip', help='frameskip to be used')
    parser.add_argument('-o', "--output", dest='output', help='full path for saving processed video output')
    args = parser.parse_args()
    if args.input is None or args.output is None or args.fskip is None:
        sys.exit("Please provide path to input or output video files! See --help")

    with open("./conf/model.json",'r') as f:
        config = json.load(f)

    #change model
    net_config = config['model']["head"]
    mot_tracker = Sort(max_age=30)
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        net_config['cfg'],
        net_config['data'],
        net_config['weights'],
        batch_size=1
    )
    #Esta lista contendra todas las lineas instanciadas para cada video
    #este argumento en un futuro sera automatizado desde el
    #sistema centralizado que manejará todos los buses.
    lines = []

    lines.append(line_track(np.array((125,110)), np.array((550,110))))
    lines.append(line_track(np.array((125,120)), np.array((550,120))))

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("demo.avi")
    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
            args.output, fourcc, 25,
            (704, 576))
    count = int(args.fskip)

    trakeable = {}

    while True:
        # loop asking for new image paths if no list is given
        
        ret,image_name = cap.read()
        frame = image_name.copy()
        if ret:
            #Object Detection
            if count == int(args.fskip):
                image, detections = image_detection(
                    image_name, network, class_names, class_colors, 0.25
                    )
                new_dets = output_to_original_tlbr(detections, image_name)
            #draw bbox on orig image

            # if new_dets is not None:
            #     for det in new_dets:
            #         x1,y1,x2,y2 = det[2]
            #         cv2.rectangle(image_name, (x1,y1), (x2,y2), (255,0,0))

            #format to use with sort
            sort_input = np.empty((0,5))
            if new_dets is not None:
                for det in new_dets:
                    x1,y1,x2,y2 = det[2]
                    score = float(det[1])
                    app = np.array([int(x1),int(y1),int(x2),int(y2),score]).astype(int)
                    sort_input = np.vstack([sort_input,app])

     


            track_bbs_ids = mot_tracker.update(sort_input)
            sortf = frame.copy()
            centroid_list = []

            # get the bbox and id and draw onto img & prepare data to line count
            for det in track_bbs_ids:
                x1 = int(det[0])
                y1 = int(det[1])
                x2 = int(det[2])
                y2 = int(det[3])
                id_ = int(det[4])
                w = x2-x1
                h = y2-y1
                cx = int(x1+w/2)
                cy = int(y1+h/2)
                centroid_list.append((id_,[cx,cy]))
                cv2.putText(sortf, str(id_), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
                cv2.rectangle(sortf, (x1,y1), (x2,y2), (255,0,0))
            del_list = []

            for (objectID, centroid) in centroid_list:

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


            if count != int(args.fskip): 
                count+=1 
            else: 
                count = 0
            
            #draw lines
            cv2.rectangle(frame, (0, 0), (130, 40), (0,0,0), -1)
            cv2.putText(frame,'in: '+str(lines[0].countup)+ ' out: '+str(lines[0].countdown), (3,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            cv2.line(frame, tuple(lines[0].pline1), tuple(lines[0].pline2), (0, 255, 255),2)

            cv2.putText(frame,'in: '+str(lines[1].countup)+ ' out: '+str(lines[1].countdown), (3,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
            cv2.line(frame, tuple(lines[1].pline1), tuple(lines[1].pline2), (255, 255, 0),2)  
            out.write(frame)
            cv2.imshow('original',image_name)
            cv2.imshow('Inference', image)    
            cv2.imshow('sort', sortf)    
            cv2.imshow('tracking',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()