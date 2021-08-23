import numpy as np
import cv2
import sys
sys.path.append("./thirdparty/darknet")
from darknet_api import * 
import json
from scipy.spatial import distance
from tracker import IDetectionMetadata

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import matplotlib.pyplot as plt
import pandas as pd
import math

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




#https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def get_iou(boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou



#transform from tlwh to tlbr 
def output_to_original_tlbr(dets,orig_frame):
    height, width, channels = orig_frame.shape
    new_dets = []
    for det in dets:
        x,y,w,h = det[2]
        nx = int(x*width/416)
        ny = int(y*height/416)
        nw = int(w*width/416)
        nh = int(h*height/416)

        new_dets.append((det[0],det[1],(nx,ny,nw,nh)))

    return new_dets


def main():
    parser = argparse.ArgumentParser(description='Sort tracking')
    parser.add_argument('-i', "--input", dest='input', help='full path to input video that will be processed')
    parser.add_argument('-f', "--fskip", dest='fskip', help='frameskip to be used')
    parser.add_argument('-o', "--output", dest='output', help='full path for saving processed video output')
    args = parser.parse_args()
    if args.input is None or args.output is None or args.fskip is None:
        sys.exit("Please provide path to input or output video files! See --help")
   
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric,max_age=30,n_init=1)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


    with open("./conf/model.json",'r') as f:
        config = json.load(f)

    #init model darknet
    net_config = config['model']["head"]

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        net_config['cfg'],
        net_config['data'],
        net_config['weights'],
        batch_size=1
    )


    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
            args.output, fourcc, 25,
            (704, 576))
    count = int(args.fskip)


    trakeable ={}
    lines = []

    lines.append(line_track(np.array((125,110)), np.array((550,110))))
    lines.append(line_track(np.array((125,120)), np.array((550,120))))


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


            if len(new_dets)!=0 :
                bboxes = []
                scores = []
                names = []
                for det in new_dets:
                    names.append(det[0])
                    scores.append(det[1])
                    bboxes.append(det[2])
                #transform data to deepsort
                features = encoder(frame, bboxes)

                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]   
                tracker.predict()
                tracker.update(detections)


                centroid_list = []
                # update tracks and prepare for linecount

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    centroid_list.append((track.track_id,(int(bbox[0]),int(bbox[1]))))
                    # draw bbox on screen
                    # color = colors[int(track.track_id) % len(colors)]
                    # color = [i * 255 for i in color]
                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

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

                        text = "ID {}".format(str(objectID))
                        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            #draw lines
            cv2.rectangle(frame, (0, 0), (130, 40), (0,0,0), -1)
            cv2.putText(frame,'in: '+str(lines[0].countup)+ ' out: '+str(lines[0].countdown), (3,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            cv2.line(frame, tuple(lines[0].pline1), tuple(lines[0].pline2), (0, 255, 255),2)

            cv2.putText(frame,'in: '+str(lines[1].countup)+ ' out: '+str(lines[1].countdown), (3,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
            cv2.line(frame, tuple(lines[1].pline1), tuple(lines[1].pline2), (255, 255, 0),2)  

            out.write(frame)
            #fskip
            if count != int(args.fskip): 
                count+=1 
            else: 
                count = 0
                
            #draw bbox on orig image
            if new_dets is not None:
                for det in new_dets:
                    x1,y1,x2,y2 = det[2]
                    cv2.rectangle(image_name, (x1,y1), (x2,y2), (255,0,0))
            
            cv2.imshow("deepsort",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()