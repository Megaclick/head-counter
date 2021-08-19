import numpy as np
import cv2
import sys
sys.path.append("./thirdparty/darknet")
from video_xompass_darknet import * 
import json
from pyimagesearch.centroidtracker import CentroidTracker
from scipy.spatial import distance


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

def init_camshift(frame,bbox):
    x1,y1,x2,y2 = bbox
    # set up the ROI for tracking
    if x1 < 0: x1 = 0
    if x2 < 0: x2 = 0
    if y1 < 0: y1 = 0
    if y2 < 0: y2 = 0


    roi = frame[y1:y2, x1:x2]
    
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[60],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist,bbox

def main():
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 3 )
    trakeable = {}

    with open("./conf/model.json",'r') as f:
        config = json.load(f)

    #change model
    net_config = config['model']["head"]

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        net_config['cfg'],
        net_config['data'],
        net_config['weights'],
        batch_size=1
    )

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("demo.avi")
    #cap = cv2.VideoCapture("flood.mp4")

    ct = CentroidTracker(maxDisappeared=10,maxDistance=60)

    bboxes_camshift = []

    camshift_list = []
    count = 10
    while True:
        # loop asking for new image paths if no list is given
        
        ret,image_name = cap.read()
        frame = image_name.copy()
        if ret:
            #Object Detection
            if count==10:
                camshift_list = []

                image, detections = image_detection(
                    image_name, network, class_names, class_colors, 0.25
                    )
                new_dets = output_to_original_tlbr(detections, image_name)
                #draw bbox on orig image

                # if new_dets is not None:
                #     for det in new_dets:
                #         x1,y1,x2,y2 = det[2]
                #         cv2.rectangle(image_name, (x1,y1), (x2,y2), (255,0,0))
                


                #generate camshift histograms
                if new_dets is not None:
                    for det in new_dets:      
                        bbox = det[2]
                        #list of histograms
                        camshift_list.append(init_camshift(image_name, bbox))
                count = 0
            
            #apply camshift
            hsv = cv2.cvtColor(image_name,cv2.COLOR_BGR2HSV)
            bboxes_camshift_prev = bboxes_camshift
            bboxes_camshift = []
            for roi_hist,bbox in camshift_list:
                x = int(bbox[0])
                y = int(bbox[1])
                w = int((bbox[2] - x)/2)
                h = int((bbox[3] - y)/2)
                track_window = (x,y,w,h)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)


                x1 = track_window[0]
                y1 = track_window[1]
                x2 = track_window[2] + x1
                y2 = track_window[3] + y1
                cx = int(x1 + track_window[2]/2)
                cy = int(y1 + track_window[3]/2)
                
                bboxes_camshift.append((x1,y1,x2,y2))
                #draw bbox
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0))
                cv2.circle(frame, (cx,cy), 1, (0,255,0),-1)
                #draw poly
                # pts = cv2.boxPoints(ret)
                # pts = np.int0(pts)
                # img2 = cv2.polylines(frame,[pts],True, 255,2)


            for box1 in bboxes_camshift:
                for box2 in bboxes_camshift_prev:
                    print(get_iou(box1, box2))


            objects = ct.update(bboxes_camshift)


            for (objectID, centroid) in objects.items():

                    #verifica si una id de alguna cabeza ya cruzo
                    #por alguna de las lineas

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
                        else:
                            trakeable[objectID].append(centroid)
                    text = "ID {}".format(objectID)

                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            count +=1
            cv2.imshow('original',image_name)
            cv2.imshow('Inference', image)    
            cv2.imshow('Camshift bbox', frame)    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()