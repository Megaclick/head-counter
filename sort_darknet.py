import numpy as np
import cv2
import sys
sys.path.append("./thirdparty/darknet")
from video_xompass_darknet import * 
import json
from sort import *

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
    
    with open("./conf/model.json",'r') as f:
        config = json.load(f)

    #change model
    net_config = config['model']["head"]
    mot_tracker = Sort()
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        net_config['cfg'],
        net_config['data'],
        net_config['weights'],
        batch_size=1
    )

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("demo.avi")
    cap = cv2.VideoCapture("flood.mp4")

    while True:
        # loop asking for new image paths if no list is given
        
        ret,image_name = cap.read()
        frame = image_name.copy()
        if ret:
            #Object Detection
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
            print(track_bbs_ids)

            sortf = frame.copy()
            for det in track_bbs_ids:
                x1 = int(det[0])
                y1 = int(det[1])
                x2 = int(det[2])
                y2 = int(det[3])
                id_ = int(det[4])
                cv2.putText(sortf, str(id_), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
                cv2.rectangle(sortf, (x1,y1), (x2,y2), (255,0,0))


           
            
            cv2.imshow('original',image_name)
            cv2.imshow('Inference', image)    
            cv2.imshow('sort', sortf)    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()