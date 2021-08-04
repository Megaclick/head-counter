    
import cv2
import numpy as np
import time
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter("./output1.avi", fourcc, 10.0,(416, 416))
class floodfill:
    def __init__(self,start_pixel,yolo_dim,memory_len):
        self.start_pixel=start_pixel
        self.yolo_dim = yolo_dim
        self.diff = (3,3,3)
        self.memory = np.array([])
        self.memory_len = memory_len
    def process(self,frame):
        self.img = frame.copy()[:70,:] #largo para abajo
        height, width, channels = self.img.shape
        self.mask = np.zeros((height+2, width+2), np.uint8)
        self.res = cv2.floodFill(self.img, self.mask, self.start_pixel, (0,255,255), self.diff, self.diff)
        self.add_to_memory(self.res)
        return self.res,self.img
    def add_to_memory(self,res):

        if len(self.memory)<self.memory_len:
            self.memory=np.concatenate((self.memory,[self.vote(res)]))
        else:
            self.memory = np.delete(self.memory, 0)
            self.memory=np.concatenate((self.memory,[self.vote(res)]))
            
    def vote(self,res):
        if res[0] > 2000:
            return 1
        else:
            return 0 
   
   
cap = cv2.VideoCapture('flood.mp4')

flood1 = floodfill((180,22),(416,416),15)
flood2 = floodfill((211,22),(416,416),15)
flood3 = floodfill((197,24),(416,416),15)
flood4 = floodfill((273,20),(416,416),17)
flood5 = floodfill((111,37),(416,416),17)
while(cap.isOpened()):
# Capture frame-by-frame
    start = time.time()
    ret, frame = cap.read()


    if ret == True:
        frame = cv2.resize(frame,(416,416)) 
        
        res ,img= flood1.process(frame)
        res2 ,img2= flood2.process(frame)
        res3 ,img3= flood3.process(frame)
        res4 ,img4= flood4.process(frame)
        res5 ,img5= flood5.process(frame)
        #Detection
        #frame = frame * mask
        # Display the resulting frame
        
        vote = 0
        if flood1.memory.sum() >= 8: vote+=1 
        if flood3.memory.sum() >= 8: vote+=1    
        if flood2.memory.sum() >= 8: vote+=1
        if flood4.memory.sum() >= 8: vote+=1
        if flood5.memory.sum() >= 8: vote+=1
        
        if vote >=3:
            text_door = 'open'
        else:
            text_door = 'closed'
        
        frame = cv2.putText(frame, text_door, (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,1, (0, 255, 0) , 2, cv2.LINE_AA) 

        cv2.imshow('floodfill',img)
        cv2.imshow('floodfill2',img2)
        cv2.imshow('floodfill3',img3)
        cv2.imshow('floodfill4',img4)
        cv2.imshow('floodfill5',img5)
        cv2.imshow('frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

    else:
        break
cap.release()
#out.release()