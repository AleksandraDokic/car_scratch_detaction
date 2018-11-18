import numpy as np
from enum import IntEnum
import cv2
import matplotlib.pyplot as plt

class Label(IntEnum):
    EMPTY = 0
    EDGE = 1
    OLD_SCRATCH = 2
    NEW_SCRATCH = 3

class EdgeDetector:    

    def __init__(self, w=200, h=200):
        self.em = np.zeros((h,w))
        self.ew = 5
        self.ewl = 30
        self.w = w
        self.h = h
        self.cnt_frame = 0
        self.threshold = 0.1
        self.cem = np.zeros_like(self.em)
        self.oh = 0
        self.ow = 0

    def load(self, filename):
        self.threshold = 0.3
        self.em = np.load(filename)
        print("LOAD -> ", np.amax(self.em))

    def store(self, filename):
        self.em = np.minimum(self.em, Label.OLD_SCRATCH)
        if (np.any(self.em == Label.NEW_SCRATCH)):
            print("NEW -> ERROR")
        elif(np.any(self.em == Label.OLD_SCRATCH)):
            print("OLD -> GOOD")
        
        np.save(filename, self.em)
     
    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        #print(h, w)
        frame = frame[h/3:h, w/3:2*w/3]
        #cv2.imshow("Cut frame", frame)
        #print(frame.shape)
        frame = cv2.resize(frame, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        #print(frame.shape)
        return frame
        
    def get_bounding_box(self, val):
        mini, minj = self.h, self.w
        maxi, maxj = 0, 0
        for i in range(self.h):
            for j in range(self.w):
                if (self.em[i][j] == val):
                    print(val, " pixel ", i, j)
                    mini = min(mini, i)
                    minj = min(minj, j)
                    maxi = max(maxi, i)
                    maxj = max(maxj, j)
        
        #print(mini, maxi, minj, maxj)
        #cv2.rectangle(self.em, (minj, mini), (maxj, maxi), (255,0,0), 2)
        #cv2.imshow("Bounding box", self.em)
        
        mini = 1./3. + 2./3.*(float(mini))/(float(self.h))
        maxi = 1./3. + 2./3.*(float(maxi))/(float(self.h))
        
        minj = 1./3. + 1./3.*(float(minj))/(float(self.w))
        maxj = 1./3. + 1./3.*(float(maxj))/(float(self.w))
            
        print(minj, maxj)
        return (mini, minj, maxi, maxj) 
        
    def get_old_scratch(self):
        return self.get_bounding_box(Label.OLD_SCRATCH)

    def get_new_scratch(self):
        return self.get_bounding_box(Label.NEW_SCRATCH)

    def add_new_edges(self, frame, final = False):
        frame = self.resize_frame(frame)
        
        if (not final):
            self.cem += frame
            self.cnt_frame += 1
        else:
            self.cem /= 255
            nem = np.zeros_like(self.em)
            #print(frame.shape)
            #print(np.unique(frame))
            # mark the new scratch
            for i in range(self.h):
                for j in range(self.w):
                    if (self.cem[i,j] > self.threshold*self.cnt_frame and self.em[i,j] == 0):
                        if (np.any(self.em[max(0,i-self.ewl):min(self.h,i+self.ewl+1),max(0,j-self.ewl):min(self.w,j+self.ewl+1)] == Label.OLD_SCRATCH)):
                            nem[max(0,i-self.ew):min(self.h,i+self.ew+1),max(0,j-self.ew):min(self.w,j+self.ew+1)] = Label.OLD_SCRATCH
                        else:
                            nem[max(0,i-self.ew):min(self.h,i+self.ew+1),max(0,j-self.ew):min(self.w,j+self.ew+1)] = Label.NEW_SCRATCH
                            print("New pixel!", i, j)
                            
            self.em = nem
            self.cem = np.zeros_like(self.em)
        #cv2.imshow("Car features", self.em)
        #cv2.imwrite("result_features.jpeg", self.em)