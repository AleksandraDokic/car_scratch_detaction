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

    def __init__(self, w=360, h=200):
        self.em = np.zeros((h,w))
        self.ew = 6
        self.w = w
        self.h = h
        self.cnt_frame = 0
        self.threshold = 0.2
        self.cem = np.zeros_like(self.em)

    def load(self, filename):
        self.threshold = 0.7
        self.em = np.load(filename)

    def store(self, filename):
        self.em = np.minimum(self.em, Label.OLD_SCRATCH)
        np.save(filename, self.em)
     
    def resize_frame(self, frame):
        h, w = frame.shape
        #print(h, w)
        frame = frame[h/3:h, w/5:4*w/5]
        cv2.imshow("Cut frame", frame)
        #print(frame.shape)
        frame = cv2.resize(frame, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        #print(frame.shape)
        return frame
        
    def get_bounding_box(self, frame, val):
        frame = self.resize_frame(frame)
        mini, minj = self.h, self.w
        maxi, maxj = 0, 0
        for i in range(self.h):
            for j in range(self.w):
                if (frame[i][j] > 0):
                    mini = min(mini, i)
                    minj = min(mini, j)
                    maxi = max(maxi, i)
                    maxj = max(maxj, j)
        
        return (mini, minj, maxi, maxj) 
        
    def get_old_scratch(self, frame):
        return self.get_bounding_box(frame, Label.OLD_SCRATCH)

    def get_new_scratch(self, frame):
        return self.get_bounding_box(frame, Label.NEW_SCRATCH)

    def add_new_edges(self, frame, val = Label.NEW_SCRATCH):
        frame = self.resize_frame(frame)
        
        if (self.cnt_frame < 20):
            self.cem += frame
            self.cnt_frame += 1
        else:
            self.cnt_frame = 0
            self.cem /= 255
            print(frame.shape)
            print(np.unique(frame))
            # mark the new scratch
            print(type(Label.NEW_SCRATCH))
            for i in range(self.h):
                for j in range(self.w):
                    if (self.cem[i,j] > self.threshold*20 and self.em[i,j] == 0):
                        self.em[max(0,i-self.ew):min(self.h,i+self.ew+1),max(0,j-self.ew):min(self.w,j+self.ew+1)] = 1#Label.NEW_SCRATCH
                    
            self.cem = np.zeros_like(self.em)
        cv2.imshow("Car features", self.em)