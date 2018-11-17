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
        self.ew = 2
        self.w = w
        self.h = h

    def load(self, filename):
        self.em = np.load(filename)
        plt.imshow(self.em)
        print(self.em[0,0])
        plt.show()

    def store(self, filename):
        self.em = np.minimum(self.em, Label.OLD_SCRATCH)
        np.save(filename, self.em)
        
    def get_bounding_box(self, frame, val):
        frame = cv2.resize(frame, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
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
        print(frame.shape)
        print(np.unique(frame))
        # mark the new scratch
        print(type(Label.NEW_SCRATCH))
        frame = cv2.resize(frame, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        for i in range(self.h):
            for j in range(self.w):
                if (frame[i,j] > 0 and self.em[i,j] == 0):
                    self.em[max(0,i-self.ew):min(self.h,i+self.ew+1),max(0,j-self.ew):min(self.w,j+self.ew+1)] = 1#Label.NEW_SCRATCH
        