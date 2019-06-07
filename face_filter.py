#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:55:39 2019

@author: snaily
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import dlib
import cv2
import time
from scipy.spatial import distance as dist
from collections import OrderedDict


# rectangle to bounding box
def rect_to_bb(rect):
    # take a bounding predicted by dlib and 
    # convert it to format (x,y,w,h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    
    # return a tuple of (x,y,w,h)
    return (x,y,w,h)

def shape_to_np(shape, dtype='int'):
    # initialize the list of (x,y) coordinates
    coords = np.zeros((shape.num_parts,2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    # return x, y coordinates
    return coords


def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) =img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width,int(h*r))
        
    resized = cv2.resize(img, dim, interpolation=inter)
    return resized

def face_filters(origW, origH, shape, img, choice):
    
    # realted only to eyes
    # sunglasses filter
    if choice >= 0 and choice <= 3:
        filterW = abs(int((shape[16][0] - shape[1][0])*1.1))
        filterH = int(filterW * origH / origW)
            
        y1 = int(shape[19][1])
        y2 = int(y1 + filterH)
        x1 = int(shape[27][0] - (filterW/2))
        x2 = int(x1 + filterW)
        
    # dog filter
    elif choice == 4:
        filterW = abs(int((shape[16][0] - shape[1][0])*1.5))
        filterH = int((shape[58][1] - shape[20][1])*1.5)
    
        y2 = int(shape[52][1])
        y1 = int(y2 - filterH)
        x1 = int(shape[27][0]- (filterW/2))
        x2 = int(x1 + filterW)
        
    # rabbit filter
    elif choice == 5:
        filterW = abs(int((shape[16][0] - shape[1][0])*1.5))
        filterH = int((shape[58][1] - shape[20][1])*2)
        
        y2 = int(shape[67][1])
        y1 = int(y2 - filterH)
        x1 = int(shape[27][0]- (filterW/2))
        x2 = int(x1 + filterW)
        
    # moustache filter
    elif choice == 6 or choice == 7:
        filterW = abs(shape[16][0] - shape[1][0])
        filterH = int((shape[63][1] - shape[34][1])*1.5)
        
        y1 = int(shape[34][1])
        y2 = int(y1 + filterH)
        x1 = int(shape[27][0]- (filterW/2))
        x2 = int(x1 + filterW)
        
    # ironman mask filter
    elif choice == 8:
        filterW = abs(int((shape[16][0] - shape[1][0])*1.5))
        filterH = int((shape[9][1] - shape[20][1])*1.8)
        
        y2 = int(shape[9][1] + 5)
        y1 = int(y2 - filterH)
        x1 = int(shape[27][0]- (filterW/2))
        x2 = int(x1 + filterW)
        
    # captain america mask
    elif choice == 9:
        filterW = abs(int((shape[16][0] - shape[1][0])*1.2))
        filterH = int((shape[58][1] - shape[20][1])*1.8)
        
        y2 = int(shape[52][1])
        y1 = int(y2 - filterH)
        x1 = int(shape[27][0]- (filterW/2))
        x2 = int(x1 + filterW)
        
    filters = cv2.resize(img, (filterW, filterH), interpolation = cv2.INTER_AREA)
    
    return y1, y2, x1, x2, filters, filterW, filterH



shape_predictor = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0


filters_n = ['im.png', 'sunglasses.png', 'sunglasses1.png', 'sunglasses2.png', 
             'dog.png', 'rabbit.png','moustache.png', 'moustache1.png', 'ironman.png', 'capAmerica.png']
f=0

cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    frame = resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)
    
    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        
        img = cv2.imread('filters/'+filters_n[f], -1)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retn, orig_mask = cv2.threshold(img2gray, 60, 255, cv2.THRESH_BINARY)
        orig_mask = img[:,:,3]
        orig_mask_inv = cv2.bitwise_not(orig_mask)
        img = img[:,:,0:3]
        origH, origW = img.shape[:2]
        
        y1, y2, x1, x2, filters, filterW, filterH = face_filters(origW, origH, shape, img, f)
        roi = frame[y1:y2, x1:x2]
        
        mask = cv2.resize(orig_mask, (filterW,filterH), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (filterW,filterH), interpolation = cv2.INTER_AREA)
        
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        roi_fg = cv2.bitwise_and(filters,filters,mask = mask)
        dst = cv2.add(roi_bg, roi_fg)
        frame[y1:y2, x1:x2] = dst
        
    cv2.imshow('Face Filters',  frame)
    
    k = cv2.waitKey(1)
    if k == ord('c'):
        f+=1
        f = f % len(filters_n)
        
    elif k == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
