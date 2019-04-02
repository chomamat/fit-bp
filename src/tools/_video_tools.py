import numpy as np
import cv2 as cv
import os
import sys

def interpolateRec(fn, img1,img2,name,out_folder,cur_depth,max_depth):
    if cur_depth == max_depth:
        return
    
    pos = max_depth - cur_depth
    name_new = name[:-pos] + '1' + ( name[-pos+1:] if pos != 1 else '' )
    
    print(name)
    print(name_new)
    print("=========")
    
    img = fn(img1,img2)
    cv.imwrite(out_folder+name_new+".png",img)
    
    interpolateRec(fn, img1,img,name,out_folder,cur_depth+1,max_depth)
    interpolateRec(fn, img,img2,name_new,out_folder,cur_depth+1,max_depth)

def interpolateFolderRec(fn, in_folder,out_folder,depth):
    images = sorted(os.listdir(in_folder))
    
    for i1,i2 in zip(images[:-1],images[1:]):
        img1 = cv.imread(in_folder+i1,cv.IMREAD_UNCHANGED)
        img2 = cv.imread(in_folder+i2,cv.IMREAD_UNCHANGED)

        interpolateRec(fn, img1,img2,i1[:-4]+'_'+''.zfill(depth),out_folder,0,depth)