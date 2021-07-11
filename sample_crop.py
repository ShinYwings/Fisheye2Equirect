import cv2
import numpy as np
import time
from copy import deepcopy

# TODO User interface (user do adjust principle points & focal length)

mPATH = "ex2.JPG"

def lerp(y0, y1, x0, x1, x):
    m = (y1 - y0) / (x1 - x0)   # length ratio
    b = y0                      # offset
    return m *(x-x0) + b

def crop(img, K, aperture):
    
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    theta = aperture/2
    
    sx = 2 * fx * np.sin(theta/2)
    sy = 2 * fy * np.sin(theta/2)

    radius = sx+sy /2
    print((sx, sy))
    x_l = int(cx-sx)
    x_h = int(cx+sx)
    y_l = int(cy-sy)
    y_h = int(cy+sy)

    img = cv2.circle(img, (int(cx),int(cy)), int(sx),(255,0,0), 3)

    print((x_l,x_h,y_l,y_h))
    res = img[y_l:y_h,x_l:x_h]

    cv2.imshow("res", res)

    cv2.waitKey(0)

    cv2.imwrite("matlab_sample.jpg", res)

if __name__=="__main__":

    # f = 533.3333333
    f = 526.1783
    print (f)
    # 806.5
    K_of = [[f, 0, 799.5]
            ,[0, f, 806.0813]
            ,[0, 0, 1]]   

    APERTURE = np.deg2rad(180)

    newsize = (1024, 512)

    src_img = cv2.imread(mPATH)
    
    # blank = np.zeros_like(src_img)
    
    start = time.time()
    crop(deepcopy(src_img), K_of, APERTURE)
    end = time.time()

    
    print("finished!")