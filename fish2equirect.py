# Reference
# https://github.com/ArashJavan/fisheye2equirect/blob/master/fisheye2equi.py

import numpy as np
import cv2 as cv
import random

mPATH = "./result/IMG_3063_cor.jpg"
# mPATH = "./equirect/example2.jpg"

def lerp(y0, y1, x0, x1, x):
    # m = (y1 - y0) / (x1 - x0)
    # b = y0

    # return m *(x-x0) + b
    m = (y1 - y0) / (x1 - x0)
    b = y0
    return m *(x-x0) + b

def calc_latitude(x0, y0, x1, y1, x):
    # (256, 0), (512, 1)
    # x is 0 -> output 1
    # x is 1 -> output 0
    m = (y1 - y0) / (x1 - x0)
    b = y0
    res =  m *(x-x0) + b
    res = np.abs(res-1)
    return res

# TODO offset relocation  

def fisheye2equi(src_img, size, aperture):

    h_src, w_src = src_img.shape[:2]
    w_dst, h_dst = size

    dst_img = np.zeros((h_dst, w_dst, 3))

    for y in reversed(range(h_dst)):
        y_dst_norm = lerp(-1, 1, 0, h_dst, y)

        for x in range(w_dst):
            x_dst_norm = lerp(-1, 1, 0, w_dst, x)

            # theta = y_dst_norm + np.pi - (np.pi/ 2)
            # r = x_dst_norm

            longitude = x_dst_norm * np.pi + np.pi / 2
            latitude = y_dst_norm * np.pi / 2 
            p_x = np.cos(latitude) * np.cos(longitude)
            p_y = np.cos(latitude) * np.sin(longitude)
            p_z = np.sin(latitude)

            rot = np.array([[1,0,0], [0, 0, -1], [0, 1, 0]])
            p_x,p_y,p_z = np.matmul(rot, [p_x,p_y,p_z])
            p_x = -p_x

            p_xz = np.sqrt(p_x**2 + p_z**2)
            r = ((2 * np.arctan2(p_xz, p_y) / aperture))
            theta = np.arctan2(p_z, p_x)
            x_src_norm = r * np.cos(theta)
            y_src_norm = r * np.sin(theta)

            x_src = lerp(0, w_src, -1, 1, x_src_norm)
            y_src = lerp(0, h_src, -1, 1, y_src_norm)

            # supppres out of the bound index error (warning this will overwrite multiply pixels!)
            x_src_ = np.minimum(w_src - 1, np.floor(x_src).astype(int))
            y_src_ = np.minimum(h_src - 1, np.floor(y_src).astype(int))

            if y > h_dst/2 -1 :
                pass
            else:
                dst_img[y, x, :] = src_img[y_src_, x_src_]
    return dst_img

def run():
    src_img = cv.imread(mPATH)
    
    size = (1024, 512)
    APERTURE = 180 * np.pi / 180
    equi_img = fisheye2equi(src_img, size, APERTURE)
    
    print("done!")

    cv.imshow("equirect", equi_img)
    cv.waitKey(0)
    cv.imwrite("{}/equirect.jpg".format("equirect"), equi_img)
    print("finished!")

if __name__ == "__main__":
    
    run()
    exit(0)