import cv2
import numpy as np
import time
from copy import deepcopy

# mPATH = "./correction/IMG_2843_cor.jpg"
# mPATH = "./correction/IMG_2843_cor.jpg"
# mPATH = "ex2.JPG"
# mPATH = "correction_ex2.jpg"
mPATH = "crop_ex2.jpg"

def lerp(y0, y1, x0, x1, x):
    m = (y1 - y0) / (x1 - x0)   # length ratio
    b = y0                      # offset
    return m *(x-x0) + b

def crop(img, K, D, aperture):
    
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    k1, k2, k3, k4 = D[0], D[1], D[2], D[3]
    mask = np.zeros_like(img)

    w = mask.shape
    theta = aperture/2

    r = np.tan(theta)

    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta2*theta2
    theta5 = theta4*theta
    theta6 = theta3*theta3
    theta7 = theta6*theta
    theta8 = theta4*theta4
    theta9 = theta8*theta
    
    rho = (theta + k1*theta3 + k2*theta5 + k3*theta7 + k4*theta9)
    
    inv_r = 1.0/r if r > 1e-8 else 1
    cdist = rho*inv_r if r > 1e-8 else 1
 
    x_l =w[0]+100
    x_h = 0
    y_l= w[1]+100
    y_h= 0

    sx = cdist*r*fx
    sy = cdist*r*fy
    # x_l = int(799-sx)
    # x_h = int(799+sx)
    # y_l = int(799-sy)
    # y_h = int(799+sy)

    x_l = int(cx-sx)
    x_h = int(cx+sx)
    y_l = int(cy-sy)
    y_h = int(cy+sy)
    # # BB
    # for i in reversed(range(w[0])):
    #     i_dst_norm = lerp(-1, 1, 0, w[0], i)
    #     for j in range(w[1]):
    #         j_dst_norm = lerp(-1, 1, 0, w[1], j)

    #         a = cdist * j_dst_norm
    #         b = cdist * i_dst_norm
            
    #         u = int(fx*a)
    #         v = int(fy*b)
    #         can = np.sqrt(u*u+v*v)
            
    #         # print("a", )
    #         # print("an : ", can)
    #         if can <= cdist*r*f:

    #             if(i < y_l):
    #                 y_l = i
    #             if(j < x_l):
    #                 x_l = j
    #             if(i > y_h):
    #                 y_h = i
    #             if(j > x_h):
    #                 x_h = j

    # crop
    # mask2 = cv2.circle(mask, (cx,cy), int(radius),(255,255,255), -1)
    # res = cv2.bitwise_and(mask2, img)
    print((x_l,x_h,y_l,y_h))
    res = img[x_l:x_h,y_l:y_h]

    # res = cv2.resize(res, (1000,1000))
    cv2.imshow("res", res)

    cv2.waitKey(0)

    cv2.imwrite("crop_ex2.jpg", res)

    return res

def rad2vec(r, theta, phi, K, D):
    
    fx, fy = K[0][0], K[1][1]
    
    cx, cy = K[0][2], K[1][2]

    k1, k2, k3, k4 = D[0], D[1], D[2], D[3]
    
    # cv2.fisheye
    theta2 = theta*theta
    theta3 = theta2*theta
    theta4 = theta2*theta2
    theta5 = theta4*theta
    theta6 = theta3*theta3
    theta7 = theta6*theta
    theta8 = theta4*theta4
    theta9 = theta8*theta
    
    rho = (theta + k1*theta3 + k2*theta5 + k3*theta7 + k4*theta9)
    
    inv_r = 1.0/r if r > 1e-8 else 1
    cdist = rho*inv_r if r > 1e-8 else 1

    x_src_norm = r * np.cos(phi)
    y_src_norm = r * np.sin(phi)

    x_src = cdist * x_src_norm
    y_src = cdist * y_src_norm

    u = fx*x_src+cx
    v = fy*y_src+cy

    return u,v

def correction(img, K, D, aperture):

    newimg = np.zeros_like(img)

    h_src, w_src = img.shape[:2]
    h_dst, w_dst = newimg.shape[:2]

    for y in reversed(range(h_dst)):

        y_dst_norm = lerp(-1, 1, 0, h_dst, y)

        for x in range(w_dst):
            x_dst_norm = lerp(-1, 1, 0, w_dst, x)

            r = np.sqrt(x_dst_norm*x_dst_norm + y_dst_norm*y_dst_norm)
            
            phi = np.arctan2(y_dst_norm, x_dst_norm)
            theta = r*aperture/2
            
            u,v = rad2vec(r, theta, phi, K, D)

            # ##################################
            # ### bilinear interpolation
            tx = np.minimum(w_src - 1, np.floor(u).astype(int))
            ty = np.minimum(w_src - 1, np.floor(v).astype(int))

            a = u - tx
            b = v - ty

            if(tx >= 0 and tx < w_src -1 and ty >= 0 and ty < h_src-1):                    
                if tx == w_src -1:
                    tx-=1
                if ty == h_src -1:
                    ty-=1
                
                c_top = img[ty+1][tx] * (1. - a) + img[ty+1][tx+1] * (a)
                c_bot = img[ty][tx] * (1. - a) + img[ty][tx+1] * (a)
                newimg[y][x] = c_bot * (1. -b) + c_top * (b)
    
    return newimg

if __name__=="__main__":

    K_of = [[529.5765235492094, 0, 804.0019757188489]
            ,[0, 527.6442756230106, 805.0017671265209]
            ,[0, 0, 1]]   
    
    D_of = [-0.03785656050016772,
            0.01755371622463982,
            -0.01627118760065555,
            0.003967796010329191]

    APERTURE = np.deg2rad(180)

    newsize = (1024, 512)

    src_img = cv2.imread(mPATH)
    
    # blank = np.zeros_like(src_img)
    
    start = time.time()
    res = correction(deepcopy(src_img), K_of, D_of, APERTURE)
    # res = crop(deepcopy(src_img), K_of, D_of, APERTURE)
    end = time.time()

    cv2.imshow("res", res)
    cv2.waitKey(0)
    print("elapsed time : {}s".format(end-start))
    cv2.imwrite("correction_cropped_ex2.jpg", res)
    
    print("finished!")