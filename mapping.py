import numpy as np
import cv2
import random
from copy import deepcopy
import fish2equirect
import time

mPATH = "ex2.JPG"
# mPATH = "model2.jpg"
# mPATH = "./equirect/example.jpg"

# cv::fisheye
def model2(img, K, D):

    h, w = img.shape[:2]
    
    fx, fy = K[0][0], K[1][1]
    
    cx, cy = K[0][2], K[1][2]

    k1, k2, k3, k4 = D[0], D[1], D[2], D[3]

    for fov in range(0, 91):

        theta = np.deg2rad(fov)
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

        color = random.randint(100,255)
        color2 = random.randint(100,255)
        color3 = random.randint(100,255)

        for phi in range(0, 361):
            
            phi_ = np.deg2rad(phi)

            x =  cdist * r * np.cos(phi_)
            y =  cdist * r * np.sin(phi_)
            
            u = int(fx*x+cx)
            v = int(fy*y+cy)

            if phi%10 == 0:
                img = cv2.circle(img, (u,v), 3, (color, color2, color3), -1, cv2.LINE_AA)
            if fov == 90 :
                img = cv2.circle(img, (u,v), 3, (0, 0, 255), -1, cv2.LINE_AA)
            elif fov%10 == 0:    
                img = cv2.circle(img, (u,v), 3, (color, color2, color3), -1, cv2.LINE_AA)
            
    return img

# equisolid angle (SIGMA lens)
def equisolid(img, K, D):
    
    fx, fy = K[0][0], K[1][1]
    
    cx, cy = K[0][2], K[1][2]

    for fov in range(0, 91, 10):

        theta =np.deg2rad(fov)
        
        rho_x = 2*fx*np.sin(theta/2)
        rho_y = 2*fy*np.sin(theta/2)

        color = random.randint(0,200)
        color2 = random.randint(0,200)
        color3 = random.randint(0,200)

        for phi in range(0, 360):
            
            phi = np.deg2rad(phi)

            x =  rho_x * np.cos(phi)
            y =  rho_y * np.sin(phi)
            
            u = int(x+cx)
            v = int(y+cy)
            
            img = cv2.circle(img, (u,v), 3, (255, 0, 0), -1, cv2.LINE_AA)
    return img

# ideal fisheye angle
def equidistant(img, K, D):
    
    fx, fy = K[0][0], K[1][1]
    
    cx, cy = K[0][2], K[1][2]

    for fov in range(0, 91, 10):

        theta =np.deg2rad(fov)
        
        rho_x = fx*theta
        rho_y = fy*theta

        color = random.randint(0,200)
        color2 = random.randint(0,200)
        color3 = random.randint(0,200)

        for phi in range(0, 360):
            
            phi = np.deg2rad(phi)

            x =  rho_x * np.cos(phi)
            y =  rho_y * np.sin(phi)
            
            u = int(x+cx)
            v = int(y+cy)
            
            img = cv2.circle(img, (u,v), 3, (color, color2, color3), -1, cv2.LINE_AA)

    return img

def lerp(y0, y1, x0, x1, x):
    m = (y1 - y0) / (x1 - x0)   # length ratio
    b = y0                      # offset
    return m *(x-x0) + b

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

def fish2equi(img, K, D, size, aperture):
    h_src, w_src = img.shape[:2]
    w_dst, h_dst = size

    dst_img = np.zeros((h_dst, w_dst, 3))

    for y in reversed(range(h_dst)):
        y_dst_norm = lerp(-1, 1, 0, h_dst, y)

        for x in range(w_dst):
            x_dst_norm = lerp(-1, 1, 0, w_dst, x)

            longitude = x_dst_norm * np.pi + np.pi / 2
            latitude = y_dst_norm * np.pi / 2 
            
            p_x = np.cos(latitude) * np.cos(longitude)
            p_y = np.cos(latitude) * np.sin(longitude)
            p_z = np.sin(latitude)

            rot = np.array([[1,0,0], [0, 0, -1], [0, 1, 0]])
            p_x = -p_x
            p_x,p_y,p_z = np.matmul(rot, [p_x,p_y,p_z])
            p_xz = np.sqrt(p_x**2 + p_z**2)

            theta = np.arctan2(p_xz, p_y)
            r = ((2 * theta/ aperture))
            phi = np.arctan2(p_z, p_x)

            u,v = rad2vec(r, theta, phi, K, D)

            if y > h_dst/2 - 1 : # under 0 degree, half of the vertical position, no represetation 
                pass
            elif y == 255:
                dst_img[y][x] = (255,0,0)
            else:
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
                    dst_img[y][x] = c_bot * (1. -b) + c_top * (b)
                    
    return dst_img

def run():

    # resulting of cv::fisheye
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
    
    # of = model2(deepcopy(blank), K_of, D_of)
    # es = equisolid(deepcopy(blank), K_of, D_of)
    # ed = equidistant(deepcopy(blank), K_of, D_of)
    
    # cv2.imshow("res", cv2.bitwise_or(of, src_img))
    # cv2.waitKey(0)
    start = time.time()
    equirect = fish2equi(deepcopy(src_img), K_of, D_of, newsize, APERTURE)
    end = time.time()
    print("elapsed time : {}s".format(end-start))
    cv2.imwrite("equirect.jpg", equirect)
    
    print("finished!")

if __name__ == "__main__":
    
    
    run()
    

    
    exit(0)