import cv2
import numpy as np
mPATH = "./correction/IMG_2843_cor.jpg"

# mPATH = "ex1.JPG"

img = cv2.imread(mPATH, cv2.IMREAD_COLOR)

mask = np.zeros_like(img)

w = mask.shape

# fov = 182.68
fov = 180
# f = 526.1783
# f = 969.4650234062414
fx = 912.2231912235586
fy = 912.2231912235586
f = 912.2231912235586

rad =np.deg2rad(fov/2)

radius = 2*f*np.sin(rad/2)

cx = int(799.1734504717965)
cy = int(803.2112360941801)

# cx = int(803.909238589)
# cy = int(806.0813101590001)

x_l =w[0]+100
x_h = 0
y_l= w[1]+100
y_h= 0

# BB
for i in range(w[0]):
    for j in range(w[1]):
        
        a = i-fx
        b = j-fy

        can = np.sqrt(a*a+b*b)

        if can <= radius:

            if(i < y_l):
                y_l = i
            if(j < x_l):
                x_l = j
            if(i > y_h):
                y_h = i
            if(j > x_h):
                x_h = j

# crop
mask2 = cv2.circle(mask, (cx,cy), int(radius),(255,255,255), -1)
res = cv2.bitwise_and(mask2, img)

res = res[x_l:x_h,y_l:y_h]

# res = cv2.resize(res, (1000,1000))
cv2.imshow("res", res)

cv2.imwrite("crop2.jpg", res)

cv2.waitKey(0)