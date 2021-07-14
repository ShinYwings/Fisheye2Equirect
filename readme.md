## run

- fisheye image calibration of opencv::fisheye (NOT USE)
make fisheye_cali
./fisheye_cali 
./fisheye_low_res 25

currrently, it is not properly work. 
Thus, I just used to work with the equisolid mapping equation (2fsin(theta/2))

- fisheye image to equirectangular image in real-time

make generateLUT

./generateLUT
make equisolid2equirect_LUT
./equisolid2equirect_LUT

## Chessboard Specification
7 x 10 and 25mm

## Camera Specification
Canon EOS 6D (full frame) with sigma 8mm fisheye lens