## Mapping fisheye image(equisolid) to equirectangular image in real-time

1. Complie generateLUT.cpp and fisheye2equirect.cu
    ```shell
        make all
    ```
2. Then, created "lut.tab"
    ```shell
    ./generateLUT.out example.jpg
    ```
3. Mapping to equirectangular image
    ```shell
    ./fisheye2equirect.out example.jpg
    ```
## previous works in the legacy directory (NOT UPDATED) 

- fisheye image calibration of opencv::fisheye (NOT USE)
make fisheye_cali
./fisheye_cali 
./fisheye_low_res 25

currrently, it is not properly work. 
Thus, I just used to work with the equisolid mapping equation (2fsin(theta/2))

make equisolid2equirect_LUT
./equisolid2equirect_LUT

## Chessboard Specification
7 x 10 and 25mm

## Camera Specification
Canon EOS 6D (full frame) with sigma 8mm fisheye lens
