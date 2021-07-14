#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <chrono>
#include <vector>
#include <thread>
#include <malloc.h>
#include "mlut.h"
#include "cuda_runtime.h"

using namespace std;
using namespace cv;

// test with CPU
// typedef struct{
//     int x;
//     int y;
// }int2_t;
// typedef struct{
//     float x;
//     float y;
//     float z;
//     float w;
// }float4_t;
// typedef struct{
//     unsigned char x;
//     unsigned char y;
//     unsigned char z;
// }uchar3_t;

uchar3* upload_image_in_GPU(cv::Mat image);

__global__ void equisolid2Equirect(uchar3* img_dev, int lutsize_dev, int2* xy_dev, int2* txty_dev, float4* coefs_dev, uchar3* equirect_dev);

// test with CPU
// void test(uchar3_t* img_dev, int lutsize_dev, int2_t* xy_dev, int2_t* txty_dev, float4_t* coefs_dev, uchar3_t* equirect_dev);

int main(int argc, char** argv)
{
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    int cuda_devices_number = cv::cuda::getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    cv::cuda::DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

    // Find Maximum thread number
    const int threadnum = std::thread::hardware_concurrency();
    std::cout << "thread number : " << threadnum << std::endl;
    
    // Load lookup table
    mlut::mLUT lutt;
    mlut::load(lutt, "lut.txt");
    std::vector<mlut::mappingData*> mapData = lutt.getMaps();
    if(mapData.empty()) std::cout<<"ABORT! empty vector"<<std::endl;

    // Define input, output image size
    cv::Mat srcimg = cv::imread("ex2.JPG");
    cv::Size viewSize = srcimg.size();
    
    const int equirect_w = 1024;
    const int equirect_h = 512;
    
    // for CPU
    cv::Size newsize(equirect_w, equirect_h);
    cv::Mat equirect = cv::Mat::zeros(newsize, srcimg.type());

    int lutsize= static_cast<int>(mapData.size());
    int2* xy = new int2[lutsize];
    int2* txty = new int2[lutsize];
    float4* coefs = new float4[lutsize];

    // test with CPU
    // int2_t* xy = new int2_t[lutsize];
    // int2_t* txty = new int2_t[lutsize];
    // float4_t* coefs = new float4_t[lutsize];

    for(int i=0; i<lutsize; i++)
    {
        mlut::mappingData map = *mapData[i];
        mlut::xy xy_ = map.getXY();
        mlut::txty txty_ = map.getTXTY();
        mlut::ipCoefs coefs_ = map.getIpCoefs();

        xy[i].x = xy_.getX();
        xy[i].y = xy_.getY();

        txty[i].x = txty_.getTX();
        txty[i].y = txty_.getTY();

        coefs[i].x = coefs_.getBL();
        coefs[i].y = coefs_.getTL();
        coefs[i].z = coefs_.getBR();
        coefs[i].w = coefs_.getTR();
    }

    //test with CPU
    // uchar3_t* d_fish = upload_image_in_GPU(srcimg);
    // uchar3_t* d_equirect = upload_image_in_GPU(equirect);

    uchar3* d_fish = upload_image_in_GPU(srcimg);
    uchar3* d_equirect = upload_image_in_GPU(equirect);

    int2* d_xy;
    int2* d_txty;
    float4* d_coefs;
    
    cudaMalloc(&d_xy, lutsize);
    cudaMalloc(&d_txty, lutsize);
    cudaMalloc(&d_coefs, lutsize);

    cudaMemcpy(d_xy, xy, lutsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_txty, txty, lutsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coefs, coefs, lutsize, cudaMemcpyHostToDevice);
    
    // Multi-threading
    // std::vector<std::vector<mlut::mappingData*>> pool;
    // vector<thread> workers;
    // int poolsize = static_cast<int>(ceil(mapData.size() / threadnum));
    // auto offset = mapData.cbegin();
    // for(int i=0; i<threadnum; i++)
    // {
    //     std::vector<mlut::mappingData*> map;
    //     auto tmp = offset+poolsize;

    //     if (tmp >= mapData.cend()) tmp = mapData.cend()-1;
    //     map.assign(offset, tmp);
    //     pool.emplace_back(map);
    //     offset = tmp;
    // }

    // test with CPU   
    // uchar3_t* res = new uchar3_t[newsize.width * newsize.height * sizeof(uchar3_t)];

    uchar3* res = new uchar3[newsize.width * newsize.height * sizeof(uchar3)];

    int blocksize = 1024;
    int threadsize = (lutsize/blocksize)+1;
    equisolid2Equirect<<<blocksize,threadsize>>>(d_fish, lutsize, d_xy, d_txty, d_coefs, d_equirect);

    cv::Mat equirect_res(Size(1024,512), CV_8UC3);
    cudaMemcpy(equirect_res.ptr(), d_equirect, equirect_h*equirect_w*sizeof(int), cudaMemcpyDeviceToHost);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // for (int i = 0; i < threadnum; i++) {
    //     workers.emplace_back(std::thread(equisolid2Equirect, std::ref(srcimg), std::ref(pool[i]), std::ref(equirect)));
    // }
    // for (int i = 0; i < threadnum; i++) {
    //     workers[i].join();
    // }
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	
    std::cout << "elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    
    //cpu test
    // test(d_fish, lutsize, xy, txty, coefs, d_equirect);
    // for(int y=0; y<512; y++)
    // {
    //     for(int x=0; x<1024; x++)
    //     {
    //         equirect_res.at<Vec3b>(y,x)[0] = static_cast<uchar>(d_equirect[y*1024+x].x);
    //         equirect_res.at<Vec3b>(y,x)[1] = static_cast<uchar>(d_equirect[y*1024+x].y);
    //         equirect_res.at<Vec3b>(y,x)[2] = static_cast<uchar>(d_equirect[y*1024+x].z);
    //     }
    // }

    imshow("equisolid2panorama", equirect_res);
    waitKey(0);
    
    // imwrite("sample_result.jpg", equirect_res);

    cudaFree(d_fish);
    cudaFree(d_equirect);
    cudaFree(d_xy);
    cudaFree(d_txty);
    cudaFree(d_coefs);

    delete xy;
    delete txty;
    delete coefs;    

    return 0;
}

uchar3* upload_image_in_GPU(cv::Mat image)
{
    uchar3 *h_image = new uchar3[image.rows * image.cols];
    uchar3 *d_in;
    unsigned char *cvPtr = image.ptr<unsigned char>(0);

    for(int i =0; i< image.rows * image.cols; ++i)
    {
        (h_image)[i].x = cvPtr[3*i+0];
        (h_image)[i].y = cvPtr[3*i+1];
        (h_image)[i].z = cvPtr[3*i+2];
    }

    size_t numRows = image.rows;
    size_t numCols = image.cols;

    cudaMalloc((void **) &d_in, numRows*numCols* sizeof(uchar3));
    cudaMemcpy(d_in, h_image, numRows * numCols * sizeof(uchar3), cudaMemcpyHostToDevice);

    free(h_image);

    return d_in;
}

__global__ void equisolid2Equirect(uchar3* img_dev, int lutsize_dev, int2* xy_dev, int2* txty_dev, float4* coefs_dev, uchar3* equirect_dev)
{
    int n = blockIdx.x + threadIdx.x;
    
    // img size 1600, 1600
    // equirect 1024, 512

    if (n < lutsize_dev)
    {
        printf("n : %d\n", n);
        unsigned long curr_pos = txty_dev[n].y*1600+txty_dev[n].x;

        printf("curr_pos : %lu\n", curr_pos);

        float bl_b = img_dev[curr_pos].x * coefs_dev[n].x;
        float bl_g = img_dev[curr_pos].y * coefs_dev[n].x;
        float bl_r = img_dev[curr_pos].z * coefs_dev[n].x;
        
        float tl_b = img_dev[curr_pos+1600].x * coefs_dev[n].y;
        float tl_g = img_dev[curr_pos+1600].y * coefs_dev[n].y;
        float tl_r = img_dev[curr_pos+1600].z * coefs_dev[n].y;

        float br_b = img_dev[curr_pos+1].x * coefs_dev[n].z;
        float br_g = img_dev[curr_pos+1].y * coefs_dev[n].z;
        float br_r = img_dev[curr_pos+1].z * coefs_dev[n].z;
        
        float tr_b = img_dev[curr_pos+1601].x * coefs_dev[n].w;
        float tr_g = img_dev[curr_pos+1601].y * coefs_dev[n].w;
        float tr_r = img_dev[curr_pos+1601].z * coefs_dev[n].w;

        equirect_dev[xy_dev[n].y*1024+xy_dev[n].x].x = bl_b + tl_b + br_b + tr_b;
        equirect_dev[xy_dev[n].y*1024+xy_dev[n].x].y = bl_g + tl_g + br_g + tr_g;
        equirect_dev[xy_dev[n].y*1024+xy_dev[n].x].z = bl_r + tl_r + br_r + tr_r;

        printf("%d, %d, %d\n", equirect_dev[xy_dev[n].y*1024+xy_dev[n].x].x, equirect_dev[xy_dev[n].y*1024+xy_dev[n].x].y, equirect_dev[xy_dev[n].y*1024+xy_dev[n].x].z);
    }  
}

// void test(uchar3_t* img_dev, int lutsize_dev, int2_t* xy_dev, int2_t* txty_dev, float4_t* coefs_dev, uchar3_t* equirect_dev)
// {
//     for(int x_dev =0; x_dev < lutsize_dev; x_dev++)
//     {
//         unsigned long curr_pos = txty_dev[x_dev].y*1600+txty_dev[x_dev].x;

//         printf("curr_pos : %lu\n", curr_pos);

//         float bl_b = img_dev[curr_pos].x * coefs_dev[x_dev].x;
//         float bl_g = img_dev[curr_pos].y * coefs_dev[x_dev].x;
//         float bl_r = img_dev[curr_pos].z * coefs_dev[x_dev].x;
        
//         float tl_b = img_dev[curr_pos+1600].x * coefs_dev[x_dev].y;
//         float tl_g = img_dev[curr_pos+1600].y * coefs_dev[x_dev].y;
//         float tl_r = img_dev[curr_pos+1600].z * coefs_dev[x_dev].y;

//         float br_b = img_dev[curr_pos+1].x * coefs_dev[x_dev].z;
//         float br_g = img_dev[curr_pos+1].y * coefs_dev[x_dev].z;
//         float br_r = img_dev[curr_pos+1].z * coefs_dev[x_dev].z;
        
//         float tr_b = img_dev[curr_pos+1601].x * coefs_dev[x_dev].w;
//         float tr_g = img_dev[curr_pos+1601].y * coefs_dev[x_dev].w;
//         float tr_r = img_dev[curr_pos+1601].z * coefs_dev[x_dev].w;

//         equirect_dev[xy_dev[x_dev].y*1024+xy_dev[x_dev].x].x = bl_b + tl_b + br_b + tr_b;
//         equirect_dev[xy_dev[x_dev].y*1024+xy_dev[x_dev].x].y = bl_g + tl_g + br_g + tr_g;
//         equirect_dev[xy_dev[x_dev].y*1024+xy_dev[x_dev].x].z = bl_r + tl_r + br_r + tr_r;

//         printf("%d, %d, %d\n", equirect_dev[xy_dev[x_dev].y*1024+xy_dev[x_dev].x].x, equirect_dev[xy_dev[x_dev].y*1024+xy_dev[x_dev].x].y, equirect_dev[xy_dev[x_dev].y*1024+xy_dev[x_dev].x].z);
//     } 
// }