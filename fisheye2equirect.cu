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
typedef struct{
    int x;
    int y;
}int2_host;
typedef struct{
    float x;
    float y;
    float z;
    float w;
}float4_host;
typedef struct{
    unsigned char x;
    unsigned char y;
    unsigned char z;
}uchar3_host;

uchar3_host* upload_image_on_CPU(cv::Mat image);

uchar3* upload_image_on_GPU(cv::Mat image);

// test on CPU
void equisolid2Equirect_CPU(uchar3_host* img_host, int lutsize_host, int2_host* xy_host, int2_host* txty_host, float4_host* coefs_host, uchar3_host* equirect_host);

__global__ void equisolid2Equirect(uchar3* img_dev, int lutsize_dev, int2* xy_dev, int2* txty_dev, float4* coefs_dev, uchar3* equirect_dev);

int main(int argc, char** argv)
{
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    std::string filename = argv[1];
    std::cout << "filename :" << filename << std::endl;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    int cuda_devices_number = cv::cuda::getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    cv::cuda::DeviceInfo _deviceInfo;
    bool _is_device_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _is_device_compatible << endl;

    // Find Maximum thread number
    const int threadnum = std::thread::hardware_concurrency();
    std::cout << "thread number : " << threadnum << std::endl;
    
    const int equirect_w = 1024;
    const int equirect_h = 512;
    cv::Size newsize(equirect_w, equirect_h);

    // Load lookup table
    mlut::mLUT lutt;
    mlut::load(lutt, "lut.tab");
    std::vector<mlut::mappingData*> mapData = lutt.getMaps();
    if(mapData.empty())
    {
        std::cerr<<"ABORT! empty vector"<<std::endl;
        exit(0);
    }
    int lutsize= static_cast<int>(mapData.size());
    
    ////////////////////////////////////////////////////////////
    // CPU version
    {
        // Define input, output image size
        cv::Mat srcimg = cv::imread(filename);
        cv::Size viewSize = srcimg.size();
        cv::Mat equirect_host = cv::Mat::zeros(newsize, srcimg.type());

        int2_host* xy_host = new int2_host[lutsize];
        int2_host* txty_host = new int2_host[lutsize];
        float4_host* coefs_host = new float4_host[lutsize];

        for(int i=0; i<lutsize; i++)
        {
            mlut::mappingData map = *mapData[i];
            mlut::xy xy_ = map.getXY();
            mlut::txty txty_ = map.getTXTY();
            mlut::ipCoefs coefs_ = map.getIpCoefs();

            xy_host[i].x = xy_.getX();
            xy_host[i].y = xy_.getY();

            txty_host[i].x = txty_.getTX();
            txty_host[i].y = txty_.getTY();

            coefs_host[i].x = coefs_.getBL();
            coefs_host[i].y = coefs_.getTL();
            coefs_host[i].z = coefs_.getBR();
            coefs_host[i].w = coefs_.getTR();
        }
        uchar3_host* h_fish = upload_image_on_CPU(srcimg);
        uchar3_host* h_equirect = upload_image_on_CPU(equirect_host);
        
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

        // for (int i = 0; i < threadnum; i++) {
        //     workers.emplace_back(std::thread(equisolid2Equirect, std::ref(srcimg), std::ref(pool[i]), std::ref(equirect)));
        // }
        // for (int i = 0; i < threadnum; i++) {
        //     workers[i].join();
        // }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        equisolid2Equirect_CPU(h_fish, lutsize, xy_host, txty_host, coefs_host, h_equirect);
        memcpy(equirect_host.ptr(), h_equirect, equirect_h*equirect_w*sizeof(uchar3_host));
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();	
        std::cout << "[HOST] elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

        imshow("host equisolid2panorama", equirect_host);
        waitKey(0);
    }

    ////////////////////////////////////////////////////////
    //// GPU version
    int2* xy = new int2[lutsize];
    int2* txty = new int2[lutsize];
    float4* coefs = new float4[lutsize];

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

    int threadsize = 512; // Maximum thread size on a block
    int blocksize = (lutsize+threadsize-1)/threadsize;

    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    {
        // Define input, output image size
        cv::Mat srcimg = cv::imread(filename);
        cv::Mat equirect = cv::Mat::zeros(newsize, srcimg.type());

        uchar3* d_fish = upload_image_on_GPU(srcimg);
        uchar3* d_equirect = upload_image_on_GPU(equirect);

        int2* d_xy;
        int2* d_txty;
        float4* d_coefs;
        
        cudaMalloc(&d_xy, lutsize*sizeof(int2));
        cudaMalloc(&d_txty, lutsize*sizeof(int2));
        cudaMalloc(&d_coefs, lutsize*sizeof(float4));

        cudaMemcpy(d_xy, xy, lutsize*sizeof(int2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_txty, txty, lutsize*sizeof(int2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_coefs, coefs, lutsize*sizeof(float4), cudaMemcpyHostToDevice);

        cudaEventRecord(start, 0);
        equisolid2Equirect<<<blocksize,threadsize>>>(d_fish, lutsize, d_xy, d_txty, d_coefs, d_equirect);
        cudaMemcpy(equirect.ptr(), d_equirect, equirect_h*equirect_w*sizeof(uchar3), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);

        std::cout << "[DEVICE] GPU operation elapsed time : " << elapsedTime << "ms" << std::endl;

        // imshow("equisolid2panorama", equirect);
        // waitKey(0);
        
        // imwrite("sample_result.jpg", equirect_res);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_fish);
        cudaFree(d_equirect);
        cudaFree(d_xy);
        cudaFree(d_txty);
        cudaFree(d_coefs);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();	
    std::cout << "[DEVICE] the elapsed time of the whole process: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    delete xy;
    delete txty;
    delete coefs;

    return 0;
}

uchar3_host* upload_image_on_CPU(cv::Mat image)
{
    uchar3_host *h_image_host = new uchar3_host[image.rows * image.cols];
    unsigned char *cvPtr_host = image.ptr<unsigned char>(0);

    for(int i =0; i< image.rows * image.cols; ++i)
    {
        (h_image_host)[i].x = cvPtr_host[3*i+0];
        (h_image_host)[i].y = cvPtr_host[3*i+1];
        (h_image_host)[i].z = cvPtr_host[3*i+2];
    }
    return h_image_host;
}

uchar3* upload_image_on_GPU(cv::Mat image)
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

void equisolid2Equirect_CPU(uchar3_host* img_host, int lutsize_host, int2_host* xy_host, int2_host* txty_host, float4_host* coefs_host, uchar3_host* equirect_host)
{
    for(int x_host =0; x_host < lutsize_host; x_host++)
    {
        unsigned long curr_pos_host = txty_host[x_host].y*1600+txty_host[x_host].x;

        // if(x_host < 55000 && 54000<x_host)
        //     printf("[HOST] tx, ty : %d, %d and offset : %d \n", txty_host[x_host].x, txty_host[x_host].y, x_host);

        float bl_b_host = img_host[curr_pos_host].x * coefs_host[x_host].x;
        float bl_g_host = img_host[curr_pos_host].y * coefs_host[x_host].x;
        float bl_r_host = img_host[curr_pos_host].z * coefs_host[x_host].x;
        
        float tl_b_host = img_host[curr_pos_host+1600].x * coefs_host[x_host].y;
        float tl_g_host = img_host[curr_pos_host+1600].y * coefs_host[x_host].y;
        float tl_r_host = img_host[curr_pos_host+1600].z * coefs_host[x_host].y;

        float br_b_host = img_host[curr_pos_host+1].x * coefs_host[x_host].z;
        float br_g_host = img_host[curr_pos_host+1].y * coefs_host[x_host].z;
        float br_r_host = img_host[curr_pos_host+1].z * coefs_host[x_host].z;
        
        float tr_b_host = img_host[curr_pos_host+1601].x * coefs_host[x_host].w;
        float tr_g_host = img_host[curr_pos_host+1601].y * coefs_host[x_host].w;
        float tr_r_host = img_host[curr_pos_host+1601].z * coefs_host[x_host].w;

        equirect_host[xy_host[x_host].y*1024+xy_host[x_host].x].x = bl_b_host + tl_b_host + br_b_host + tr_b_host;
        equirect_host[xy_host[x_host].y*1024+xy_host[x_host].x].y = bl_g_host + tl_g_host + br_g_host + tr_g_host;
        equirect_host[xy_host[x_host].y*1024+xy_host[x_host].x].z = bl_r_host + tl_r_host + br_r_host + tr_r_host;
    } 
}

__global__ void equisolid2Equirect(uchar3* img_dev, int lutsize_dev, int2* xy_dev, int2* txty_dev, float4* coefs_dev, uchar3* equirect_dev)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    // img size 1600, 1600
    // equirect 1024, 512

    if (offset < lutsize_dev)
    {
        unsigned long curr_pos = txty_dev[offset].y*1600+txty_dev[offset].x;

        float bl_b = img_dev[curr_pos].x * coefs_dev[offset].x;
        float bl_g = img_dev[curr_pos].y * coefs_dev[offset].x;
        float bl_r = img_dev[curr_pos].z * coefs_dev[offset].x;
        
        float tl_b = img_dev[curr_pos+1600].x * coefs_dev[offset].y;
        float tl_g = img_dev[curr_pos+1600].y * coefs_dev[offset].y;
        float tl_r = img_dev[curr_pos+1600].z * coefs_dev[offset].y;

        float br_b = img_dev[curr_pos+1].x * coefs_dev[offset].z;
        float br_g = img_dev[curr_pos+1].y * coefs_dev[offset].z;
        float br_r = img_dev[curr_pos+1].z * coefs_dev[offset].z;
        
        float tr_b = img_dev[curr_pos+1601].x * coefs_dev[offset].w;
        float tr_g = img_dev[curr_pos+1601].y * coefs_dev[offset].w;
        float tr_r = img_dev[curr_pos+1601].z * coefs_dev[offset].w;

        equirect_dev[xy_dev[offset].y*1024+xy_dev[offset].x].x = bl_b + tl_b + br_b + tr_b;
        equirect_dev[xy_dev[offset].y*1024+xy_dev[offset].x].y = bl_g + tl_g + br_g + tr_g;
        equirect_dev[xy_dev[offset].y*1024+xy_dev[offset].x].z = bl_r + tl_r + br_r + tr_r;
        }  
}