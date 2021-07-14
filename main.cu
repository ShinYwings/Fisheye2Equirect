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

typedef struct{
    int width;
    int height;
    int* element;
}Matrix;

__global__ void CUDAFLERP_kernel(int* __restrict const d_in, const mlut::mappingData md, int* __restrict const d_out, const int neww, const int newh) {
	
    uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("here?\n");

	const uint32_t y = blockIdx.y;
	const float fy = (y + 0.5f)*gys - 0.5f;
	const float wt_y = fy - floor(fy);
	const float invwt_y = 1.0f - wt_y;
#pragma unroll
	for (int i = 0; i < 3; ++i, ++x) {
		const float fx = (x + 0.5f)*gxs - 0.5f;
        printf("%d", d_in);
		const float4 f = tex2Dgather<float4>(d_in, fx + 0.5f, fy + 0.5f);
		const float wt_x = fx - floor(fx);
		const float invwt_x = 1.0f - wt_x;
		const float xa = invwt_x*f.w + wt_x*f.z;
		const float xb = invwt_x*f.x + wt_x*f.y;
		const float res = invwt_y*xa + wt_y*xb;
		if (x < neww) d_out[y*neww + x] = res; // its right
	}
}

void CUDAFLERP(int* __restrict const d_in, const int oldw, const int oldh, int* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	
    const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
    
	CUDAFLERP_kernel<<<{neww, newh}, 256>>>(d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}

int* mat2arr1D(cv::Mat image) {

	auto image_width = image.cols;
	auto image_height = image.rows;
 	int* original_image = new int[image_width * image_height * 3];

	for (int i = 0; i < image_height; ++i) {
		for (int j = 0; j < image_width; ++j) {
			for (int k =0; k< 3; ++k) //BGR
            {
                int idx = (i*image_width)+(3*j+k);
                original_image[idx] = static_cast<int>(image.at<Vec3b>(i,j)[k]);
            }
		}
	}
    return original_image;
}

void equisolid2Equirect(cv::Mat img, const vector<mlut::mappingData*>& maps, cv::Mat& equirect)
{
    Size viewSize = img.size();

    for(vector<mlut::mappingData*>::const_iterator elem = maps.begin(); elem != maps.end(); elem++)
    {
        mlut::mappingData map = **elem;
        mlut::xy xy_ = map.getXY();
        mlut::txty txty_ = map.getTXTY();
        mlut::ipCoefs coefs_ = map.getIpCoefs();

        int tx = txty_.getTX();
        int ty = txty_.getTY();

        if((tx == viewSize.width -1) | (ty == viewSize.height -1))
        {
            equirect.at<Vec3b>(xy_.getY(),xy_.getX()) = img.at<Vec3b>(ty,tx);
        }
        else
        {
            Vec3d c_botleft = img.at<Vec3b>(ty,tx) * coefs_.getBL();
            Vec3d c_topleft = img.at<Vec3b>(ty+1,tx) * coefs_.getTL();
            Vec3d c_botright = img.at<Vec3b>(ty,tx+1) * coefs_.getBR();
            Vec3d c_topright = img.at<Vec3b>(ty+1,tx+1) * coefs_.getTR();
            
            equirect.at<Vec3b>(xy_.getY(),xy_.getX()) = c_topleft + c_botleft + c_botright + c_topright;
        }
    }
}

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
    if(mapData.empty()) std::cerr<<"ABORT! empty vector"<<std::endl;

    // Define input, output image size
    cv::Mat srcimg = cv::imread("ex2.JPG");
    cv::Size viewSize = srcimg.size();
    
    const int fisheye_w = viewSize.width;
    const int fisheye_h = viewSize.height;
    constexpr int equirect_w = 1024;
    constexpr int equirect_h = 512;
    
    // for CPU
    cv::Size newsize(equirect_w, equirect_h);
    cv::Mat equirect = cv::Mat::zeros(newsize, srcimg.type());

    int* srcimg_1d = mat2arr1D(srcimg);
    int* equirect_1d = mat2arr1D(equirect);

    
    //test
    void* img = nullptr;
    cudaAllocMapped(&img, 1600, 1600, IMAGE_RGB8);


    cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

    const size_t fisheye_total = static_cast<size_t>(fisheye_w)*static_cast<size_t>(fisheye_h)*3;
    const size_t equirect_total = static_cast<size_t>(equirect_w)*static_cast<size_t>(equirect_h)*3;

    cudaArray* d_srcimg;
    // cudaArray* d_equirect;

    cudaMallocArray(&d_srcimg, &chandesc_img, fisheye_w, fisheye_h);
	cudaMemcpy2DToArray(d_srcimg, 0, 0, srcimg_1d, sizeof(int) * static_cast<size_t>(fisheye_w), sizeof(int) * static_cast<size_t>(fisheye_w), static_cast<size_t>(fisheye_h), cudaMemcpyHostToDevice);
	// cudaMallocArray(&d_equirect, &chandesc_img, equirect_w, equirect_h);
	// cudaMemcpy2DToArray(d_equirect, 0, 0, equirect_1d, sizeof(unsigned char) * static_cast<size_t>(equirect_w), sizeof(unsigned char) * static_cast<size_t>(equirect_w), static_cast<size_t>(equirect_h), cudaMemcpyHostToDevice);
	
    struct cudaResourceDesc resdesc_fisheye_img;
	memset(&resdesc_fisheye_img, 0, sizeof(resdesc_fisheye_img));
	resdesc_fisheye_img.resType = cudaResourceTypeArray;
	resdesc_fisheye_img.res.array.array = d_srcimg;
    
    // struct cudaResourceDesc resdesc_equirect_img;
	// memset(&resdesc_equirect_img, 0, sizeof(resdesc_equirect_img));
	// resdesc_equirect_img.resType = cudaResourceTypeArray;
	// resdesc_equirect_img.res.array.array = d_equirect;
    
    int* d_in = nullptr;
	cudaMalloc(&d_in, sizeof(int) * fisheye_total);

    int* d_out = nullptr;
	cudaMalloc(&d_out, sizeof(int) * equirect_total);
    
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
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // for (int i = 0; i < threadnum; i++) {
    //     workers.emplace_back(std::thread(equisolid2Equirect, std::ref(srcimg), std::ref(pool[i]), std::ref(equirect)));
    // }
    // for (int i = 0; i < threadnum; i++) {
    //     workers[i].join();
    // }

    CUDAFLERP(d_img_tex, fisheye_w, fisheye_h, d_out, equirect_w, equirect_h);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    int* h_out = new int[equirect_w * equirect_h * 3];
    
	cudaMemcpy(h_out, d_out, sizeof(int)*equirect_total, cudaMemcpyDeviceToHost);

    std::cout << "dout :" << d_out << std::endl;
    std::cout << "elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    imshow("equisolid2panorama", equirect);
    waitKey(0);
    
    imwrite("sample_result.jpg", equirect);

    return 0;
}