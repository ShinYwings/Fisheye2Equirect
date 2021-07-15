#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <thread>
#include "mLUT.h"

using namespace std;
using namespace cv;

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

        //  tx-=1;
        // if(tx == viewSize.width -1)
        // {
        //     Vec3d c_topleft = img.at<Vec3b>(ty+1,tx) * coefs_.getTL();
        //     Vec3d c_botleft = img.at<Vec3b>(ty,tx) * coefs_.getBL();

        //     equirect.at<Vec3b>(xy_.getY(),xy_.getX()) = c_topleft + c_botleft;
        // }
        // // ty-=1;
        // else if(ty == viewSize.height -1)
        // {
        //     Vec3d c_botleft = img.at<Vec3b>(ty,tx) * coefs_.getBL();
        //     Vec3d c_botright = img.at<Vec3b>(ty,tx+1) * coefs_.getBR();

        //     equirect.at<Vec3b>(xy_.getY(),xy_.getX()) = c_botright + c_botleft;
        // }
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

    // cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    // int cuda_devices_number = cv::cuda::getCudaEnabledDeviceCount();
    // cout << "CUDA Device(s) Number: "<< cuda_devices_number << endl;
    // cv::cuda::DeviceInfo _deviceInfo;
    // bool _isd_evice_compatible = _deviceInfo.isCompatible();
    // cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

    // Find Maximum thread number
    constexpr int threadnum = std::thread::hardware_concurrency();

    std::cout << "thread number : " << threadnum << std::endl;
    Mat srcimg = cv::imread("ex2.JPG");
    Size viewSize = srcimg.size();
    Size newsize(1024,512);
    Mat equirect = cv::Mat::zeros(newsize, srcimg.type());
    
    mlut::mLUT lutt;
    mlut::load(lutt, "lut.txt");

    std::vector<mlut::mappingData *> mapData = lutt.getMaps();
    if(mapData.empty()) std::cerr<<"ABORT! empty vector"<<std::endl;

    std::vector<std::vector<mlut::mappingData*>> pool;
    vector<thread> workers;
    int poolsize = static_cast<int>(ceil(mapData.size() / threadnum));
    auto offset = mapData.cbegin();
    for(int i=0; i<threadnum; i++)
    {
        std::vector<mlut::mappingData*> map;
        auto tmp = offset+poolsize;

        if (tmp >= mapData.cend()) tmp = mapData.cend()-1;
        map.assign(offset, tmp);
        pool.emplace_back(map);
        offset = tmp;
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < threadnum; i++) {
        workers.emplace_back(std::thread(equisolid2Equirect, std::ref(srcimg), std::ref(pool[i]), std::ref(equirect)));
    }
    for (int i = 0; i < threadnum; i++) {
        workers[i].join();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    imshow("equisolid2panorama", equirect);
    waitKey(0);
    
    imwrite("sample_result.jpg", equirect);

    return 0;
}