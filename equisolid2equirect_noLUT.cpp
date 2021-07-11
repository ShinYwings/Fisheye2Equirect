#include <algorithm>
#include <cstdio>
#include <iostream>
#include <dirent.h>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;

inline double lerp(double y0, double y1, double x0, double x1, double x)
{
    double m = (y1-y0) / (x1-x0);
    double b = y0;
    return m * (x-x0) + b;
}

inline cv::Vec2d rad2vec(double x_dst_norm, double y_dst_norm, const cv::Vec2d& f, const cv::Vec2d& c, const double& aperture)
{   
    double longitude = x_dst_norm * CV_PI + CV_PI / 2;
    double latitude = y_dst_norm * CV_PI / 2;

    double p_x = - cos(latitude) * cos(longitude);
    double p_y = cos(latitude) * sin(longitude);
    double p_z = sin(latitude);

    Vec3d p = Vec3d(p_x, p_y, p_z);
    cv::Mat rot = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
    cv::Mat pt = rot * p;
    
    double p_xz = sqrt(pow(pt.at<double>(0,0), 2)+pow(pt.at<double>(0,2), 2)); // cv::cuda::pow uses only float (imprecision)
    double theta = atan2(p_xz,pt.at<double>(0,1));
    // double r = ((2 * theta) / aperture);
    double phi = atan2(pt.at<double>(0,2), pt.at<double>(0,0));

    double x_src_norm = cos(phi);
    double y_src_norm = sin(phi);

    double x_src = 2 * f[0] * x_src_norm * sin(theta / 2);
    double y_src = 2 * f[1] * y_src_norm * sin(theta / 2);

    return cv::Vec2d(x_src+c[0], y_src+c[1]);
}

void equisolid2Equirect(InputArray img, const cv::Vec2d& f, const cv::Vec2d& c, const cv::Size& newsize, const double& aperture, Mat& equirect)
{
    Mat img_ = img.getMat();
    Size viewSize = img_.size();

    // Scan vertical line above FoV 0 with margin 10
    for(int y = 0; y < newsize.height / 2 + 10 ; y++)
    {
        double y_dst_norm = lerp(-1,1,0,newsize.height, y);

        for(int x=0; x< newsize.width; x++)
        {
            double x_dst_norm = lerp(-1,1,0,newsize.width, x);
            
            Vec2d uv = rad2vec(x_dst_norm, y_dst_norm, f, c, aperture);
            
            if(y == newsize.height / 2 ) equirect.at<Vec3b>(y,x) = Vec3b(0, 0., 255.);
            else
            {
                double tx = min(viewSize.width - 1, static_cast<int>(floor(uv[0])));
                double ty = min(viewSize.height - 1, static_cast<int>(floor(uv[1])));

                double a = uv[0] - tx;
                double b = uv[1] - ty;

                if(tx >= 0 && tx < viewSize.width -1 && ty >=0 && ty < viewSize.height -1)
                {
                    if(tx == viewSize.width -1) tx-=1;
                    if(ty == viewSize.height -1) ty-=1;
                    Vec3d c_top = img_.at<Vec3b>(ty+1,tx) * (1. - a) + img_.at<Vec3b>(ty+1,tx+1) * (a); 
                    Vec3d c_bot = img_.at<Vec3b>(ty,tx) * (1. - a) + img_.at<Vec3b>(ty,tx+1) * (a);
                    equirect.at<Vec3b>(y,x) = c_bot * (1. - b) + c_top * b;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    Mat srcimg = cv::imread("crop_sample_with_line.jpg");
    Size viewSize = srcimg.size();
    Size newsize(1024,512);
    Mat equirect = cv::Mat::zeros(newsize, srcimg.type());
    
    double aperture = 180. * CV_PI / 180.;
    double cx = viewSize.height / 2;
    double cy = viewSize.width / 2;
    double fx = cx / sqrt(2);
    double fy = cy / sqrt(2);
    
    // Ideal intrinsic parameters
    // double cx = 799.5;
    // double cy = 806.5;
    // double fx = 526.1783;
    // double fy = 526.1783;
    
    Vec2d f(fx, fy);
    Vec2d c(cx, cy);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 
    equisolid2Equirect(srcimg, f, c, newsize, aperture, equirect);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << endl;
    imshow("equisolid2panorama", equirect);
    imwrite("sample_result.jpg", equirect);
    waitKey(0);
    
    return 0;
}