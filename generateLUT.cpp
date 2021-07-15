// Reference of data serialization
// https://cheind.wordpress.com/2011/12/06/serialization-of-cvmat-objects-using-boost/

#include <iostream>
#include <algorithm>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <cmath>
#include "mlut.h"

using namespace std;
using namespace cv;

inline double lerp(double y0, double y1, double x0, double x1, double x)
{
    double m = (y1-y0) / (x1-x0);
    double b = y0;
    return m * (x-x0) + b;
}

cv::Vec2d rad2vec(double x_dst_norm, double y_dst_norm, const cv::Vec2d& f, const cv::Vec2d& c, const double& aperture)
{   
    double longitude = x_dst_norm * CV_PI + CV_PI / 2;
    double latitude = y_dst_norm * CV_PI / 2;

    double p_x = -cos(latitude) * cos(longitude);
    double p_y = cos(latitude) * sin(longitude);
    double p_z = sin(latitude);

    cv::Vec3d p = cv::Vec3d(p_x, p_y, p_z);
    cv::Mat rot = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
    cv::Mat pt = rot * p;
    
    double p_xz = sqrt(pow(pt.at<double>(0,0), 2)+pow(pt.at<double>(0,2), 2)); // cv::cuda::pow uses only float (imprecision)
    double theta = atan2(p_xz,pt.at<double>(0,1));
    double phi = atan2(pt.at<double>(0,2), pt.at<double>(0,0));

    double x_src_norm = cos(phi);
    double y_src_norm = sin(phi);

    double x_src = 2 * f[0] * x_src_norm * sin(theta / 2);
    double y_src = 2 * f[1] * y_src_norm * sin(theta / 2);

    return {x_src+c[0], y_src+c[1]};
}

void mGenerateLUT(InputArray img, const cv::Vec2d& f, const cv::Vec2d& c, const cv::Size& newsize, const double& aperture, Mat& equirect)
{
    Mat img_ = img.getMat();
    Size viewSize = img_.size();
    mlut::mLUT lut;

    for(int y = 0; y < newsize.height / 2 + 10; y++) // 10 for margin
    {
        double y_dst_norm = lerp(-1,1,0,newsize.height, y);

        for(int x=0; x< newsize.width; x++)
        {
            double x_dst_norm = lerp(-1,1,0,newsize.width, x);
            
            Vec2d uv = rad2vec(x_dst_norm, y_dst_norm, f, c, aperture);
            int tx = min(viewSize.width - 1, static_cast<int>(floor(uv[0])));
            int ty = min(viewSize.height - 1, static_cast<int>(floor(uv[1])));

            double a = uv[0] - tx;
            double b = uv[1] - ty;

            if(tx >= 0 && tx < viewSize.width -1 && ty >=0 && ty < viewSize.height -1)
            {
                double bl = (1.-a)*(1.-b);
                double tl = (1.-a)*(b);
                double br = (a)*(1.-b);
                double tr = (a)*(b);

                mlut::mappingData* md = new mlut::mappingData(x, y, tx, ty, bl, tl, br, tr);
                lut.append(md);
            }
        }
    }
    save(lut, "lut.tab");
}

int main(int argc, char** argv)
{
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    std::string filename = argv[1];
    std::cout << "filename :" << filename << std::endl;

    Mat srcimg = cv::imread(filename);
    Size viewSize = srcimg.size();
    Size newsize(1024,512);
    Mat equirect = cv::Mat::zeros(newsize, srcimg.type());
    
    double aperture = 180. * CV_PI / 180.;
    double cx = 799.5;
    double cy = 806.5;
    double fx = 526.1783;
    double fy = 526.1783;

    Vec2d f(fx, fy);
    Vec2d c(cx, cy);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 
    mGenerateLUT(srcimg, f, c, newsize, aperture, equirect);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << endl;
    
    return 0;
}