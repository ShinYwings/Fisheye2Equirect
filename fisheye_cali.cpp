// fisheye correction refers from
// https://github.com/astar-ai/calicam_mono/blob/master/calicam_mono.cpp
//proportional to latitude.  (equisolid -> equidistant)
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <regex>
#include <dirent.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <string>
#include <chrono>
#include <cmath>

const int BOARDWIDTH = 7;
const int BOARDHEIGHT = 10;

float SQUARESIZE = 25;

using namespace std;
using namespace cv;

vector<string> viewlistname;

inline double MatRowMul(cv::Matx33d m, double x, double y, double z, int r) {
  return m(r,0) * x + m(r,1) * y + m(r,2) * z;
}

inline double lerp(double y0, double y1, double x0, double x1, double x)
{
    double m = (y1-y0) / (x1-x0);
    double b = y0;
    return m * (x-x0) + b;
}

class CalibSettings
{
    public:
    int getFlag()
    {
        int flag = 0;
        flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
        flag |= cv::fisheye::CALIB_CHECK_COND;
        flag |= cv::fisheye::CALIB_FIX_SKEW;
        return flag;
    }

    Size getBoardSize()
    {
        return Size(BOARDWIDTH, BOARDHEIGHT);
    }

    float getSquareSize()
    {
        return SQUARESIZE;
    }
};

CalibSettings s;
static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
}

vector<string> getImageList(string path)
{
    vector<string> imagesName;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string tmpFileName = ent->d_name;
            if (tmpFileName.length() > 4) {
                auto nPos = tmpFileName.find(".jpg");
                if (nPos != string::npos) {
                    imagesName.push_back(path + '/' + tmpFileName);
                    viewlistname.push_back(tmpFileName);
                } else {
                    nPos = tmpFileName.find(".JPG");
                    if (nPos != string::npos)
                        imagesName.push_back(path + '/' + tmpFileName);
                        viewlistname.push_back(tmpFileName);
                }
            }
        }
        closedir(dir);
    }
    return imagesName;
}

cv::Vec2d rad2vec(double r, double theta, double phi, Vec2d f, Vec2d c, Vec4d kp)
{
    double fx = f[0], fy = f[1];
    double cx = c[0], cy = c[1];
    
    double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
        theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

    double theta_d = theta + kp[0]*theta3 + kp[1]*theta5 + kp[2]*theta7 + kp[3]*theta9;

    double inv_r = r > 1e-8 ? 1.0/r : 1;
    double cdist = r > 1e-8 ? theta_d * inv_r : 1;

    double x_src_norm = r * cos(phi);
    double y_src_norm = r * sin(phi);

    double x_src = cdist * x_src_norm;
    double y_src = cdist * y_src_norm;

    return cv::Vec2d(fx*x_src+cx, fy*y_src+cy);
} 

void fisheye2Equirect(InputArray img, InputArray K, InputArray D, const cv::Size& newsize, const double& aperture, Mat& equirect)
{
    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert(K.size() == Size(3, 3) && (D.empty() || D.total() == 4));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
    }

    Mat img_ = img.getMat();
    Size viewSize = img_.size();

    Vec4d kp = Vec4d::all(0);
    if (!D.empty())
        kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    Vec3d p;

    for(int y=0; y< newsize.height; y++)
    {
        double y_dst_norm = lerp(-1,1,0,newsize.height, y);

        for(int x=0; x< newsize.width; x++)
        {
            double x_dst_norm = lerp(-1,1,0, newsize.width, x);
            
            double longitude = x_dst_norm * CV_PI + CV_PI / 2;
            double latitude = y_dst_norm * CV_PI / 2;

            double p_x = - cos(latitude) * cos(longitude);
            double p_y = cos(latitude) * sin(longitude);
            double p_z = sin(latitude);
    
            p = Vec3d(p_x, p_y, p_z);
            cv::Mat rot = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
            cv::Mat pt = rot * p;
            double p_xz = sqrt(pow(pt.at<double>(0,0), 2)+pow(pt.at<double>(0,2), 2));
            double theta = atan2(p_xz,pt.at<double>(0,1));
            double r = ((2 * theta) / aperture);
            double phi = atan2(pt.at<double>(0,2), pt.at<double>(0,0));

            Vec2d uv = rad2vec(r, theta, phi, f, c, kp);

            if( y > newsize.height / 2 -1 )
            {
                continue;
            }
            else if(y == 255)
            {
                equirect.at<Vec3b>(y,x) = Vec3b(255, 0., 0.);
            }
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

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <pic path> [square size(mm)]" << endl;
        return 0;
    }
    if (argc == 3) {
        std::string size;
        size = argv[2];
        SQUARESIZE = std::stof(size);
    }
    string pathDirectory = argv[1];
    auto imagesName = getImageList(pathDirectory);
    
    vector<vector<Point2f>> imagePoints;
    Size imageSize;
    vector<vector<Point3f>> objectPoints;
    vector<Mat> viewlist;
    for (auto image_name : imagesName) {
        Mat view;
        view = imread(image_name.c_str());

        imageSize = view.size();
        vector<Point2f> pointBuf;
        // find the corners
        bool found = findChessboardCorners(view, s.getBoardSize(), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_ACCURACY |CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            Mat viewGray;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix(viewGray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT + TermCriteria::MAX_ITER, imagesName.size(), 1e-10));
            imagePoints.push_back(pointBuf);
            drawChessboardCorners(view, s.getBoardSize(), Mat(pointBuf), found);
            cout << image_name << endl;
            viewlist.push_back(view);
            vector<Point3f> obj;
            calcBoardCornerPositions(s.getBoardSize(), s.getSquareSize(), obj);
            objectPoints.push_back(obj);
        } else {
            cout << image_name << " found corner failed! & removed!" << endl;
        }
    }

    cv::Mat cameraMatrix, xi, distCoeffs;

    vector<Mat> rvec, tvec;
    
    Mat mIdx;

    cout << "-------------imageSize--------------" << endl;
    cout << imageSize << endl;

    double rms = fisheye::calibrate(objectPoints, imagePoints, imageSize,cameraMatrix, distCoeffs, rvec, tvec, s.getFlag(), TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, objectPoints.size(), 1e-10));
    
    cout << "-------------mean Reprojection error--------------" << endl;
    cout << rms << endl;
    cout << "-------------cameraMatrix--------------" << endl;
    cout << cameraMatrix << endl;
    cout << "---------------distCoeffs--------------" << endl;
    cout << distCoeffs << endl;
    
    Size newsize(1024,512);
    double aperture = 180. * CV_PI / 180.;
    
    Mat srcimg = cv::imread("ex2.JPG");
    Mat equirect = cv::Mat::zeros(newsize, srcimg.type());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 
    fisheye2Equirect(srcimg, cameraMatrix, distCoeffs, newsize, aperture, equirect);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << "elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << endl;
    imshow("test", equirect);
    imwrite("equirect2.jpg", equirect);
    waitKey(0);
    
    return 0;
}