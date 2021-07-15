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
#include <time.h>
#include <opencv4/opencv2/ccalib.hpp>
#include <opencv4/opencv2/ccalib/omnidir.hpp>
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

struct CalibSettings
{
    int getFlag()
    {
        int flag = 0;
        flag |= omnidir::CALIB_USE_GUESS;
        flag |= omnidir::CALIB_FIX_SKEW;
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

double FocalLength(InputArray K, InputArray D, InputArray xi, cv::Size mSize) {

    double fx;
    double fy;
    double cx;
    double cy;
    double s;

    double k1;
    double k2;
    double p1;
    double p2;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        fx = camMat(0,0);
        fy = camMat(1,1);
        cx = camMat(0,2);
        cy = camMat(1,2);
        s  = camMat(0,1);
    }
    else
    {
        Matx33d camMat = K.getMat();
        fx = camMat(0,0);
        fy = camMat(1,1);
        cx = camMat(0,2);
        cy = camMat(1,2);
        s  = camMat(0,1);
    }

    Vec4d kp = Vec4d::all(0);
    if (!D.empty())
        kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();
    
    k1 = kp[0];
    k2 = kp[1];
    p1 = kp[2];
    p2 = kp[3];

    double _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();

    double u = cx;
    double v = 5.;
    double x = (u * fy - cx * fy - s * (v - cy)) / (fx * fy);
    double y = (v - cy) / fy;
    double e = x;
    double f = y;

    for (int i = 0; i < 20; ++i) {
        double r2 = e * e + f * f;
        double r4 = r2 * r2;
        double rr = 1. + k1 * r2 + k2 * r4;
        e = (x - 2. * p1 * e * f - p2 * (r2 + 2 * e * e)) / rr;
        f = (y - 2. * p2 * e * f - p1 * (r2 + 2 * f * f)) / rr;
    }

    u = cx;
    v = mSize.width - 5.;
    x = (u * fy - cx * fy - s * (v - cy)) / (fx * fy);
    y = (v - cy) / fy;
    double e0 = x;
    double f0 = y;

    for (int i = 0; i < 20; ++i) {
        double r2 = e0 * e0 + f0 * f0;
        double r4 = r2 * r2;
        double rr = 1. + k1 * r2 + k2 * r4;
        e0 = (x - 2. * p1 * e0 * f0 - p2 * (r2 + 2 * e0 * e0)) / rr;
        f0 = (y - 2. * p2 * e0 * f0 - p1 * (r2 + 2 * f0 * f0)) / rr;
    }

    if (fabs(f) > f0) {
        e = e0;
        f = -f0;
    }

    double ef = e * e + f * f;
    double zx = (_xi + sqrt(1. + (1. - _xi*_xi) * ef)) / (ef + 1.);
    cv::Vec3d Xc(e * zx, f * zx, zx - _xi);
    // double zx = (sqrt(1. + ef)) / (ef + 1.);
    // cv::Vec3d Xc(e * zx, f * zx, zx);
    Xc /= norm(Xc);

    f = Xc(1) / (Xc(2) + _xi);
    // f = Xc(1) / Xc(2);
    return - mSize.height / 2. / f;
}

enum{
        RECTIFY_PERSPECTIVE         = 1,
        RECTIFY_CYLINDRICAL         = 2,
        RECTIFY_LONGLATI            = 3,
        RECTIFY_STEREOGRAPHIC       = 4,
        RECTIFY_FISHEYE             = 5
    };

// void mProjectPoints(InputArray objectPoints, OutputArray imagePoints,
//                 InputArray rvec, InputArray tvec, InputArray K, double xi, InputArray D)
// {
//     CV_Assert(objectPoints.type() == CV_64FC3 || objectPoints.type() == CV_32FC3);
//     CV_Assert((rvec.depth() == CV_64F || rvec.depth() == CV_32F) && rvec.total() == 3); 
//     CV_Assert((tvec.depth() == CV_64F || tvec.depth() == CV_32F) && tvec.total() == 3);
//     CV_Assert((K.type() == CV_64F || K.type() == CV_32F) && K.size() == Size(3,3));
//     CV_Assert((D.type() == CV_64F || D.type() == CV_32F) && D.total() == 4);

//     imagePoints.create(objectPoints.size(), CV_MAKETYPE(objectPoints.depth(), 2));

//     int n = (int)objectPoints.total();

//     Vec3d om = rvec.depth() == CV_32F ? (Vec3d)*rvec.getMat().ptr<Vec3f>() : *rvec.getMat().ptr<Vec3d>();
//     Vec3d T  = tvec.depth() == CV_32F ? (Vec3d)*tvec.getMat().ptr<Vec3f>() : *tvec.getMat().ptr<Vec3d>();
    
//     Vec2d f,c;
//     double s;
//     if (K.depth() == CV_32F)
//     {
//         Matx33f Kc = K.getMat();
//         f = Vec2f(Kc(0,0), Kc(1,1));
//         c = Vec2f(Kc(0,2),Kc(1,2));
//         s = (double)Kc(0,1);
//     }
//     else
//     {
//         Matx33d Kc = K.getMat();
//         f = Vec2d(Kc(0,0), Kc(1,1));
//         c = Vec2d(Kc(0,2),Kc(1,2));
//         s = Kc(0,1);
//     }

//     Vec4d kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>() : *D.getMat().ptr<Vec4d>();
//     //Vec<double, 4> kp= (Vec<double,4>)*D.getMat().ptr<Vec<double,4> >();

//     const Vec3d* Xw_alld = objectPoints.getMat().ptr<Vec3d>();
//     const Vec3f* Xw_allf = objectPoints.getMat().ptr<Vec3f>();
//     Vec2d* xpd = imagePoints.getMat().ptr<Vec2d>();
//     Vec2f* xpf = imagePoints.getMat().ptr<Vec2f>();

//     // cv::Matx33d R  = cv::Matx33d::eye();
//     // if (!rvec.empty() && rvec.total() * rvec.channels() == 3)
//     // {
//     //     cv::Vec3d _rvec;
//     //     rvec.getMat().convertTo(_rvec, CV_64F);
//     //     cv::Rodrigues(_rvec, R);
//     // }
//     // else if (!rvec.empty() && rvec.size() == Size(3, 3))
//     //     rvec.getMat().convertTo(R, CV_64F);

//     // cv::Matx33d PP = cv::Matx33d::eye();
//     // if (!tvec.empty())
//     //     tvec.getMat().Range(0, 3).convertTo(PP, CV_64F);
//     // else
//     //     PP = K.getMat();

//     // cv::Matx33d iKR = (PP*RR).inv(cv::DECOMP_SVD);
//     // cv::Matx33d iK = PP.inv(cv::DECOMP_SVD);
//     // cv::Matx33d iR = RR.inv(cv::DECOMP_SVD);

//     Matx33d R;
//     Matx<double, 3, 9> dRdom;
//     Rodrigues(om, R, dRdom);

//     double k1=kp[0],k2=kp[1];
//     double p1 = kp[2], p2 = kp[3];

//     for (int i = 0; i < n; i++)
//     {
//         // convert to camera coordinate
//         Vec3d Xw = objectPoints.depth() == CV_32F ? (Vec3d)Xw_allf[i] : Xw_alld[i];

//         Vec3d Xc = (Vec3d)(R*Xw + T);

//         // convert to unit sphere
//         Vec3d Xs = Xc/cv::norm(Xc);

//         // convert to normalized image plane
//         Vec2d xu = Vec2d(Xs[0]/(Xs[2]+xi), Xs[1]/(Xs[2]+xi));

//         // add distortion
//         Vec2d xd;
//         double r2 = xu[0]*xu[0]+xu[1]*xu[1];
//         double r4 = r2*r2;

//         xd[0] = xu[0]*(1+k1*r2+k2*r4) + 2*p1*xu[0]*xu[1] + p2*(r2+2*xu[0]*xu[0]);
//         xd[1] = xu[1]*(1+k1*r2+k2*r4) + p1*(r2+2*xu[1]*xu[1]) + 2*p2*xu[0]*xu[1];

//         // convert to pixel coordinate
//         Vec2d final;
//         final[0] = f[0]*xd[0]+s*xd[1]+c[0];
//         final[1] = f[1]*xd[1]+c[1];

//         if (objectPoints.depth() == CV_32F)
//         {
//             xpf[i] = final;
//         }
//         else
//         {
//             xpd[i] = final;
//         }
//         /*xpd[i][0] = f[0]*xd[0]+s*xd[1]+c[0];
//         xpd[i][1] = f[1]*xd[1]+c[1];*/
//     }
// }

void mInitUndistortRectifyMap(InputArray K, InputArray D, InputArray xi, InputArray R, InputArray P,
    const cv::Size& size, int m1type, OutputArray map1, OutputArray map2, int flags)
{
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32F || m1type <=0 );
    map1.create( size, m1type <= 0 ? CV_16SC2 : m1type );
    map2.create( size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F );

    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert(K.size() == Size(3, 3) && (D.empty() || D.total() == 4));
    CV_Assert(P.empty()|| (P.depth() == CV_32F || P.depth() == CV_64F));
    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
    CV_Assert(R.empty() || (R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(flags == RECTIFY_PERSPECTIVE || flags == RECTIFY_CYLINDRICAL || flags == RECTIFY_LONGLATI
        || flags == RECTIFY_STEREOGRAPHIC || flags == RECTIFY_FISHEYE);
    CV_Assert(xi.total() == 1 && (xi.depth() == CV_32F || xi.depth() == CV_64F));

    cv::Vec2d f, c;
    double s;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
        s = (double)camMat(0,1);
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
        s = camMat(0,1);
    }

    Vec4d kp = Vec4d::all(0);
    if (!D.empty())
        kp = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();
    double _xi;
    if(!xi.empty())
        _xi = xi.depth() == CV_32F ? (double)*xi.getMat().ptr<float>() : *xi.getMat().ptr<double>();
    Vec2d k = Vec2d(kp[0], kp[1]);
    Vec2d p = Vec2d(kp[2], kp[3]);
    cv::Matx33d RR  = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        cv::Rodrigues(rvec, RR);
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    cv::Matx33d PP = cv::Matx33d::eye();
    if (!P.empty())
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
    else
        PP = K.getMat();

    cv::Matx33d iKR = (PP*RR).inv(cv::DECOMP_SVD);
    cv::Matx33d iK = PP.inv(cv::DECOMP_SVD);
    cv::Matx33d iR = RR.inv(cv::DECOMP_SVD);

    if (flags == RECTIFY_PERSPECTIVE)
    {
        for (int i = 0; i < size.height; ++i)
        {
            float* m1f = map1.getMat().ptr<float>(i);
            float* m2f = map2.getMat().ptr<float>(i);
            short*  m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;

            double _x = i*iKR(0, 1) + iKR(0, 2),
                   _y = i*iKR(1, 1) + iKR(1, 2),
                   _w = i*iKR(2, 1) + iKR(2, 2);

            for(int j = 0; j < size.width; ++j, _x+=iKR(0,0), _y+=iKR(1,0), _w+=iKR(2,0))
            {
                // project back to unit sphere
                double r = sqrt(_x*_x + _y*_y + _w*_w);
                double Xs = _x / r;
                double Ys = _y / r;
                double Zs = _w / r;
                // project to image plane
                double xu = Xs / (Zs + _xi),
                    yu = Ys / (Zs + _xi);
                // add distortion
                double r2 = xu*xu + yu*yu;
                double r4 = r2*r2;
                double xd = (1+k[0]*r2+k[1]*r4)*xu + 2*p[0]*xu*yu + p[1]*(r2+2*xu*xu);
                double yd = (1+k[0]*r2+k[1]*r4)*yu + p[0]*(r2+2*yu*yu) + 2*p[1]*xu*yu;
                // to image pixel
                double u = f[0]*xd + s*yd + c[0];
                double v = f[1]*yd + c[1];

                if( m1type == CV_16SC2 )
                {
                    int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                    int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                    m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                    m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                    m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
                }
                else if( m1type == CV_32FC1 )
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }
            }
        }
    }
    else if(flags == RECTIFY_CYLINDRICAL || flags == RECTIFY_LONGLATI ||
        flags == RECTIFY_STEREOGRAPHIC || flags == RECTIFY_FISHEYE)
    {
        for (int i = 0; i < size.height; ++i)
        {
            float* m1f = map1.getMat().ptr<float>(i);
            float* m2f = map2.getMat().ptr<float>(i);
            short*  m1 = (short*)m1f;
            ushort* m2 = (ushort*)m2f;

            // for RECTIFY_LONGLATI, theta and h are longittude and latitude
            double theta = i*iK(0, 1) + iK(0, 2),
                   h     = i*iK(1, 1) + iK(1, 2);

            for (int j = 0; j < size.width; ++j, theta+=iK(0,0), h+=iK(1,0))
            {
                double _xt = 0.0, _yt = 0.0, _wt = 0.0;
                if (flags == RECTIFY_CYLINDRICAL)
                {
                    _xt = std::cos(theta);
                    _yt = std::sin(theta);
                    _wt = h;
                }
                else if (flags == RECTIFY_LONGLATI)
                {
                    _xt = -std::cos(theta);
                    _yt = -std::sin(theta) * std::cos(h);
                    _wt = std::sin(theta) * std::sin(h);
                }
                else if (flags == RECTIFY_STEREOGRAPHIC)
                {
                    double a = theta*theta + h*h + 4;
                    double b = -2*theta*theta - 2*h*h;
                    double c2 = theta*theta + h*h -4;

                    _yt = (-b-std::sqrt(b*b - 4*a*c2))/(2*a);
                    _xt = theta*(1 - _yt) / 2;
                    _wt = h*(1 - _yt) / 2;
                }
                else if (flags == RECTIFY_FISHEYE){
                    // iK(r,0) * j + iK(r,1) * i + iK(r,2) * 1.;   r == 0 || 1  , j : x axis, i : y axis
                    double ee = MatRowMul(iK, j, i, 1., 0); 
                    double ff = MatRowMul(iK, j, i, 1., 1);

                    double ef = ee * ee + ff * ff;
                    double zz = (_xi + sqrt(1. + (1. - _xi * _xi) * ef)) / (ef + 1.);
                    // double zz = (sqrt(1. + ef)) / (ef + 1.);

                    _xt = zz * ee;
                    _yt = zz * ff;
                    _wt = zz - _xi;
                    // _wt = zz;
                }
                double _x = iR(0,0)*_xt + iR(0,1)*_yt + iR(0,2)*_wt;
                double _y = iR(1,0)*_xt + iR(1,1)*_yt + iR(1,2)*_wt;
                double _w = iR(2,0)*_xt + iR(2,1)*_yt + iR(2,2)*_wt;

                double r = sqrt(_x*_x + _y*_y + _w*_w);
                double Xs = _x / r;
                double Ys = _y / r;
                double Zs = _w / r;
                // project to image plane
                double xu = Xs / (Zs + _xi),
                       yu = Ys / (Zs + _xi);
                // double xu = Xs / (Zs),
                //        yu = Ys / (Zs);
                // add distortion
                double r2 = xu*xu + yu*yu;
                double r4 = r2*r2;
                double xd = (1+k[0]*r2+k[1]*r4)*xu + 2*p[0]*xu*yu + p[1]*(r2+2*xu*xu);
                double yd = (1+k[0]*r2+k[1]*r4)*yu + p[0]*(r2+2*yu*yu) + 2*p[1]*xu*yu;
                // to image pixel
                double u = f[0]*xd + s*yd + c[0];
                double v = f[1]*yd + c[1];

                if( m1type == CV_16SC2 )
                {
                    // get rid of the floating number (to integer (short))
                    int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                    int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                    m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                    m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                    m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
                }
                else if( m1type == CV_32FC1 )
                {
                    m1f[j] = (float)u;
                    m2f[j] = (float)v;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    srand(time(0));

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

    double rms = omnidir::calibrate(objectPoints, imagePoints, imageSize,cameraMatrix, xi, distCoeffs, rvec, tvec, s.getFlag(), TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, objectPoints.size(), 1e-10), mIdx);
    
    cout << "-------------mean Reprojection error--------------" << endl;
    cout << rms << endl;
    cout << "-------------cameraMatrix--------------" << endl;
    
    // camera matrix from Matlab (Scramuzza calibration method)
    // cout << cameraMatrix << endl;
    // cameraMatrix.at<double>(0,0) = 5.26178278235e+02;
    // cameraMatrix.at<double>(1,1) = 5.26178278235e+02;
    // cameraMatrix.at<double>(0,2) = 7.991734504717965e+02;
    // cameraMatrix.at<double>(1,2) = 8.032112360941801e+02; 
    
    cout << cameraMatrix << endl;

    cout << "---------------distCoeffs--------------" << endl;
    // distortion coefs from Matlab (Scramuzza calibration method)
    // cout << distCoeffs << endl;
    // distCoeffs.at<double>(0,0) = -8.832417165240590e-04;
    // distCoeffs.at<double>(0,1) = 8.069012047835724e-07;
    // // Divide Radial distortion into focal length
    // distCoeffs.at<double>(0,0) /= cameraMatrix.at<double>(0,0);
    // distCoeffs.at<double>(0,1) /= cameraMatrix.at<double>(0,0);
    // // assumes no Tangential distortion
    // distCoeffs.at<double>(0,2) = 0.;
    // distCoeffs.at<double>(0,3) = 0.;
    cout << distCoeffs << endl;
    cout << "---------------xi--------------" << endl;
    cout << xi << endl;

    // for(int i=0; i<viewlist.size(); i++)
    // {
    //     mdrawingCircle(viewlist[i], rvec[i], tvec[i], cameraMatrix, distCoeffs);
    //     imshow("test", viewlist[i]);
    //     waitKey(0);
    //     cout << viewlistname[i] << endl;
    //     string name_ = "./dots/"+viewlistname[i];
    //     imwrite(name_, viewlist[i]);
    // }

    // Recompute Reprojection Error with the resulting intrinsic parameter of the Scaramuzza method
    // vector<vector<Point2f>> rep_imgPoints;
	
    // double tot_error=0, err{0};
    // int total_points=0;

    // double _xi = xi.at<double>(0,0);

    // for(int loc= 0; loc< mIdx.total(); loc++)
    // {   
    //     int i = mIdx.at<int>(0,loc);
        
    //     cout << "i is " << i << ", loc is " << loc << endl;

    //     vector <Point2f> rep_corner_pts;

    //     omnidir::projectPoints(cv::Mat(objectPoints[i]), rep_corner_pts, rvec[loc], tvec[loc], cameraMatrix, xi, distCoeffs);
        
    //     rep_imgPoints.push_back(rep_corner_pts);
    //     err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(rep_imgPoints[loc]), cv::NORM_L2);

    //     int n = static_cast<int>(objectPoints[i].size());

    //     tot_error += err*err;
    //     total_points+= n;
    // }
    
    // tot_error = sqrt(tot_error/ total_points);
    // cout << "-------------Mean Reprojection error --------------" << endl;
    // cout << tot_error << endl;

    int idx = 0;
    for (auto image_name : imagesName) {
        
        string imgpath = image_name.c_str();
        Mat view = imread(imgpath);

        // Mat pano = Mat::zeros(Size(1024,512), view.type());
        // Size new_size =  pano.size();
        // Matx33f Knew = Matx33f(new_size.width/CV_PI, 0, 0, 0, new_size.height/CV_PI, 0, 0, 0, 1);
        
        Mat pano = Mat::zeros(view.size(), view.type());
        Mat view2 = pano.clone();
        Size new_size =  view.size();
        //for fisheye correction
        double focal_len = FocalLength(cameraMatrix, distCoeffs, xi, new_size);
        Matx33f Knew = Matx33f(focal_len, 0, new_size.width /2 -0.5 , 0, focal_len, new_size.height/2 -0.5, 0, 0, 1);
        //for rectilinear mapping
        // Matx33f Knew = Matx33f(new_size.width/4, 0, new_size.width /2, 0, new_size.height/4, new_size.height/2, 0, 0, 1);
        
        cout << "index " << idx << endl;
        cout<<Knew<<endl;
        cout<< view.type()<< endl;
        cv::Mat map1, map2, R;
        
        mInitUndistortRectifyMap(cameraMatrix, distCoeffs, xi, R, Knew, new_size, CV_16SC2, map1, map2, RECTIFY_FISHEYE);
        
        idx++;
        auto start_time = clock();
        cv::remap(view, pano, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
        auto end_time = clock();
        cout << "time in While  " << 1000.000*(end_time - start_time) / CLOCKS_PER_SEC << endl<< endl;
        
        namedWindow("undist", cv::WINDOW_NORMAL);
        cv::resize(pano, view2, cv::Size(new_size.width/2, new_size.height/2), 0.5, 0.5);
        imshow("undist", view2);
        waitKey(0);
        
        std::regex re("IMG_\\d{3,4}");
        std::smatch match;
        string imgname;
        if(std::regex_search(imgpath, match, re))
        {
            for (size_t i=0; i<match.size(); i++)
            {
                imgname = match[i].str();
            }
        }
        string name = "./correction/" + imgname + "_cor.jpg";
        cout << "written in " << name << endl;
        cv::imwrite(name, pano);
    } 
     
    return 0;
}