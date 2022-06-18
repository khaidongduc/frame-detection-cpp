#ifndef FRAME_DETECTION_CPP_FRAME_DETECTOR_H
#define FRAME_DETECTION_CPP_FRAME_DETECTOR_H

#endif //FRAME_DETECTION_CPP_FRAME_DETECTOR_H

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;

class FrameDetector{
private:
    Ptr<Feature2D> orb;
    Mat query_img;
    std::vector<KeyPoint> kp1;
    Mat des1;

public:
    FrameDetector();
    void load_query(const Mat &query_img);
    bool detect(const Mat &img);
};