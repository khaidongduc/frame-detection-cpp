#ifndef FRAME_DETECTION_CPP_FRAME_DETECTOR_H
#define FRAME_DETECTION_CPP_FRAME_DETECTOR_H

#endif //FRAME_DETECTION_CPP_FRAME_DETECTOR_H

#include "opencv2/opencv.hpp"

using namespace cv;

class FrameDetector{
public:
    FrameDetector();
    void load_query(const Mat &query_img);
    bool detect(const Mat &img);
};