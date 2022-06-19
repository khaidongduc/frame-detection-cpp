#ifndef FRAME_DETECTION_CPP_FRAME_DETECTOR_H
#define FRAME_DETECTION_CPP_FRAME_DETECTOR_H

#endif //FRAME_DETECTION_CPP_FRAME_DETECTOR_H

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;

class FrameDetector{
private:
    Ptr<Feature2D> orb;
    Mat scene_img;
    std::vector<KeyPoint> scene_keypoints;
    Mat scene_descriptors;

public:
    FrameDetector();
    void load_scene(const Mat &scene_query);
    bool detect(const Mat &object_img);
};