#pragma once

#include <iostream>
#include "opencv2/opencv.hpp"
#include "../include/settings.h"

using namespace cv;

class FrameDetector{
private:
    Ptr<Feature2D> sift;
    Mat scene_img;
    std::vector<KeyPoint> scene_keypoints;
    Mat scene_descriptors;

public:
    FrameDetector();
    void load_scene(const Mat &scene_query);
    bool detect(const Mat &object_img);
};