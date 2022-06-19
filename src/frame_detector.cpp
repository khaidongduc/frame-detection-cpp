#include "../include/frame_detector.h"
#include "../include/settings.h"

#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

const float ratio_thresh = 0.75f;
const int min_match_count = 30;
const float homography_area_thresh = 0.4f;

FrameDetector::FrameDetector() {
    this->sift = SIFT::create();
}


void FrameDetector::load_scene(const Mat &scene_query) {
    this->scene_img = scene_query;
    // extract key points and descriptor from scene_query
    sift->detectAndCompute(scene_query, Mat(), this->scene_keypoints, this->scene_descriptors);
}


bool FrameDetector::detect(const Mat &object_img) {
    // extract key points and descriptor from object_img
    vector<KeyPoint> object_keypoints;
    Mat object_descriptors;
    sift->detectAndCompute(object_img, Mat(), object_keypoints, object_descriptors);

    // matching features
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(object_descriptors, this->scene_descriptors, knn_matches, 2);
    vector<DMatch> good_matches;
    for (auto &knn_match: knn_matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }
    bool res = false;
    std::vector<Point2f> obj, scene, obj_corners(4), scene_corners(4);;
    if (good_matches.size() >= min_match_count) {
        for (auto &good_match: good_matches) {
            // Get the keypoints from the good matches
            obj.push_back(object_keypoints[good_match.queryIdx].pt);
            scene.push_back(this->scene_keypoints[good_match.trainIdx].pt);
        }
        Mat H = findHomography(obj, scene, RANSAC);
        // Get the corners from the image_1 ( the object to be "detected" )
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f((float) object_img.cols, 0);
        obj_corners[2] = Point2f((float) object_img.cols, (float) object_img.rows);
        obj_corners[3] = Point2f(0, (float) object_img.rows);
        perspectiveTransform(obj_corners, scene_corners, H);
        double object_area_ratio = contourArea(scene_corners) / (scene_img.rows * scene_img.cols);
        res = object_area_ratio >= homography_area_thresh;
    }

    if (DEBUG) {
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        Mat img_matches;
        drawMatches(object_img, object_keypoints, this->scene_img, this->scene_keypoints, good_matches, img_matches,
                    Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        line(img_matches, scene_corners[0] + Point2f((float) object_img.cols, 0),
             scene_corners[1] + Point2f((float) object_img.cols, 0), Scalar(0, 255, 0), 10);
        line(img_matches, scene_corners[1] + Point2f((float) object_img.cols, 0),
             scene_corners[2] + Point2f((float) object_img.cols, 0), Scalar(0, 255, 0), 10);
        line(img_matches, scene_corners[2] + Point2f((float) object_img.cols, 0),
             scene_corners[3] + Point2f((float) object_img.cols, 0), Scalar(0, 255, 0), 10);
        line(img_matches, scene_corners[3] + Point2f((float) object_img.cols, 0),
             scene_corners[0] + Point2f((float) object_img.cols, 0), Scalar(0, 255, 0), 10);
        imshow("Good Matches & Object detection", img_matches);
        waitKey();
    }

    return res;
}

