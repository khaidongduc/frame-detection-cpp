#include "../include/frame_detector.h"


#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

const float ratio_thresh = 0.9f;
const int min_match_count = 30;
const float homography_area_thresh = 0.4f;

FrameDetector::FrameDetector() {
    this->orb = ORB::create();
}


void FrameDetector::load_query(const Mat &query) {
    this->query_img = query;
    // extract key points and descriptor from query
    orb->detectAndCompute(query, Mat(), this->kp1, this->des1);
}


bool FrameDetector::detect(const Mat &img) {
    // extract key points and descriptor from img
    vector<KeyPoint> kp2;
    Mat des2;
    orb->detectAndCompute(img, Mat(), kp2, des2);

    // matching features
    BFMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(this->des1, des2, knn_matches, 2);
    vector<DMatch> good_matches;
    for (auto &knn_match: knn_matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }

    Mat img_matches;
    drawMatches(this->query_img, kp1, img, kp2, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    bool res = false;
    if (good_matches.size() >= min_match_count) {
        vector<Point2f> src_pts, dst_pts;
        for (auto &good_match: good_matches) {
            src_pts.push_back(this->kp1[good_match.queryIdx].pt);
            dst_pts.push_back(kp2[good_match.trainIdx].pt);
        }

        imshow("matches", img_matches);
        waitKey();
        res = true;
    }

    imshow("Good Matches", img_matches);
    return res;
}

