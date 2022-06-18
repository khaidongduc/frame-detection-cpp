#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const int DEFAULT_WIDTH = 500;

void resize(Mat &img, Mat &out, int height, int width, int inter) {
    cv::Size dim;
    cv::Size size = img.size();
    if (height == 0 and width == 0)
        return;
    if (width == 0) {
        float r = (float) height / (float) size.height;
        dim = cv::Size(int(float(size.width) * r), height);
    } else {
        float r = (float) width / (float) size.width;
        dim = cv::Size(width, int(float(size.height) * r));
    }
    cv::resize(img, out, dim, 0, 0, inter);
}


int main() {
    VideoCapture cap("/home/khaidong/CLionProjects/frame-detection-cpp/samples/sample vid.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file";
    }
    Mat frame, distorted_frame;
    distorted_frame = imread("/home/khaidong/CLionProjects/frame-detection-cpp/samples/capture.jpg", IMREAD_GRAYSCALE);
    resize(distorted_frame, distorted_frame, 0, DEFAULT_WIDTH, INTER_LINEAR);

    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame, COLOR_BGR2GRAY); // convert to grayscale
        resize(frame, frame, 0, DEFAULT_WIDTH, INTER_LINEAR);
        imshow("frame", frame);
        char c = (char) waitKey(25);
        if (c == 27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
