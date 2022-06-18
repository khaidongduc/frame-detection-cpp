#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const int DEFAULT_WIDTH = 500;

int main() {

    VideoCapture cap("/home/khaidong/CLionProjects/frame-detection-cpp/samples/sample vid.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file";
    }
    Mat frame;
    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame, COLOR_BGR2GRAY); // convert to grayscale
        imshow("frame", frame);
        char c = (char) waitKey(25);
        if (c == 27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
