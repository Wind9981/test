#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(void) {
    //class represent for capturing video from camera and reading video file or image sequences
    VideoCapture videoCapture;
    Mat videoFrame;
    const std::string gst_pipeline="v4l2src ! image/jpeg, width = 1280, height = 720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
    //open camera
    videoCapture.open(gst_pipeline ,cv::CAP_GSTREAMER);
    namedWindow("VideoCapture", WINDOW_AUTOSIZE);
    //check open camera open sucessed or failed
    if(!videoCapture.isOpened()){
        cout << "Can't open camera" << endl;
    }else{
        while (true) {
            std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
            //read video frame from camera and show in windows
            videoCapture.read(videoFrame);
            imshow("VideoCapture", videoFrame);
            cout << " Rec delay: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count() << endl;
            if(waitKey(27) >= 0) break;
        }
    }
    return 0;
}
