#include "UltraFace/UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using std::string;

int main(void)
{
    cv::VideoCapture cap;
    cv::Mat img;
    const std::string gst_pipeline="v4l2src ! image/jpeg, width = 1280, height = 720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
    cap.open(gst_pipeline ,cv::CAP_GSTREAMER);
    cv::namedWindow("Detect", cv::WINDOW_AUTOSIZE);
    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);
    
    std::cout << "Hit ESC to exit"
              << "\n";

    if(!cap.isOpened()){
        cout << "Can't open camera" << endl;
    }else{
        while (true) {
            std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
            //read video frame from camera and show in windows
            cap.read(img);
            //cv::imshow("Detect", img);
            //cv::resize(img,img,cv::Size(800,480));
            cv::Mat image_clone = img.clone();
            ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);

            std::vector<FaceInfo> face_info;
            ultraface.detect(inmat, face_info);

            for (int i = 0; i < face_info.size(); i++)
            {
                auto face = face_info[i];
                cv::rectangle(img,cv::Point(face.x1,face.y1),cv::Point(face.x2,face.y2),cv::Scalar(255,0,0),3);
                cv::imshow("Detect",img);
            }

            cout << " Rec delay: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count() << endl;
            if(cv::waitKey(27) >= 0) break;
        }
    }
/*
	//cap.read(img);
        //cv::resize(img,img,cv::Size(800,480));
	cv::imshow("Detect",img);
	cout<<"Debug"<<"\n";
/*
        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);


        for (int i = 0; i < face_info.size(); i++)
        {
            auto face = face_info[i];
	    cv::rectangle(image_clone,cv::Point(face.x1,face.y1),cv::Point(face.x2,face.y2),cv::Scalar(255,0,0),3);
            //rectangle rect(point(face.x1, face.y1), point(face.x2, face.y2));
            //image_window::overlay_rect orect(rect, rgb_pixel(255, 0, 0), "Unknow");
	    cv::imshow("Detect",image_clone);                              
        }
	cv::imshow("Detect",image_clone);
*/

        //cout << "Detection delay: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count() << endl;
    return 0;
}
