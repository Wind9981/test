#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <experimental/filesystem>


#include <map>
#include <string>
#include "UltraFace/UltraFace.hpp"

#include <math.h>

using namespace dlib;
using namespace std;
namespace fs = std::experimental::filesystem;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

int main()
{

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }

    cv::Mat img;

    frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("/home/nhan/data/shape_predictor_68_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("/home/nhan/data/dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::map<std::string, dlib::matrix<float, 0, 1>> data_faces;
    if (!fs::exists("data_faces.dat"))
    {
        serialize("data_faces.dat") << data_faces;
    }
    else
    {
        deserialize("data_faces.dat") >> data_faces;
    }

    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;

    string name;
    cout << "enter name: ";
    cin >> name;
    int cout_img = 0;
    std::vector<matrix<rgb_pixel>> array_face;
    while (true)
    {

        if (!cap.read(img))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }
	cv::resize(img,img,cv::Size(800,480));
        //cap.read(img);
        //if(skip)
        //    continue;

        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        cv_image<bgr_pixel> cimg(img);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        //faces.clear();

        win.clear_overlay();
        for (int i = 0; i < face_info.size(); i++)
        {
            auto face = face_info[i];
            rectangle rect(point(face.x1, face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(0, 0, 255), name);
            auto shape = sp(matrix, rect);
            //double left_eye_ratio = get_blinking_ratio(l_eye_poits, shape);
            //double right_eye_ratio = get_blinking_ratio(r_eye_points, shape);

            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
            // Also put some boxes on the faces so we can see that the detector is finding
            // them.
            win.add_overlay(orect);
        }
	win.set_image(matrix);

        if (faces.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            continue;
        }

        cout_img++;
        cout << cout_img << endl;
        if (cout_img == 300)
            break;
        array_face.push_back(faces[0]);
    }
    data_faces.erase(name);
    data_faces.insert(std::pair<std::string, dlib::matrix<float, 0, 1>>(name, mean(mat(net(array_face)))));
    cout << data_faces[name];
    serialize("data_faces.dat") << data_faces;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

