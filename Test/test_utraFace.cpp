#include <opencv2/opencv.hpp>
/*
#include <unistd.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
*/
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/sqlite.h>
#include <map>
#include <string>
#include "UltraFace/UltraFace.hpp"

#include <math.h>
using namespace dlib;
using namespace std;



template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;


using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;



int main()
{


 
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }

    shape_predictor sp;
    deserialize("/home/nhan/data/shape_predictor_68_face_landmarks.dat") >> sp;
    //int r_eye_points[] = {42, 43, 44, 45, 46, 47};
    //int l_eye_poits[] = {36, 37, 38, 39, 40, 41};
    // And finally we load the DNN responsible for face recognition.
    
    anet_type net;
    deserialize("/home/nhan/data/dlib_face_recognition_resnet_model_v1.dat") >> net;
    

    //Load know faces
    //std::map<std::string, dlib::matrix<float,0,1>> data_faces;   
    //deserialize("data_faces.dat") >> data_faces;
    
    //Template database

    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    std::cout << "Hit ESC to exit" << "\n" ;

    cv::Mat img;
    
    while(true)
    {
        

    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
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
        faces.clear();

        win.clear_overlay();
        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
            auto shape = sp(matrix,rect);
            //double left_eye_ratio = get_blinking_ratio(l_eye_poits, shape);
            //double right_eye_ratio = get_blinking_ratio(r_eye_points, shape);
           
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        
        
        win.set_image(matrix);
        //cout << "Detection delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;
        m_StartTime = std::chrono::system_clock::now();
        
        
        if (faces.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            continue;
        }
/*
	else
	{
            cout << "Have faces found in image!" << endl;
            continue;
        }
*/

        std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
        
        

        for (size_t i = 0; i < face_descriptors.size(); ++i)
        {
/*
            for(auto& x:data_faces )
            {
                //double tmp_distance = length(face_descriptors[i]-x.second );
                if (length(face_descriptors[i]-x.second ) < 0.5)
                {
                    
                    cout<<x.first<<": "<<length(face_descriptors[i]-x.second )<<endl;
                }
            }
*/
		cout<<": "<<length(face_descriptors[i])<<endl;
        }
        cout <<" Rec delay: "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count()<<endl;

    }
    cap.release();
    return 0;
}
