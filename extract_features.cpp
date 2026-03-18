#include <opencv2/opencv.hpp> //Used for reading images
#include <dlib/opencv.h> //Used to convert images captured by OpenCV to Dlib images
#include <dlib/image_processing.h> //Used for facial landmark detection
#include <dlib/image_processing/frontal_face_detector.h> //Used to load the pre-trained frontal face detector file
#include <filesystem> //Used to read files and directories automatically
#include <fstream> //Used to read and write files
#include <vector> //Used to provide vector containers to store list of points 
using namespace std; //Used to avoid writing std::
using namespace cv; //Used to avoid writing cv::
namespace fs = std::filesystem; //Used to avoid writing std::filesystem

double dist(Point a, Point b){ //Used to calculate the Euclidean Distance between 2 points
    return norm(a - b);
}

double EAR(vector<Point> eye){ //Used to calculate Eye Aspect Ratio (EAR)
    return (dist(eye[1],eye[5]) + dist(eye[2],eye[4])) / (2.0 * dist(eye[0],eye[3]));
}

double MAR(vector<Point> mouth){ //Used to calculate Mouth Aspect Ratio (MAR)
    return (dist(mouth[2],mouth[10]) + dist(mouth[4],mouth[8])) / (2.0 * dist(mouth[0],mouth[6]));
}

int main(){
    string dataset = "train"; //The specified dataset directory
    ofstream file("features.csv"); //Creates a CSV file which will store the extracted features
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector(); //Creates a face detection model using Dlib
    dlib::shape_predictor predictor; //Creates an object to load the facial landmark detection model in 
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor; //Loads the facial landmark detection model

    for(auto &emotion_folder : fs::directory_iterator(dataset)){ //Loops through each emotion in the directory
        string emotion = emotion_folder.path().filename().string(); //Gets the folder's name
        int label; //Used to store the emotions with a numerical value

        if(emotion=="angry" || emotion=="fear" || emotion=="sad" || emotion=="disgust")
            label = 1;
        else
            label = 0;

        for(auto &img_path : fs::directory_iterator(emotion_folder)){ //Loops through each image within a folder
            Mat img = imread(img_path.path().string()); //Loads the image using OpenCV

            if(img.empty()) continue; //If the image fails to load, then it is skipped

            resize(img,img,Size(96,96)); //Resizes the image to 96 x 96
            dlib::cv_image<dlib::bgr_pixel> dimg(img); //Converts the OpenCV image into Dlib format
            auto faces = detector(dimg); //Used to detect faces within the image

            if(faces.empty()) continue; //If no face is detected, then the image is skipped

            auto shape = predictor(dimg,faces[0]); //Places the 68 facial landmark points for the detected face
            vector<Point> left_eye,right_eye,mouth; //Used to store the points on the left eye, right eye and mouth

            for(int i=36;i<=41;i++) //These points correspond to the left eye
                left_eye.push_back(Point(shape.part(i).x(),shape.part(i).y()));

            for(int i=42;i<=47;i++) //These points correspond to the right eye
                right_eye.push_back(Point(shape.part(i).x(),shape.part(i).y()));

            for(int i=48;i<=67;i++) //These points correspond to the mouth
                mouth.push_back(Point(shape.part(i).x(),shape.part(i).y()));

            double ear = (EAR(left_eye)+EAR(right_eye))/2; //Calculates the average EAR for both the eyes 
            double mar = MAR(mouth); //Calculates MAR
            double eye_width_l = dist(left_eye[0],left_eye[3]); //Calculates the width of the left eye
            double eye_width_r = dist(right_eye[0],right_eye[3]); //Calculates the width of the right eye
            double mouth_width = dist(Point(shape.part(48).x(),shape.part(48).y()), Point(shape.part(54).x(),shape.part(54).y())); //Calculates the width of the mouth
            double lip_open = dist(Point(shape.part(62).x(),shape.part(62).y()), Point(shape.part(66).x(),shape.part(66).y())); //Calculates how open the lips are
            double eyebrow_dist = dist(Point(shape.part(21).x(),shape.part(21).y()), Point(shape.part(22).x(),shape.part(22).y())); //Calculates the distance between the eyebrows
            double eyebrow_eye = dist(Point(shape.part(19).x(),shape.part(19).y()), Point(shape.part(37).x(),shape.part(37).y())); //Calculates the vertical distance between the eyebrow and the eye
            double jaw_open = dist(Point(shape.part(62).x(),shape.part(62).y()), Point(shape.part(66).x(),shape.part(66).y())); //Calculates how open the jaw is
            double face_height = dist(Point(shape.part(27).x(),shape.part(27).y()), Point(shape.part(8).x(),shape.part(8).y())); //Calculates the height of the face
            double face_width = dist(Point(shape.part(1).x(),shape.part(1).y()), Point(shape.part(15).x(),shape.part(15).y())); //Calculates the width of the face
            double face_ratio = face_height / face_width; //Calculates the height-to-width ratio of the face
            file << ear << "," << mar << "," << eye_width_l << "," << eye_width_r << "," << mouth_width << "," << lip_open << "," << eyebrow_dist << "," << eyebrow_eye << "," << jaw_open << "," << face_height << "," << face_width << "," << face_ratio << "," << label << "\n"; //Writes these values to the CSV file
        }
    }
    file.close(); //Closes the CSV file
    cout<<"Feature extraction complete\n";
}