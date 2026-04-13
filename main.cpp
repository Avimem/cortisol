#include <opencv2/opencv.hpp> //Used to display and capture the frames from the webcam
#include <dlib/opencv.h> //Used to convert OpenCV images into Dlib compatible format
#include <dlib/image_processing.h> //Used to perform landmark detection on the faces
#include <dlib/image_processing/frontal_face_detector.h> //Used to detect faces in a frame
#include <dlib/image_processing/correlation_tracker.h> //Used to track the face between frames
#include <winsock2.h> //Used for network programming on Windows
#include <ws2tcpip.h> ////Used for TCP socket communication with the Python server
#include <iostream> //Used for I/O
#include <vector> //Used to make dynamic arrays
#include <deque> //Used to make a buffer for smoothing predictions
#pragma comment(lib,"ws2_32.lib") //Used to tell the compiler to link the Windows Socket Library
using namespace cv; //Used to avoid writing cv::
using namespace std; //Used to avoid writing std::

double dist(Point a, Point b){ //Used to calculate the Euclidean Distance between 2 points
    return norm(a-b);
}

double EAR(const vector<Point>& eye){ //Used to calculate the Eye Aspect Ratio (EAR)
    return (dist(eye[1],eye[5]) + dist(eye[2],eye[4])) / (2.0*dist(eye[0],eye[3]));
}

double MAR(const vector<Point>& mouth){ //Used to calculate the Mouth Aspect Ratio
    return (dist(mouth[2],mouth[10]) + dist(mouth[4],mouth[8])) / (2.0*dist(mouth[0],mouth[6]));
}

int main(){
    WSADATA wsData; //Used to start the Windows Socket API
    WSAStartup(MAKEWORD(2,2), &wsData); //Used to specify the API version the program will be using 
    SOCKET sock = socket(AF_INET,SOCK_STREAM,0); //Used to create a socket
    sockaddr_in serverAddr; //Used to store the server info
    serverAddr.sin_family = AF_INET; //Used to specify the address type which is being used, which in this case is IPv4
    serverAddr.sin_port = htons(5000); //Used to specify the port number which is being used, which in this case is 5000
    inet_pton(AF_INET,"127.0.0.1",&serverAddr.sin_addr); //Used to set the server's IP address, which in this case is 127.0.0.1
    connect(sock,(sockaddr*)&serverAddr,sizeof(serverAddr)); //Used to establish the TCP connection
    cout<<"Connected to Python server\n";
    VideoCapture cap(0); //Used to open the default webcam
    cap.set(CAP_PROP_FRAME_WIDTH,640); //Used to set the width of the OpenCV window, which in this case is 640
    cap.set(CAP_PROP_FRAME_HEIGHT,480); //Used to set the height of the OpenCV window, which in this case is 480
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector(); //Used to load the face detection model
    dlib::correlation_tracker tracker; //Used to store a boolean value which indicates whether tracking is active or not
    bool tracking=false;
    dlib::shape_predictor predictor; //Predicts the 68 facial landmark points on the face
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor; //Used to load the facial landmark model
    deque<int> predictions; //Used to store recent predictions
    const int BUFFER = 12; //Used to set the buffer size, which in this case is 12
    int lastPrediction = 0; //Used to store the last prediction
    Mat frame; //Used to store each webcam frame

    while(true){ //Used to run the program until it is closed
        cap >> frame; //Used to read each frame from the webcam

        if(frame.empty()) break; //Used to stop the program from running if frame capture does not work properly

        resize(frame,frame,Size(640,480)); //Used to ensure that the size of the frame remains consistent
        dlib::cv_image<dlib::bgr_pixel> dimg(frame); //Used to convert each captured frame into a Dlib compatible format
        dlib::rectangle faceRect; //Used to store the detected face in a bounding box
        if(!tracking){ //Enters this block if no face is currently being detected
            auto faces = detector(dimg); //Used to detect faces
            if(!faces.empty()){ //Enters this block if a face is found
                faceRect = faces[0]; //Used to store the face in a dynamic array
                tracker.start_track(dimg,faceRect); //Used to start tracking the face
                tracking=true;
            }
        }
        else{
            double conf = tracker.update(dimg); //Used to store the tracking confidence
            faceRect = tracker.get_position(); //Used to update the current position of the tracked face

            if(conf < 5){ //Enters this block if tracking confidence is low
                tracking=false;
                continue;
            }
        }

        if(tracking){ //Enters this block if a face is being tracked
            rectangle(frame, Point(faceRect.left(),faceRect.top()), Point(faceRect.right(),faceRect.bottom()), Scalar(0,255,0), 2); //Used to draw a green bounding box around the face which is being tracked
            auto shape = predictor(dimg,faceRect); //Used to detect the 68 facial landmark points on the face which is being tracked
            vector<Point> leftEye,rightEye,mouth; //Used to store the location of the landmark points in a dynamic array

            for(int i=36;i<=41;i++) //Used to store the location of the points on the left eye
                leftEye.push_back(Point(shape.part(i).x(),shape.part(i).y()));

            for(int i=42;i<=47;i++) //Used to store the location of the points on the right eye
                rightEye.push_back(Point(shape.part(i).x(),shape.part(i).y()));

            for(int i=48;i<=67;i++) //Used to store the location of the points on the mouth
                mouth.push_back(Point(shape.part(i).x(),shape.part(i).y()));

            for(int i=36;i<=47;i++) //Used to draw the landmark points of the eyes on the OpenCV window
                circle(frame,Point(shape.part(i).x(),shape.part(i).y()),2,Scalar(0,255,0),-1);

            for(int i=17;i<=26;i++) //Used to draw the landmark points of the eyebrows on the OpenCV window
                circle(frame,Point(shape.part(i).x(),shape.part(i).y()),2,Scalar(0,255,0),-1);

            circle(frame,Point(shape.part(48).x(),shape.part(48).y()),2,Scalar(0,255,0),-1); //Used to draw the landmark point of the left corner of the mouth on the OpenCV window
            circle(frame,Point(shape.part(54).x(),shape.part(54).y()),2,Scalar(0,255,0),-1); //Used to draw the landmark point of the right corner of the mouth on the OpenCV window
            double ear = (EAR(leftEye)+EAR(rightEye))/2; //Used to calculate and store the average EAR value of both the eyes
            double mar = MAR(mouth); //Used to store the MAR value
            double eye_w_l = dist(leftEye[0],leftEye[3]); //Used to store the width of the left eye
            double eye_w_r = dist(rightEye[0],rightEye[3]); //Used to store the width of the right eye
            double mouth_w = dist(Point(shape.part(48).x(),shape.part(48).y()), Point(shape.part(54).x(),shape.part(54).y())); //Used to store the width of the mouth
            double lip_open = dist(Point(shape.part(62).x(),shape.part(62).y()), Point(shape.part(66).x(),shape.part(66).y())); //Used to store the height of the mouth
            double brow_dist = dist(Point(shape.part(21).x(),shape.part(21).y()), Point(shape.part(22).x(),shape.part(22).y())); //Used to store the distance between the eyebrows
            double brow_eye = dist(Point(shape.part(19).x(),shape.part(19).y()), Point(shape.part(37).x(),shape.part(37).y())); //Used to store the distance between the eyebrow and the eye 
            double jaw_open = dist(Point(shape.part(62).x(),shape.part(62).y()), Point(shape.part(66).x(),shape.part(66).y())); //Used to store the distance of the opening of the mouth
            double face_h = dist(Point(shape.part(27).x(),shape.part(27).y()), Point(shape.part(8).x(),shape.part(8).y())); //Used to store the height of the face
            double face_w = dist(Point(shape.part(1).x(),shape.part(1).y()), Point(shape.part(15).x(),shape.part(15).y())); //Used to store the width of the face
            double face_ratio = face_h/face_w; //Used to calculate and the store the face geometry ratio

            if(ear < 0.1 || ear > 0.6) //Used to reject unrealistic EAR values caused by bad detections
                continue;

            string msg = to_string(ear)+ "," + to_string(mar) + "," + to_string(eye_w_l) + "," + to_string(eye_w_r) + "," + to_string (mouth_w) + "," + to_string(lip_open) + "," + to_string(brow_dist) + "," + to_string(brow_eye) + "," + to_string(jaw_open) + "," + to_string(face_h) + "," + to_string(face_w) + "," + to_string(face_ratio); //Used to create a comma-separated feature string
            send(sock,msg.c_str(),msg.size(),0); //Used to send the said feature string to the Python server via TCP
            char buffer[32]; //Used to create a buffer for the server response
            int bytes = recv(sock,buffer,32,0); //Used to receive the prediction
            buffer[bytes]='\0'; //Used to add the string terminator
            lastPrediction = atoi(buffer); //Used to convert the string into an integer
            predictions.push_back(lastPrediction); //Used to store the prediction 

            if(predictions.size()>BUFFER) //Enters this block if the size of the variable is more than the buffer size set for it
                predictions.pop_front();

            int stressCount = 0; //Used to count the "Stress" predictions in the buffer

            for(int p : predictions){ //Used to loop through the stored predictions
                if(p==1) stressCount++;
            }

            bool stressed = stressCount > BUFFER / 2; //Used to store the final stress detection
            string emotion; //Used to store the emotion
            double smileRatio = mouth_w / face_w; //Used to calculate if a person is smiling
            double browRatio = brow_dist / face_w; //Used to calculate if a person is showing fear

            if(smileRatio > 0.42){ //Enters this block if person is smiling
                emotion = "Happy";
            }
            else if(mar > 0.65 && ear > 0.27){ //Enters this block if person is in fear
                emotion = "Fear";
            }
            else if(browRatio < 0.17 && ear < 0.27){ //Enters this block if person is angry
                emotion = "Angry";
            }
            else if(ear < 0.22){ //Enters this block if person is sad
                emotion = "Sad";
            }
            else{ //Enters this block if none of the above cases are true
                emotion = "Neutral";
            }

            if(emotion == "Fear"){ //Sets the stress label to be true if person is in a state of fear
                stressed = true;
            }
            if(emotion == "Angry"){ //Sets the stress label to be true if person is in a state of anger
                stressed = true; 
            }
            if(emotion == "Happy"){ //Sets the stress label to be true of person is in a state of happiness
                stressed = false;
            }

            static deque<string> emotionBuffer; //Used to create an emotion buffer
            emotionBuffer.push_back(emotion); //Used to push the detected emotion into the buffer

            if(emotionBuffer.size() > BUFFER){ //Enters this block if the emotion buffer overflows
                emotionBuffer.pop_front();
            }

            map<string, int> emotionCount; //Used to smooth the emotion prediction results

            for(string e : emotionBuffer){ //Used to check which emotions are present in the buffer
                emotionCount[e]++;
            }

            string finalEmotion = "Neutral"; //Used to store the final emotion detection with a default value of "Neutral"
            int maxCount = 0; //Used to count which emotion has the most frequency in the buffer

            for(auto &p : emotionCount){ //Iterates through the dictionary to see which emotion has the most frequency
                if(p.second > maxCount){
                    maxCount = p.second;
                    finalEmotion = p.first;
                }
            }

            Mat overlay; //Used to create a semi-transparent overlay
            frame.copyTo(overlay); //Used to copy the current frame to the overlay
            rectangle(overlay, Point(10,10), Point(320,130), Scalar(0,0,0), -1); //Used to draw a black filled rectangle on the overlay
            addWeighted(overlay, 0.5, frame, 0.5, 0, frame); //Used to blend the overlay with the original frame
            putText(frame, "Stress Detection System", Point(20,30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 2); //Used to draw the title on the overlay
            string statusText = stressed ? "STRESSED" : "RELAXED"; //Used to store the current status of the user which will be displayed on the screen
            putText(frame, "Status: " + statusText, Point(20,60), FONT_HERSHEY_SIMPLEX, 0.7, stressed ? Scalar(0,0,255) : Scalar(0,255,0), 2); //Used to display the user's current status in red text if they are stressed or in green text if they are relaxed
            putText(frame, "Emotion: " + finalEmotion, Point(20,85), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2); //Used to display the user's current emotion on the screen
            int confidence = (maxCount * 100) / emotionBuffer.size(); //Used to calculate the confidence percentage of the user's current status
            rectangle(frame, Point(20,100), Point(220,115), Scalar(100,100,100), -1); //Used to draw a grey background bar on the overlay
            int barWidth = (confidence * 200) / 100; //Used to scale the bar according to the confidence level
            Scalar barColor; //Used to store the color of the confidence bar

            if(confidence > 70){ //Enters this block if the confidence level is more than 70
                barColor = Scalar(0,255,0);
            }    
            else if(confidence > 40){ //Enters this block if the confidence level is more than 40
                barColor = Scalar(0,255,255);
            }    
            else{ //Enters this block if the confidence level is less than 40
                barColor = Scalar(0,0,255);
            }    

            rectangle(frame, Point(20,100), Point(20+barWidth,115), barColor, -1); //Used to draw the resulting color in the bar
            putText(frame, to_string(confidence) + "%", Point(230,112), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 2); //Used to display the confidence percentage
        }

        imshow("Stress Detection",frame); //Used to display the webcam feed

        if(waitKey(1)==27) break; //Used to stop the program when the "Esc" key is pressed
    }

    closesocket(sock); //Used to close the socket
    WSACleanup(); //Used to close the connection
    cap.release(); //Used to release the webcam
    destroyAllWindows(); //Used to close all the windows
    return 0;
}
