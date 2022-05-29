#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>
#include <unistd.h>
#include <limits.h>

// Initialize the parameters
float confThreshold = 0.5; //  Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold

int inpWidth = 416; // 608 Width of network's input image
int inpHeight = 416;  // 608 Height of network's input image

std::vector<std::string> classes;

//Prototype functions
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);


//================================================
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<int>& classIds, std::vector<int> &centersX, std::vector<int> &centersY, std::vector<cv::Rect>& boxes)
{
    //vector<int> classIds;
    std::vector<float> confidences;
    //vector<Rect> boxes;
	//vector<int> centersX;
	//vector<int> centersY;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
				//if(classIdPoint.x==0){ // filter to show only people 
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
				centersX.push_back(centerX);
				centersY.push_back(centerY);
				//}
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];	
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                box.x + box.width, box.y + box.height, frame);
		
    }
}
//========================================================
// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}
//========================================================
// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
//===========================================================
//===========================================================
//===========================================================
int main(int argc,char** argv)
{
    std::cout << "Staring yolo" << std::endl;

    //Choose here the yolo version [yolov3","yolov4","yolov4-tiny"]
    std::string yolo_version = "yolov4-tiny";

    //get pwd
    std::string current_folder;
    char cwd[PATH_MAX];
    std::stringstream ss;
    if (getcwd(cwd, sizeof(cwd)) != NULL) 
    {
      ss << cwd;
      ss >> current_folder;
    }

	// Load names of classes
	std::cout << "loading classes...";    	
	std::string classesFile = current_folder + "/../" +"/models/" + yolo_version + "/coco.names";
	std::string modelConfiguration = current_folder + "/../" + "/models/"+ yolo_version + "/" + yolo_version + ".cfg";
    std::string modelWeights = current_folder + "/../" + "/models/" + yolo_version + "/" + yolo_version + ".weights";
    
    
    //string classesFile = "/home/fischer/Desktop/Fischer/Projects/Stereo_Vision_Depth_Img/yoloDepth/models/CYTi/CYTI.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);
	std::cout << "done\n";
    

    // Load the network
	std::cout << "loading network...";
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

	//run in CPU
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); 
    //std::cout<<"Using CPU" << "\n";

	//run in GPU
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); 
    std::cout<<"Using Cuda" <<"\n";
	std::cout << "done\n";  

    cv::VideoCapture cap(atoi(argv[1]));
    if (!cap.isOpened()) {return 0;}

    while(true) 
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
        cv::Mat frame;
        cap >> frame;
        if(frame.empty()) {break;} // end of video stream

        //resize image
        cv::resize(frame,frame,cv::Size(848,480), cv::INTER_LINEAR);

        //Object detection
		cv::Mat blob; 
		
		// Create a 4D blob from a frame.
		cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
		
		//Sets the input to the network
		net.setInput(blob);
		
		// Runs the forward pass to get output of the output layers
		std::vector<cv::Mat> outs;
		net.forward(outs, getOutputsNames(net));

		std::vector<int> classId;
		std::vector<int> centersX;
		std::vector<int> centersY;
		std::vector<cv::Rect> boxes; //[left, top, width, height]
		std::vector<cv::Rect> people_boxes;

		
		// Remove the bounding boxes with low confidence
		postprocess(frame, outs, classId, centersX, centersY, boxes);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        double fps = 1/(ttrack);
        std::cout << "fps: " << fps  << std::endl;

        cv::imshow("camera feed",frame);
        if(cv::waitKey(10)==27){break;} //stop capturing by pressing ESC

    }
    std::cout<<"End yolo" <<"\n";
    return 0;

}