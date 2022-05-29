# yolo_opencv_inference
C++ inference implementation for object detection using YOLO deeplearning model

# Download yolo weights
Download yolov3 weigths
```
cd models/yolov3
wget https://pjreddie.com/media/files/yolov3.weights
```

Download yoloV4 weights
```
cd models/yolov4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

# Build 
```
mkdir build
cd build
cmake ..
make
```

# Run inference
The inference code is pre-configured to use your GPU for faster inference time and will use your webcamera as input of the deep learning model.
```
cd build
./yolo 0
```
