# Tello-Head_Pose-Control
Required
tello sdk 2.0
Openvino 2020.1.023
uses head pose estimation model to guide tello drone

Recently received my Udacity nanodegree after finishing the Intel Edge AI for IoT developers scholarship. After completing my final project, (a python app that uses 4 trained network models - face detection -head pose estimation -facial landmark detection -gaze estimation to move a computer cursor by the users gaze instead of a mouse) I adapted the app to guide a drone using the pose angles of the users head. I applied processing optimizations using openvino 2020.1.023 techniques (such as using 8 bit model weights and offsets without accuracy loss (slight gain) using a Mobilenet model and pruning the 3x3 convolutions and by using a binary XNOR+POPCOUNT approach. The app also utilizes asynchronous model execution on input video for near real time execution. The Tello SDK provides functionality for the drone and translation for commands. 
