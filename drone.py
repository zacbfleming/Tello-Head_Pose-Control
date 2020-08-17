import socket
#import multiprocessing
import threading
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork, IEPlugin
import sys
import logging as log
#from time import time 
from argparse import ArgumentParser, SUPPRESS
import os
import time 
from tello import Tello
#from math import *
#from signs import MASL




class InferReqWrap:
    def __init__(self, request, id, num_iter=1):
        self.id = id
        self.request = request
        self.num_iter = 2
        self.cur_iter = 0
        self.cv = threading.Condition()
        self.request.set_completion_callback(self.callback, self.id)

    def callback(self,statusCode, userdata):
        if (userdata != self.id):
            print(self.id)
            log.error("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            print(statusCode)
        self.cur_iter += 1
        if self.cur_iter < self.num_iter:
            self.request.async_infer(self.input)
            
        else:
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

    def execute(self, exec_net,mode, input_data):
        if (mode == "async"):
            self.input = input_data
            self.request.async_infer(input_data)
            self.infer_status = exec_net.requests[-1].wait()
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
        elif (mode == "sync"):
            for self.cur_iter in range(self.num_iter):
                self.request.infer(input_data)

            
        
###initialize IECore and IENetworks
class Model_X:
    def __init__(self, model):
        self.structure=model+'.xml'
        self.weights=model+'.bin'
        self.device='CPU'
        self.batch=1
        print('model initialized')
    
    def load_model(self):
        print('loading model...')
        self.inet=IECore()
        self.enet= IENetwork(self.structure,self.weights)
        self.xnet=self.inet.load_network(self.enet,self.device,num_requests=3)
        self.input_name=next(iter(self.enet.input_info))
        self.output_name=next(iter(self.enet.outputs))
        self.output_shape=self.enet.outputs[self.output_name].shape
        return self
        
    def head_predict(self, image):
        exec_net = self.xnet
        self.request_id = 1
        self.num_iter=2
        self.infer_request =exec_net.requests[self.request_id]
        request_wrap = InferReqWrap(self.infer_request, self.request_id)
        request_wrap.execute(exec_net,"async", {self.input_name: image})
        return self
 
    def predict(self, image,h,w):
        self.request_id=0
        exec_net = self.xnet
        self.num_iter=2
        self.infer_request = exec_net.requests[self.request_id]
        request_wrap = InferReqWrap(self.infer_request, self.request_id, self.num_iter)
        request_wrap.execute(self.xnet,"async", {self.input_name: image})
        self.result = self.infer_request.outputs[self.output_name]
        box = self.ssd(h,w)
        return box
        
    def predict_sign(self, frm_set):
        self.request_id=2
        exec_net = self.xnet
        self.num_iter=1
        self.infer_request = exec_net.requests[self.request_id]
        request_wrap = InferReqWrap(self.infer_request, self.request_id, self.num_iter)
        request_wrap.execute(self.xnet,"async", {self.input_name: image})
        self.result = self.infer_request.outputs[self.output_name]
        
        return self.result

    def ssd(self,h,w):
        for det in self.result.reshape(-1, 7):
            box = []
            conf = float(det[2])
            if conf >= 0.02:
                xmin = int(det[3] * w)
                box.append(xmin)
                ymin= int(det[4] * h)
                box.append(ymin)
                xmax = int(det[5] * w)
                box.append(xmax)
                ymax = int(det[6] * h)
                box.append(ymax)
            return box
                
    def angles(self, req):
        resy = self.infer_request.outputs['angle_y_fc']
        resp = self.infer_request.outputs['angle_p_fc']
        resr = self.infer_request.outputs['angle_r_fc']
        return resy,resp,resr 

    def reframe(self,image,wdt,hgt,b):
        frame = cv2.resize(image, (wdt,hgt))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape(b,3,hgt,wdt)
        return frame
  

def main():
    yaw_list = [ ]
    pitch_list = [ ] 
    roll_list = [ ]
    CLIENT=Tello(host='192.168.10.1', retry_count=1)
    c=('command')
    try:
        CLIENT.send_control_command(c, timeout=12)
    except: pass
    yaw=0
    pitch=0
    roll=0
    box = [0,0,0,0]
    data_count = 0
    frame_count = 0
    pose_model = '/home/artichoke/A_Tello/work/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
    face_model = '/home/artichoke/A_Tello/work/intel/face-detection-retail-0004/FP32/face-detection-retail-0004'
    _F = Model_X(face_model)
    _F.load_model()
    _P = Model_X(pose_model)
    _P.load_model()
    cap = cv2.VideoCapture(0)
    cap.open(0)
    
    while cap.isOpened():
        chk, frame2 = cap.read()
        if not chk:
            break
        frame_count+=1
        h,w,c = np.shape(frame2)
        frm = np.full_like(frame2, frame2)
        faceinf = time.time()
        image = _F.reframe(frame2,300,300,1)
        box = _F.predict(image,h,w)
        p_frame = frame2[box[1]:box[3],box[0]:box[2]]
        face_t = time.time()-faceinf
        pimage = _P.reframe(p_frame,60,60,1)
        yaw,pitch,roll = _P.angles(_P.head_predict(pimage))
        yaw_list.append(yaw)
        pitch_list.append(pitch)
        yw=-2*((int(yaw/10))*10)
        pt=2*((int(pitch/10))*10)
        rl = 2*((int(roll/10))*10)
        frame2 = cv2.rectangle(frame2,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
        
       # cv2.putText(frame2, ('takeoff: %.0f ' % data_count), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0,0), 1,cv2.LINE_AA)
        if abs(yw) < 11 and abs(pt) < 11:
            data_count+=1
            
      #  if data_count < 10 and abs(yw) > 11 or data_count<10 and pt > abs(11):
       #     data_count = 0
        if data_count == 10:
            try:
                #print('takeoff')
                CLIENT.send_control_command('takeoff')
            except: pass
        if data_count > 10:            
            try:
                CLIENT.send_rc_control(left_right_velocity=0, forward_backward_velocity=int(pt), up_down_velocity=0,yaw_velocity=int(yw))
            except: pass
            
           # cv2.putText(frame2, ('takeoff: %.0f ' % data_count), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0,0), 1,cv2.LINE_AA)
        cv2.putText(frame2, ('Yaw: %.3f ' % (yw) ), (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1,cv2.LINE_AA)
        cv2.putText(frame2, ('pitch: %.3f ' % pt), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1,cv2.LINE_AA)
        cv2.putText(frame2, ('Roll: %.3f ' % rl), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 1,cv2.LINE_AA)
        key_pressed=cv2.waitKey(60)
        cv2.imshow('frame',frame2)
        if key_pressed == 27:
            #print(np.std(yaw_list))
            #print(np.std(pitch_list))
                CLIENT.send_control_command('land')
                cv2.waitKey(60)
                cv2.destroyAllWindows()
                break
        




if __name__ == '__main__':
    main()



   
    
   



