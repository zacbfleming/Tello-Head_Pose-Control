import socket
import multiprocessing
import threading
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork, IEPlugin
#import math
import sys
import logging as log
from time import time 
from argparse import ArgumentParser, SUPPRESS
import os
import time 
from tello import Tello
from math import *
from signs import MASL




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
  



class handSignals:        
        self.model_path =  '/home/artichoke/A_Tello/work/intel/asl-recognition-0004/FP32/asl-recognition-0004'
        _S = Model_X(model_path)
        _S.load_model()
    
    
    def remake(self,frm, frame_list):
        imgin  =_S.reframe(frm,224,224,1)
        if len(frame_list) == 16:
            img_set = np.asarray(frame_set,dtype=int)
            sign_set = img_set.reshape(1,3,16,224,224)
            frame_list = [ ]
            print('sign_set',_S.predict_sign(sign_set))  
        if len(frame_list) < 16:
            frame_list.append(imgin)
               
            
    def decode(key):
        for j in MASL:
            if j['label'] == key:
                print(j['org_text'])
                tx = j['org_text']
            return tx
