from ultralytics import YOLO
import cv2
import numpy as np
import time
from plot import Annotator

model=YOLO("yolov8n-pose.onnx")   


cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
annot=Annotator(cap.read())


while True:
    ret,img_orig=cap.read()
    start_time=time.time()
    results=model(img_orig,save_txt=False, stream=True,verbose=False,imgsz=(224,224))


    for r in results:
        kpoints =r.keypoints.data
        
    for keyPoints in kpoints:
        img_orig=annot.kpts(img_orig,keyPoints)

    cv2.imshow('output',img_orig)
    cv2.waitKey(1)

    # print("FPS: ",1.0/(time.time()-start_time))
    # print("ms ",(time.time()-start_time))


cap.release()