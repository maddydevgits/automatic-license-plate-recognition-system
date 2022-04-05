import streamlit as st
from PIL import Image
import os

import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4,decode

import cv2
import tensorflow as tf
import time
import numpy as np

import demo 
import test as db

size=608 # my training dataset is created with size of 608x608
number_of_classes=len(utils.read_class_names(cfg.YOLO.CLASSES)) # number of classes
STRIDES=np.array(cfg.YOLO.STRIDES)
ANCHORS=utils.get_anchors(cfg.YOLO.ANCHORS,False)
XYSCALE=cfg.YOLO.XYSCALE

st.header("Automatic License Plate Recognition System")

src_file=st.file_uploader("Upload Car Image",type=["png","jpg","jpeg"])

def load_image(image_file):
    img=Image.open(image_file)
    return img

def licensePlateRecognition():
    
    input_layer=tf.keras.layers.Input([size,size,3]) # R,G,B
    feature_maps=YOLOv4(input_layer,number_of_classes)
    bbox_values= [] # bb - bounding box

    for i,fm in enumerate(feature_maps):
        box_value=decode(fm,number_of_classes,i)
        bbox_values.append(box_value)
    
    model=tf.keras.Model(input_layer,bbox_values)
    utils.load_weights(model,'data/YOLOv4-obj_1000.weights')

    frame=cv2.imread('inputs/car.jpg')
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    frame_size=frame.shape[:2] # height and width
    image_data=utils.image_preprocess(np.copy(frame),[size,size]) # image dimens are changed to a fixed 
    image_data=image_data[np.newaxis,...].astype(np.float32)

    pred=model.predict(image_data)
    pred=utils.postprocess_bbbox(pred,ANCHORS,STRIDES,XYSCALE)

    bboxes=utils.postprocess_boxes(pred,frame_size,size,0.25)
    bboxes=utils.nms(bboxes,0.213,method='nms')
    
    for i in range(len(bboxes)):
        img=frame[int(bboxes[i][1]):int(bboxes[i][3]),int(bboxes[i][0]):int(bboxes[i][2])]
        img=cv2.resize(img,(640,300))
        cv2.imwrite('inputs/result.jpg',img)
        k=(demo.readText())

        if (k==None):
            st.error("Re-Captured It again")
        else:
            #st.success(k)
            a=db.readFromDatabase(k)
            s='License Plate: ' + str(a[1]) +'\n,'
            s+=' Owner Name: ' + str(a[2]) +'\n,'
            s+=' Vehicle Type: ' + str(a[3]) +'\n,'
            s+=' Owner Address: ' + str(a[4]) +'\n,'
            s+=' Mobile Number: ' + str(a[5]) +'\n,'
            s+=' Email ID: ' + str(a[6]) +'\n,'
            s+=' Chalan: ' + str(a[7])
            st.success(s)

if src_file is not None:
    file_details={
        "filename": src_file.name,
        "filetype": src_file.type,
        "filesize": src_file.size
    }
    st.write(file_details)
    st.image(load_image(src_file),width=250)

    with open(os.path.join("inputs","car.jpg"),"wb") as f:
        f.write(src_file.getbuffer())
    
    licensePlateRecognition()
    

    

