import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import numpy as np
import streamlit as st
import tempfile
from PIL import Image
with st.sidebar.container():
 img = Image.open("abc1280x960.png")
 st.image(img,caption='1.American Sign Language')
 img = Image.open("dg.jpg")
 st.image(img,caption='2.Dynamic Gestures')
st.title('Sign Language decoder')
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
            st.write('\n')
v_spacer(height=3, sb=True)
st.text('Click on the button to make it run the specific.........')
col1,col2,col3,col4,col5 = st.columns(5)
with col1:
	run = st.button("1. American Sign Language")
with col2:
	st.text(" ")
with col3:
	run1 = st.button("2. Dynamic Gestures")
with col5:
	run2 = st.button("3. Dynamic Gestures Basic")
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
            st.write('\n')
with col2:
	v_spacer(height=8, sb=True)
	run3 = st.button("5. Test (American Sign Language)")
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
            st.write('\n')
with col4:
	v_spacer(height=9, sb=True)
	run4 = st.button("4. Test (Dinamic Gestures)")
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
            st.write('\n')
v_spacer(height=3, sb=True)
st.markdown('This model is made using yolov5 (Pytorch) The model can be sometimes ambigious while predicting a and m, f and w n and m i and d.')
st.markdown('The versions used while training are :- YOLOv5 🚀 v6.2-211-g32a9218 Python-3.7.15 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)')
st.text('')
if run==1:
	model= torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp25/weights/best.pt',force_reload=True)

	#cap= cv2.VideoCapture('asl (online-video-cutter.com).mp4')
	cap= cv2.VideoCapture(0)
	while cap.isOpened():
	 ret, frame = cap.read()
	 results=model(frame)
	 cv2.imshow('YOLO',np.squeeze(results.render()))
	 if cv2.waitKey(10) & 0xFF == ord('q'):
	  break
	cap.release()
	cv2.destroyAllWindows()
if run1==1:
	model= torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp26/weights/last.pt',force_reload=True)

	#cap= cv2.VideoCapture('asl (online-video-cutter.com).mp4')
	cap= cv2.VideoCapture(0)
	while cap.isOpened():
	 ret, frame = cap.read()
	 results=model(frame)
	 cv2.imshow('YOLO',np.squeeze(results.render()))
	 if cv2.waitKey(10) & 0xFF == ord('q'):
	  break
	cap.release()
	cv2.destroyAllWindows()
if run2==1:
	model= torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp10/weights/best.pt',force_reload=True)

	#cap= cv2.VideoCapture('asl (online-video-cutter.com).mp4')
	cap= cv2.VideoCapture(0)
	while cap.isOpened():
	 ret, frame = cap.read()
	 results=model(frame)
	 cv2.imshow('YOLO',np.squeeze(results.render()))
	 if cv2.waitKey(10) & 0xFF == ord('q'):
	  break
	cap.release()
	cv2.destroyAllWindows()
if run3==1:
	model= torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp25/weights/best.pt',force_reload=True)
	cap= cv2.VideoCapture('asl_60FTDA5y.mp4')
	#cap= cv2.VideoCapture(0)
	while cap.isOpened():
	 ret, frame = cap.read()
	 results=model(frame)
	 cv2.imshow('YOLO',np.squeeze(results.render()))
	 if cv2.waitKey(10) & 0xFF == ord('q'):
	  break
	cap.release()
	cv2.destroyAllWindows()