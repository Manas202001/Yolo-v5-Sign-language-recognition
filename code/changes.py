#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import numpy as np
import streamlit as st

st.title("Emotion Detection")
st.text("Built with Streamlit and Open CV")
#model=torch.hub.load('ultralytics/yolov5','yolov5s')
model= torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp25/weights/last.pt',force_reload=True)

#cap= cv2.VideoCapture('asl_60FTDA5y.mp4')
cap= cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
my_placeholder = st.empty()
while cap.isOpened():
  ret, frame = cap.read()
  my_placeholder.image(img, use_column_width=True)
  
  results=model(frame)
        
  cv2.imshow('YOLO',np.squeeze(results.render()))

  if cv2.waitKey(10) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

#if __name__ == "__main__":
  #  app.run(debug=True)