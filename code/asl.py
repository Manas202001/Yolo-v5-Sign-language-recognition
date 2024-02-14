import cv2
import streamlit as st
import time

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

my_placeholder = st.empty()

while True:

        success, img = cap.read()
        #cv2.imshow("immagine",img)
        my_placeholder.image(img, use_column_width=True)


        if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # wait for ESC key to exit

cap.release()
cv2.destroyAllWindows()