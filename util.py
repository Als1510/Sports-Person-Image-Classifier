from turtle import st
import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import streamlit as st

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        print(f)
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)

load_saved_artifacts()

def local_css(file_name):
  with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def img_to_bytes(img):
    img_bytes = img.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def classify_image(img):
    image_base64_data = img_to_bytes(img)
    imgs = get_cropped_image_if_2_eyes(image_base64_data)
    if len(imgs) == 0:
        return 0
    result = []
    scalled_raw_img = cv2.resize(imgs, (32, 32))
    img_har = w2d(imgs, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

    len_image_array = 32*32*3 + 32*32

    final = combined_img.reshape(1,len_image_array).astype(float)
    result.append({
        'class': class_number_to_name(__model.predict(final)[0]),
        'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
        'class_dictionary': __class_name_to_number
    })

    return result

def get_cv2_image_from_base64_string(b64str):
    nparr = np.frombuffer(base64.b64decode(b64str), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
    return img

def get_cropped_image_if_2_eyes(image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    img = get_cv2_image_from_base64_string(image_base64_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return cropped_faces