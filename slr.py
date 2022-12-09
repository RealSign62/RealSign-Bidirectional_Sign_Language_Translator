import streamlit as st
import cv2
import copy
import mediapipe as mp
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
from gtts import gTTS
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

stframe = st.empty()
model = load_model('path_to_model')

def get_bbox_coordinates(handLadmark, image_shape):
    
    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
        all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0])) # multiply y by image height

    return min(all_x), min(all_y), max(all_x), max(all_y) # return as (xmin, ymin, xmax, ymax)

def remove_consec_duplicates(s):
    new_s = ""
    prev = ""
    for c in s:
        if len(new_s) == 0:
            new_s += c
            prev = c
        if c == prev:
            continue
        else:
            new_s += c
            prev = c
    return new_s

def add_to_word(res):
    # global word

    # global st.session_state.word

    if(res==0):
        st.session_state.word = st.session_state.word + 'a'
    elif(res==1):
        st.session_state.word = st.session_state.word + 'b'
    elif(res==2):
        st.session_state.word = st.session_state.word + 'c'
    elif(res==3):
        st.session_state.word = st.session_state.word + 'd'
    elif(res==4):
        st.session_state.word = st.session_state.word + 'e'
    elif(res==5):
        st.session_state.word = st.session_state.word + 'f'
    elif(res==6):
        st.session_state.word = st.session_state.word + 'g'
    elif(res==7):
        st.session_state.word = st.session_state.word + 'h'
    elif(res==8):
        st.session_state.word = st.session_state.word + 'i'
    elif(res==9):
        st.session_state.word = st.session_state.word + 'j'
    elif(res==10):
        st.session_state.word = st.session_state.word + 'k'
    elif(res==11):
        st.session_state.word = st.session_state.word + 'l'
    elif(res==12):
        st.session_state.word = st.session_state.word + 'm'
    elif(res==13):
        st.session_state.word = st.session_state.word + 'n'
    elif(res==14):
        st.session_state.word = st.session_state.word + 'o'
    elif(res==15):
        st.session_state.word = st.session_state.word + 'p'
    elif(res==16):
        st.session_state.word = st.session_state.word + 'q'
    elif(res==17):
        st.session_state.word = st.session_state.word + 'r'
    elif(res==18):
        st.session_state.word = st.session_state.word + 's'
    elif(res==19):
        st.session_state.word = st.session_state.word + 't'
    elif(res==20):
        st.session_state.word = st.session_state.word + 'u'
    elif(res==21):
        st.session_state.word = st.session_state.word + 'v'
    elif(res==22):
        st.session_state.word = st.session_state.word + 'w'
    elif(res==23):
        st.session_state.word = st.session_state.word + 'x'
    elif(res==24):
        st.session_state.word = st.session_state.word + 'y'
    elif(res==25):
        st.session_state.word = st.session_state.word + 'z'
    

def track_hands_in_video():

    # global word
    # This uses the server's camera, i.e the camera of the system the code is running on.
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            if results.multi_hand_landmarks:
                brect = []
                for hand_landmarks in results.multi_hand_landmarks:
                    brect.append(get_bbox_coordinates(hand_landmarks, (h, w)))
                    
                if len(brect) == 1:
                    
                    cv2.rectangle(image, (brect[0][0] - 35, brect[0][1] - 35), (brect[0][2] + 35, brect[0][3] + 35), (0, 255, 0), 2)
                    roi = image[brect[0][1] - 35:brect[0][3] + 35, brect[0][0] - 35:brect[0][2] + 35]
                    roi = cv2.resize(roi, (128, 128))
                    time.sleep(1)
                    res=np.argmax(model.predict(roi.reshape(1, 128, 128, 3))[0])
                    add_to_word(res)
                    # st.text(st.session_state.word)
                    # st.text(res)

                if len(brect) > 1:
                    
                    cv2.rectangle(image, (min(brect[0][0], brect[1][0]) - 35, (min(brect[0][1], brect[1][1])) - 35), (max(brect[0][2], brect[1][2]) + 35, (max(brect[0][3], brect[1][3])) + 35), (0, 255, 0), 2)
                    roi = image[min(brect[0][1], brect[1][1]) - 35:max(brect[0][3], brect[1][3]) + 35,  min(brect[0][0], brect[1][0]) - 35:max(brect[0][2], brect[1][2]) + 35]
                    roi = cv2.resize(roi, (128, 128))
                    time.sleep(1)
                    res=np.argmax(model.predict(roi.reshape(1, 128, 128, 3))[0])
                    add_to_word(res)
                    # st.text(st.session_state.word)
                    # st.text(res)
            

            stframe.image(cv2.flip(image, 1))
            if cv2.waitKey(25) & 0xFF == ord('r'):
                break
    

    cap.release()
    cv2.destroyAllWindows()


st.title('Sign Language to Speech')

# Empty string to carry word

if model is not None:
    
    if 'word' not in st.session_state:
        st.session_state.word = ''

    start_button = st.button('Start')
    stop_button = st.button('Stop')

    if start_button:
        
        # word = ''
        track_hands_in_video()

    if stop_button:

        st.session_state.word = remove_consec_duplicates(st.session_state.word)

        text_placeholder = st.empty()
        text_placeholder.write(st.session_state.word)
        
        # speak(st.session_state.word)
        
        # word = ''
        # st.session_state.word = ''

        start_button = 0
        stop_button = 0

