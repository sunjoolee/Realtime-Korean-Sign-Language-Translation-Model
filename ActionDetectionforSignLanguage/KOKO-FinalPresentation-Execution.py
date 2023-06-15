#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install opencv-python mediapipe scikit-learn matplotlib
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard 

from PIL import ImageFont, ImageDraw, Image


# In[ ]:


global mp_holistic
mp_holistic= mp.solutions.holistic # Holistic model
global mp_drawing
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Actions that we try to detect
global actions
actions = np.array(['house', 'thief', 'outofbreath', 'down', 'fire', 'stranger', 'car_accident'])
global actions_korean
actions_korean = ['집', '도둑', '숨이 안쉬어져요', '아래', '불이 났어요', '낯선사람', '교통사고']

global label_map
label_map = {label:num for num, label in enumerate(actions)}

# 1 Video = 50 frames
global sequence_length
sequence_length = 50


# In[ ]:


def mediapipe_detection(image, model): 
    #image = feed frame
    #model = Holistic model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR -> RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB -> BGR
    return image, results


# In[ ]:


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# In[ ]:


def extract_keypoints(results):
    #result의 landmarks의 모든 key point values -> 하나의 numpy array 로 flatten
    #if landmarks has no value, fill numpy array with zero
    if results.pose_landmarks: #pose landmarks
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() 
    else:
        pose = np.zeros(132) #33*4

    if results.left_hand_landmarks: #left hand landmarks
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() 
    else:
        lh = np.zeros(63) #21*3

    if results.right_hand_landmarks: #right hand landmarks
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() 
    else:
        rh = np.zeros(63) #21*3
    
    return np.concatenate([pose, lh, rh])


# In[ ]:


#한글 텍스트 출력
def putKoreanText(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('C:/Users/User/ActionDetectionforSignLanguage/fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill= font_color)
    return np.array(img_pil)


# In[ ]:


def main():
    #build neural network architecture
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
 
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    model.load_weights('kslaction_50_frame.h5')
    
    # 1. Detection variables
    sequence = [] #collect 50 frames to make a sequence(=video)
    sentence = [] #concatenate history of predictions together
    threshold = 0.999

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-50:] #generate sequence with last 30 frames

            if len(sequence) == 50:
                #sequence.shape = (50, 258)
                #the input shape model expects = (number of sequences, 50, 258)
                res = model.predict(np.expand_dims(sequence, axis=0))[0] #predict one sequence at a time
                print(actions[np.argmax(res)])

             #3. Rendering logic
                if res[np.argmax(res)] > threshold: 
                    cur_action_korean = actions_korean[np.argmax(res)]

                    if len(sentence) > 0: 
                        #sentence에 저장된 prediction이 있는 경우 
                        #새로운 prediction인 경우에만 sentence에 추가
                        if cur_action_korean != sentence[-1]:
                            sentence.append(cur_action_korean)
                    else: 
                        #sentence에 저장된 prediction 없는 경우 바로 sentence에 추가
                        sentence.append(cur_action_korean)

                #sentence가 너무 길어지지 않도록 마지막 5개의 prediction만 유지
                if len(sentence) > 5: 
                    sentence = sentence[-5:]


                cv2.rectangle(image, (0,0), (640, 40), (0, 0, 0), -1) 
                #putKoreanText(src, text, pos, font_size, font_color
                image = putKoreanText(image, ' '.join(sentence), (3,10), 20, (255, 255, 255))


                # Show to screen
                cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# In[ ]:


if __name__ == '__main__':
    main()

