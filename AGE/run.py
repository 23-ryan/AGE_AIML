import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Utils.FaceDetector import FaceDetector
import sys
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if len(sys.argv) != 2:
    print("Use ./run.py <FileName> ")
    exit(0)

file_name = sys.argv[-1]
img = cv2.imread(file_name)
face_img = FaceDetector(img=img)

if len(face_img) == 0:
    print("No face found")
    exit(0)

GENDER = ["Male", "Female"]

age_model = f"Saved_Models/base_AGE_None_0.0001_0.5.keras"
gender_model = f"Saved_Models/base_GEN_32_0.0001_0.5.keras"
eth_model = f"Saved_Models/base_ETH_32_0.0001_0.5.keras"
age_loaded_model = tf.keras.models.load_model(age_model)
gender_loaded_model = tf.keras.models.load_model(gender_model)
eth_loaded_model = tf.keras.models.load_model(eth_model)

age = age_loaded_model.predict(np.array([face_img]))[0,0]
gender = gender_loaded_model.predict(np.array([face_img]))[0,0]
eth = eth_loaded_model.predict(np.array([face_img]))

print("Age : ", age)
print("Ethnicity : ", np.argmax(eth))
print("Gender : ", GENDER[int(gender > .5)])

plt.imshow(face_img)
plt.show()