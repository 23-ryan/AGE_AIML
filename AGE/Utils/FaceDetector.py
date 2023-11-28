import dlib
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()

def FaceDetector(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb)
    a = 0
    x, y, w, h = 0, 0, 0, 0
    for face in faces:
        x_, y_, w_, h_ = face.left(), face.top(), face.width(), face.height()
        # print(1, x_, y_, w_, h_)
        if w_*h_ > a:
            x, y, w, h = x_, y_, w_, h_
            a = w_*h_
  
    crop_image = img[max(0, y -int(.5*h)): min(img.shape[0], y+int(1.5*h)), max(0, x -int(.5*w)): min(img.shape[1], x+int(1.5*w))]
    try:
        resized_image = cv2.resize(crop_image, (128, 128))
    except Exception as e :
        return []
    return resized_image[:,:,::-1]
