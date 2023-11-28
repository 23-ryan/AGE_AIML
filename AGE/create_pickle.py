import pickle
from Utils.load_img import load_img
import numpy as np
import cv2
training_data = 'Data/Train/'

train_images, Age, Gender, Eth = load_img(training_data)

Obj = {'X': train_images, 'Age': Age, 'Gender': Gender, 'Eth': Eth}
file_name = 'Img_Pickle/Train_Img.pickle'

with open(file_name, 'wb') as file:
    pickle.dump(Obj, file)

num, height, width, channels = train_images.shape

# Resize images to (num, 244, 244, 3)
resized_images = np.zeros((num, 224, 224, channels), dtype=np.uint8)

for i in range(num):
    resized_images[i] = cv2.resize(train_images[i], (224, 224))

Obj = {'resized': resized_images, 'Age': Age, 'Gender': Gender, 'Eth': Eth}
file_name = 'Img_Pickle/Train_Img_resized.pickle'
print("done")
with open(file_name, 'wb') as file:
    pickle.dump(Obj, file)
print("done")
