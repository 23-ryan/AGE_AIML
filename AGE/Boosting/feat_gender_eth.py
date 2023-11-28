import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
from tensorflow.keras.models import Model
from tqdm import tqdm
from adaboost_clf import AdaBoostClassifierTF
import cv2

from sklearn.model_selection import train_test_split
from vgg_models import VGG_GEN, VGG_ETH

file_name = '../Img_Pickle/Train_Img_resized.pickle'
with open(file_name, 'rb') as file:
    loaded_object = pickle.load(file)

X = loaded_object['resized']
Gender = loaded_object['Gender']
# print("Gender",Gender.shape)
Eth = loaded_object['Eth']
Age = loaded_object['Age']


train_images, val_images, train_gender, val_gender, train_eth, val_eth = train_test_split(
         X, Gender, Eth, test_size=0.1, random_state=42
)

train_gender = train_gender.reshape((-1,1))
train_eth = train_eth.reshape((-1,1))
# print("Gender",train_gender.shape)





backbone_output_lis = [128, 256, 512]

for backbone_output in backbone_output_lis:
   
    model = VGG_GEN(backbone_output)
    # train_images.reshape((224,224,3))
    # print(train_images.shape, train_gender.shape)
    model.fit(train_images, train_gender, epochs=10, batch_size=128)
    # print(val_images.shape)
    val_acc = model.evaluate(val_images, val_gender)
    print(f"Val Accuracy: {val_acc}")
    # print(f"Val Loss: {val_loss}")
    model_path = f'../Saved_Models/vgg_transfer_gender_{backbone_output}.keras'
    model.save(model_path)

    # model_path = f'../Saved_Models/vgg_transfer_gender_{backbone_output}.keras'
    

    vgg_gender_model = tf.keras.models.load_model(model_path)
    feature_layer_model = Model(inputs=vgg_gender_model.input, outputs=vgg_gender_model.get_layer("face_features").output)
    features = feature_layer_model.predict(X)
    features_path_X = f'../Saved_Models/Features/vgg_transfer_gender_{backbone_output}_X.keras'
    np.save(features_path_X, features)
    features_path_Y = f'../Saved_Models/Features/vgg_transfer_gender_{backbone_output}_Y.keras'
    np.save(features_path_Y, Gender)


for backbone_output in backbone_output_lis:
    model = VGG_ETH(backbone_output)
    model.fit(train_images, train_eth, epochs=10, batch_size=128)
    
    model_path = f'../Saved_Models/vgg_transfer_eth_{backbone_output}.keras'
    model.save(model_path)

    val_acc = model.evaluate(val_images, val_eth)

    print(f"Val Accuracy: {val_acc}")
    # print(f"Val Loss: {val_loss}")

   

    vgg_eth_model = tf.keras.models.load_model(model_path)
    feature_layer_model = Model(inputs=vgg_eth_model.input, outputs=vgg_eth_model.get_layer("face_features").output)
    features = feature_layer_model.predict(X)
    features_path_X = f'../Saved_Models/Features/vgg_transfer_eth_{backbone_output}_X.keras'
    np.save(features_path_X, features)
    features_path_Y = f'../Saved_Models/Features/vgg_transfer_eth_{backbone_output}_Y.keras'
    np.save(features_path_Y, Eth)