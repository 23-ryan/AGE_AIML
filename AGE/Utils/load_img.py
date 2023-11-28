import os
import cv2
import numpy as np

def load_img(folder_path, dim=128):
    X = []
    Age = []
    Gender = []
    Eth = []

    images = os.listdir(folder_path)
    for path in images:
        if path.split('_')[1].isdigit() and path.split('_')[2].isdigit() and path.split('_')[3].isdigit():
            img = cv2.imread(f"{folder_path}/{path}")
            if dim!=128:
                img= cv2.resize(img, (dim, dim))
            X.append(list(img))
            Age.append(int(path.split('_')[1]))
            Gender.append(int(path.split('_')[2]))
            Eth.append(int(path.split('_')[3]))
    X = np.array(X)
    Age = np.array(Age)
    Gender = np.array(Gender)
    Eth = np.array(Eth)
    return X, Age, Gender, Eth