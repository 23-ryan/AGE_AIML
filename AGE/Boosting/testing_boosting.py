import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adaboost_clf import AdaBoostClassifierTF

print("GENDER PREDICTION")

backbone_output_lis = [128, 256, 512]

for backbone_output in backbone_output_lis:
    features_path_X = f'../Saved_Models/Features/vgg_transfer_gender_{backbone_output}_X.keras'
    feature_X = np.load(features_path_X)

    features_path_Y = f'../Saved_Models/Features/vgg_transfer_gender_{backbone_output}_Y.keras'
    feature_Y = np.load(features_path_Y)
    feature_Y = np.where(feature_Y == 0, -1, 1)

    # You might want to split data here

    classifier = AdaBoostClassifierTF().fit(feature_X, feature_Y, n_estimators=40)
    acc = (classifier.predict(feature_X) == feature_Y).mean()
    print(f'Training error: {acc:.3%}')



for backbone_output in backbone_output_lis:
    features_path_X = f'../Saved_Models/Features/vgg_transfer_eth_{backbone_output}_X.keras'
    feature_X = np.load(features_path_X)

    features_path_Y = f'../Saved_Models/Features/vgg_transfer_eth_{backbone_output}_Y.keras'
    feature_Y = np.load(features_path_Y)
    feature_Y = np.where(feature_Y == 0, -1, 1)

    # You might want to split data here

    classifier = AdaBoostClassifierTF().fit(feature_X, feature_Y, n_estimators=40)
    acc = (classifier.predict(feature_X) == feature_Y).mean()
    print(f'Training error: {acc:.3%}')