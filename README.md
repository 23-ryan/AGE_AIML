# ***A.G.E*** ( *Age-Gender-Ethnicity* ) ***Prediction***

This project has been done as part of the course `CS337` (Artifical Intelligence and Machine Learning) <br>
The aim of this project is to build the best possible models for `Age, Gender and Ethnicity` prediction using several Machine Learning techniques

# Important Techniques Used
- **Face Detection**
- **Convolutional Neural Networks**
- **Boosting**
- **Bagging**

# How to run code ?


## Training Process
First create the pickle files for image dataset using `create_pickle.py`

Then use the file `train.py` for training the models, the usage has been shown below
```
usage: train.py [-h] [--model MODEL] [--lr LR] [--dropout DROPOUT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--n_estimators N_ESTIMATORS] [--backbone_output BACKBONE_OUTPUT]
```
You can also refer to `Scripts/data_gen.sh` file for training all models simultaneously <br>

**NOTE** : For training of `Bagging` part, we had to train several single model because of the memory limit within the gpu and some cuda errors. So, while evalating it we have combined all separately trained models. <br>
In order to train the Bagging related models you can look up the `bagging_train.sh` bash script


## Testing Process for Test Dataset
Use `test.py` for testing the models on whole image test dataset, it will also generate the corresponding **plots** for all the models <br>

**NOTE** : You might want to modify the `model's list` withing `test.py` file if you want to test and plot graphs for some specific data

## Demo
Use `run.py` for predicting the **age** **gender** and **ethnicity** for a given sample image
```
python3 run.py <sample-image>
```
