import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

from Utils.load_img import load_img
import tensorflow as tf
import matplotlib.pyplot as plt
from Models.Bagging_GEN import Bagging_GEN
from Models.Bagging_ETH import Bagging_ETH
from Models.Bagging_AGE import Bagging_AGE
from Models.Boosting_GEN import Boosting_GEN
from Models.Boosting_ETH import Boosting_ETH

# plt.style.use('dark_background')

# model_lis = ["base_GEN", "base_ETH", "base_AGE"]
# model_lis = ["AGE_ETH_GEN"]
# model_lis = ["base_AGE", "base_GEN", "base_ETH", "AGE_ETH_GEN", "AGE_G_GEN", "ETH_G_GEN"]
# model_lis = ["Bagging_AGE"]
# model_lis = ["AGE_G_GEN", "ETH_G_GEN"]
model_lis = ["base_AGE", "base_GEN", "base_ETH", "AGE_ETH_GEN", "AGE_G_GEN", "ETH_G_GEN", "Bagging_GEN", "Bagging_AGE", "Bagging_ETH"]
batch_size_lis = ["32", "128", "None"]
lr_lis = ["0.001", "0.0005", "0.0001"]
n_estimators_lis = [2, 4, 8, 16, 32]
dropout = 0.5
backbone_output = [128, 256, 512]
n_esti = [5, 10, 15]
model_path = "Saved_Models/"

test_data = 'Data/Test/'
test_images, test_age, test_gender, test_eth = load_img(test_data)

test_data = 'Data/Test/'
test_images_224, test_age, test_gender, test_eth = load_img(test_data, 224)

X_labels = []
for batch_size in batch_size_lis:
    for lr in lr_lis:
        X_labels.append(f"{batch_size}-{lr}")

for model in model_lis:
    
    if model == "AGE_ETH_GEN":
        age_loss_list = []
        gender_loss_list = []
        eth_loss_list = []

        age_acc_list = []
        gender_acc_list = []
        eth_acc_list = []

        for batch_size in batch_size_lis:
            for lr in lr_lis:
                model_path = f"Saved_Models/{model}_{batch_size}_{lr}_{dropout}.keras"
                loaded_model = tf.keras.models.load_model(model_path)
                test_label = {"gender_output" : test_gender, "age_output" : test_age, "eth_output" : test_eth}
                test_res = loaded_model.evaluate(test_images, test_label)

                gender_acc = test_res[4]
                eth_acc = test_res[-1]
                age_loss = test_res[2]
                gender_loss = test_res[1]
                eth_loss = test_res[3]

                age_loss_list.append(age_loss)
                gender_loss_list.append(gender_loss)
                eth_loss_list.append(eth_loss)
                
                gender_acc_list.append(gender_acc)
                eth_acc_list.append(eth_acc)


        plt.figure(figsize=(12, 6))


        plt.plot(X_labels, gender_loss_list, label='Gender loss') 
        plt.scatter(X_labels, gender_loss_list)

        plt.plot(X_labels, eth_loss_list, label='Ethnicity loss') 
        plt.scatter(X_labels, eth_loss_list)

        plt.plot(X_labels, age_loss_list, label='Age MAE loss')
        plt.scatter(X_labels, age_loss_list)

        plt.legend()

        plt.xlabel('Batchsize--LR')
        plt.ylabel('Loss')
        plt.title("Combined Age, Gender and Ethnicity Predictor Model")
        plt.savefig("Plots/AGE_ETH_GEN_Loss.png")
        plt.clf()


        plt.figure(figsize=(12, 6))

        plt.plot(X_labels, gender_acc_list, label='Gender Accuracy')
        plt.scatter(X_labels, gender_acc_list)

        plt.plot(X_labels, eth_acc_list, label='Ethnicity Accuracy')
        plt.scatter(X_labels, eth_acc_list)

        plt.legend()

        plt.xlabel('Batchsize--LR')
        plt.ylabel('Accuracy')
        plt.title("Combined Age, Gender and Ethnicity Predictor Model")
        plt.savefig("Plots/AGE_ETH_GEN_Acc.png")
        plt.close()

    elif model == "base_GEN":
        gender_acc = []
        gender_loss = []

        for batch_size in batch_size_lis:
            for lr in lr_lis:
                model_path = f"Saved_Models/{model}_{batch_size}_{lr}_{dropout}.keras"
                loaded_model = tf.keras.models.load_model(model_path)
                test_loss, test_acc = loaded_model.evaluate(test_images, test_gender)
                gender_loss.append(test_loss)
                gender_acc.append(test_acc)

        plt.figure(figsize=(12, 6))
        plt.plot(X_labels, gender_loss, label='Loss') 
        plt.scatter(X_labels, gender_loss)

        plt.plot(X_labels, gender_acc, label='Accuracy')
        plt.scatter(X_labels, gender_acc)

        plt.legend()

        plt.xlabel('Batchsize--LR')
        plt.ylabel('Loss and Accuracy')
        plt.title("Basic Gender Predictor Model")
        plt.savefig("Plots/base_GEN.png")
        plt.close()

    elif model == "base_AGE":
        age_loss = []
        for batch_size in batch_size_lis:
            for lr in lr_lis:
                model_path = f"Saved_Models/{model}_{batch_size}_{lr}_{dropout}.keras"
                loaded_model = tf.keras.models.load_model(model_path)
                test_loss = loaded_model.evaluate(test_images, test_age)
                age_loss.append(test_loss)


        plt.figure(figsize=(12, 6))
        plt.plot(X_labels, age_loss, label='Loss') 
        plt.scatter(X_labels, age_loss)

        plt.legend()
        plt.xlabel('Batchsize--LR')
        plt.ylabel('Loss')
        plt.title("Basic Age Predictor Model")
        plt.savefig("Plots/base_AGE.png")
        plt.close()


    elif model == "base_ETH":
        eth_loss = []
        eth_acc = []
        for batch_size in batch_size_lis:
            for lr in lr_lis:
                model_path = f"Saved_Models/{model}_{batch_size}_{lr}_{dropout}.keras"
                loaded_model = tf.keras.models.load_model(model_path)
                test_loss, test_acc = loaded_model.evaluate(test_images, test_eth)
                eth_loss.append(test_loss)
                eth_acc.append(test_acc)

        plt.figure(figsize=(12,6))
        plt.plot(X_labels, eth_loss, label='Loss') 
        plt.scatter(X_labels, eth_loss)
        plt.plot(X_labels, eth_acc, label='Accuracy')
        plt.scatter(X_labels, eth_acc)
        plt.legend()
        plt.ylabel('Loss and Accuracy')
        plt.title("Basic Ethnicity Predictor Model")
        plt.xlabel('Batchsize--LR')
        plt.savefig("Plots/base_ETH.png")
        plt.close()


    elif model == "AGE_G_GEN":
        age_loss = []
        for batch_size in batch_size_lis:
            for lr in lr_lis:
                model_path = f"Saved_Models/{model}_{batch_size}_{lr}_{dropout}.keras"
                loaded_model = tf.keras.models.load_model(model_path)
                test_loss, _ = loaded_model.evaluate([test_images, test_gender], test_age)
                age_loss.append(test_loss)

        plt.figure(figsize=(12,6))
        plt.plot(X_labels, age_loss, label='Loss') 
        plt.scatter(X_labels, age_loss)
        plt.title("AGE Predictor Given Gender")
        plt.legend()
        plt.xlabel('Batchsize--LR')
        plt.ylabel("Loss")
        plt.savefig("Plots/AGE_G_GEN.png")
        plt.close()

    elif model == "ETH_G_GEN":
        eth_acc = []
        eth_loss = []

        for batch_size in batch_size_lis:
            for lr in lr_lis:
                model_path = f"Saved_Models/{model}_{batch_size}_{lr}_{dropout}.keras"
                loaded_model = tf.keras.models.load_model(model_path)
                test_loss, test_acc = loaded_model.evaluate([test_images, test_gender], test_eth)
                eth_acc.append(test_acc)
                eth_loss.append(test_loss)

        plt.figure(figsize=(12,6))
        plt.plot(X_labels, eth_loss, label='Loss') 
        plt.scatter(X_labels, eth_loss)
        plt.plot(X_labels, eth_acc, label='Accuracy')
        plt.scatter(X_labels, eth_acc)
        plt.legend()
        plt.xlabel('Batchsize--LR')
        plt.savefig("Plots/ETH_G_GEN.png")
        plt.close()


    elif model == "Boosting_GEN":
        gen_acc = []
        gen_loss = []

        model_path = f"Saved_Models/Boosting/Boosting_GEN"
        # loaded_model = tf.keras.models.load_model(model_path)
        for bb in backbone_output:
            l=[]
            for n in n_esti:
                loaded_model = Boosting_GEN()
                loaded_model.load(f'Saved_Models/Boosting/Boosting_GEN',n, bb)
                test_acc = loaded_model.evaluate(test_images_224, test_gender)
                l.append(test_acc)
            gen_acc.append(l)

        data1 = gen_acc[0]
        data2 = gen_acc[1]
        data3 = gen_acc[2]

        ind = np.arange(len(data1))

        # Set up the figure and axis
        fig, ax = plt.subplots()

        # Bar width
        width = 0.2

        # Create bars
        rects1 = ax.bar(ind - width, data1, width, label='Features = 128')
        rects2 = ax.bar(ind, data2, width, label='Features = 256')
        rects3 = ax.bar(ind + width, data3, width, label='Features = 512')

        # Add labels, title, and legend
        ax.set_ylabel('Accuracy')
        ax.set_title('Gender Prediction Using Boosting')
        ax.set_xticks(ind)
        ax.set_xticklabels(['n_estimators = 5', 'n_estimators = 10', 'n_estimators = 15'])
        ax.legend()

        # Display the plot
        plt.savefig("Plots/Boosting_GEN.png")
        plt.close()
    
    elif model == "Boosting_ETH":
        eth_acc = []

        model_path = f"Saved_Models/Boosting/Boosting_ETH"
        # loaded_model = tf.keras.models.load_model(model_path)
        for bb in backbone_output:
            l=[]
            for n in n_esti:
                loaded_model = Boosting_ETH()
                loaded_model.load(f'Saved_Models/Boosting/Boosting_ETH',n, bb)
                test_acc = loaded_model.evaluate(test_images_224, test_eth)
                l.append(test_acc)
            eth_acc.append(l)

        data1 = eth_acc[0]
        data2 = eth_acc[1]
        data3 = eth_acc[2]

        ind = np.arange(len(data1))

        # Set up the figure and axis
        fig, ax = plt.subplots()

        # Bar width
        width = 0.2

        # Create bars
        rects1 = ax.bar(ind - width, data1, width, label='Features = 128')
        rects2 = ax.bar(ind, data2, width, label='Features = 256')
        rects3 = ax.bar(ind + width, data3, width, label='Features = 512')

        # Add labels, title, and legend
        ax.set_ylabel('Accuracy')
        ax.set_title('Ethinicity Prediction Using Boosting')
        ax.set_xticks(ind)
        ax.set_xticklabels(['n_estimators = 5', 'n_estimators = 10', 'n_estimators = 15'])
        ax.legend()

        # Display the plot
        plt.savefig("Plots/Boosting_ETH.png")
        plt.close()

    elif model == "Bagging_GEN":
        gen_acc = []

        for i in n_estimators_lis:
            loaded_model = Bagging_GEN(None, i, f'Saved_Models/Bagging/')
            predict = loaded_model.predict(test_images)
            test_acc = loaded_model.evaluate(predict, test_gender)
            gen_acc.append(test_acc)

        plt.plot(n_estimators_lis, gen_acc, label='Accuracy') 
        plt.scatter(n_estimators_lis, gen_acc)

        plt.legend()
        plt.xlabel('number of estimators')
        plt.ylabel("Accuracy")
        plt.title("Gender Predictor Using Bagging")
        plt.savefig("Plots/Bagging_GEN.png")
        plt.close()

 
    elif model == "Bagging_AGE":
        age_loss = []

        for i in n_estimators_lis:
            loaded_model = Bagging_AGE(None, i, f'Saved_Models/Bagging')
            predict = loaded_model.predict(test_images)
            test_loss = loaded_model.evaluate(predict, test_age)
            age_loss.append(test_loss)
            
        plt.plot(n_estimators_lis, age_loss, label='Loss') 
        plt.scatter(n_estimators_lis, age_loss)

        plt.legend()
        plt.xlabel('number of estimators')
        plt.ylabel("Loss")
        plt.title("Age Predictor Using Bagging")
        plt.savefig("Plots/Bagging_AGE.png")
        plt.close()

 
    elif model == "Bagging_ETH":
        eth_acc = []

        for i in n_estimators_lis:
            loaded_model = Bagging_ETH(None, i, f'Saved_Models/Bagging')
            predict = loaded_model.predict(test_images)
            test_acc = loaded_model.evaluate(predict, test_eth)
            eth_acc.append(test_acc)

        plt.plot(n_estimators_lis, eth_acc, label='Accuracy') 
        plt.scatter(n_estimators_lis, eth_acc)

        plt.legend()
        plt.xlabel('number of estimators')
        plt.ylabel("Accuracy")
        plt.title("Ethnicity Predictor Using Bagging")
        plt.savefig("Plots/Bagging_ETH.png")
        plt.close()
    
    else:
        print("Model not found")
        
