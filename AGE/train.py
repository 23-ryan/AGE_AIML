import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from Models.AGE_ETH_GEN import AGE_ETH_GEN
from Models.base_GEN import base_GEN
from Models.base_ETH import base_ETH
from Models.base_AGE import base_AGE
from Models.AGE_ETH_GEN import AGE_ETH_GEN
from Models.AGE_G_GEN import AGE_G_GEN
from Models.ETH_G_GEN import ETH_G_GEN
from Models.Bagging_GEN import Bagging_GEN
from Models.Bagging_ETH import Bagging_ETH
from Models.Bagging_AGE import Bagging_AGE
from Models.Boosting_GEN import Boosting_GEN
from Models.Boosting_ETH import Boosting_ETH
import argparse
import pickle


parser = argparse.ArgumentParser(description='use this file to train models')
parser.add_argument("--model", default="AGE_ETH_GEN", type=str)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--batch_size", default="None", type=str)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--n_estimators", default=5, type=int)
parser.add_argument("--backbone_output", default=128, type=int)
image_input_shape = (128, 128, 3)

args = parser.parse_args()
args.batch_size = eval(args.batch_size)

file_name = 'Img_Pickle/Train_Img.pickle'
with open(file_name, 'rb') as file:
    loaded_object = pickle.load(file)

X = loaded_object['X']
Age = loaded_object['Age']
Gender = loaded_object['Gender']
Eth = loaded_object['Eth']

model_path = "Saved_Models"

print(args)

if args.model == "AGE_ETH_GEN":
    model = AGE_ETH_GEN(image_input_shape=image_input_shape, lr=args.lr, batch_size= args.batch_size, dropout= args.dropout)
    model.fit(train_images=X, train_age=Age, train_gender=Gender, train_eth=Eth, epochs=args.epochs)
    model.save(f'{model_path}/{args.model}')

elif args.model == "base_AGE":
    model = base_AGE(image_input_shape=image_input_shape, lr=args.lr, batch_size= args.batch_size, dropout= args.dropout)
    model.fit(train_images=X, train_age=Age, epochs=args.epochs)
    model.save(f'{model_path}/{args.model}')

elif args.model == "base_GEN":
    model = base_GEN(image_input_shape=image_input_shape, lr=args.lr, batch_size= args.batch_size, dropout= args.dropout)
    model.fit(train_images=X, train_gender=Gender, epochs=args.epochs)
    model.save(f'{model_path}/{args.model}')

elif args.model == "base_ETH":
    model = base_ETH(image_input_shape=image_input_shape, lr=args.lr, batch_size= args.batch_size, dropout= args.dropout)
    model.fit(train_images=X, train_eth=Eth, epochs=args.epochs)
    model.save(f'{model_path}/{args.model}')
    
elif args.model == "AGE_G_GEN":
    model = AGE_G_GEN(image_input_shape=image_input_shape, lr=args.lr, batch_size= args.batch_size, dropout= args.dropout)
    model.fit(train_images=X, train_age=Age, train_gender=Gender, epochs=args.epochs)
    model.save(f'{model_path}/{args.model}')    

elif args.model == "ETH_G_GEN":
    model = ETH_G_GEN(image_input_shape=image_input_shape, lr=args.lr, batch_size= args.batch_size, dropout= args.dropout)
    model.fit(train_images=X, train_gender=Gender, train_eth=Eth, epochs=args.epochs)
    model.save(f'{model_path}/{args.model}')

elif args.model == "Boosting_GEN":
    model = Boosting_GEN()
    model.fit(n_estimators=args.n_estimators, backbone_output=args.backbone_output)
    model.save(f'Saved_Models/Boosting/Boosting_GEN', args.n_estimators, args.backbone_output)

elif args.model == "Boosting_ETH":
    model = Boosting_ETH()
    model.fit(n_estimators=args.n_estimators, backbone_output=args.backbone_output)
    model.save(f'Saved_Models/Boosting/Boosting_ETH', args.n_estimators, args.backbone_output)

elif args.model == "Bagging_GEN":
    model = Bagging_GEN(image_input_shape, args.n_estimators)
    model_path += "/Bagging"
    model.fit(train_images=X, train_gender=Gender, epochs=args.epochs, model_path=f'{model_path}/{args.model}')

elif args.model == "Bagging_ETH":
    model = Bagging_ETH(image_input_shape, args.n_estimators)
    model.fit(train_images=X, train_gender=Eth, epochs=args.epochs)
    model_path += "/Bagging"
    model.save(f'{model_path}/{args.model}')

elif args.model == "Bagging_AGE":
    model = Bagging_AGE(image_input_shape, args.n_estimators)
    model.fit(train_images=X, train_age=Age, epochs=args.epochs)
    model_path += "/Bagging"
    model.save(f'{model_path}/{args.model}')

else:
    print("No Model Found")