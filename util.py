# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2019/06/13
@Author  :   peiji
@Contact :   peiji.yang@foxmail.com
'''


from sklearn.utils import shuffle
import hyperparameter
import os 
import pickle

def read_data(data_dir):
    data = {}

    def read(mode):
        x, y = [], []

        with open(os.path.join(data_dir, mode+".txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()

        x = [line.split('\t')[0].strip().split() for line in lines]
        y = [line.split('\t')[1].strip() for line in lines]

        data[mode + "_x"] = x
        data[mode + "_y"] = y
    
    read("train")
    read("dev")
    read("test")

    print(f"size of train set{len(data['train_x'])}")

    return data

def read_data_cn(data_dir):
    data = {}

    def read(mode):
        x, y = [], []

        with open(os.path.join(data_dir, mode+".tsv"), "r", encoding="utf-8") as f:
            lines = f.readlines()

        x = [line.strip().split('\t')[0] for line in lines]
        y = [line.strip().split('\t')[1] for line in lines]

        data[mode + "_x"] = x
        data[mode + "_y"] = y
    
    read("train")
    read("dev")
    read("test")

    print(f"size of train set{len(data['train_x'])}")

    return data

def save_model(model):
    
    path = f"{hyperparameter.model_save_dir}/CNN_{hyperparameter.EPOCH}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")

def load_model(params):    
    path = f"{hyperparameter.model_save_dir}/CNN_{hyperparameter.EPOCH}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}")
        exit()

