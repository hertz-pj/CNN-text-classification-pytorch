# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2019/06/13
@Author  :   peiji
@Contact :   peiji.yang@foxmail.com
'''

from torch.autograd import Variable
# from model import CNN
from model import CNN
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
# from gensim.models.keyedvectors import KeyedVectors
from util import *

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import argparse
import copy 
import hyperparameter
import os 


def train(data):
    """[summary]
    
    Args:
        data ({}): { name :
            train_x, dev_x, test_x
            train_y, dev_y, test_y
            classes
            word_to_idx
        }
    """

    model = CNN().cuda(hyperparameter.GPU)
    
    # optimizer and the loss function
    optimizer = optim.Adadelta(model.parameters(), lr=hyperparameter.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0

    for epoch in range(hyperparameter.EPOCH):
        # shuffle the dataset
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        # get batch data 
        for i in range(0, len(data["train_x"]), hyperparameter.BATCH_SIZE):
            batch_size = min(hyperparameter.BATCH_SIZE, len(data["train_x"])-i)

            batch_x = []
            for sent in data["train_x"][i:i+batch_size]:
                sent = sent if len(sent) < hyperparameter.MAX_SENT_LEN else sent[:hyperparameter.MAX_SENT_LEN]
                batch_x.append([data["word_to_idx"][w] if w in data["word_to_idx"]
                        else hyperparameter.VOCAB_SIZE for w in sent]+
                        (hyperparameter.MAX_SENT_LEN-len(sent))*[hyperparameter.VOCAB_SIZE+1])

            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i+batch_size]]
            batch_x = torch.LongTensor(batch_x).cuda(hyperparameter.GPU)
            batch_y = torch.LongTensor(batch_y).cuda(hyperparameter.GPU)

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=hyperparameter.NORM_LIMIT)
            optimizer.step()
        
        dev_acc = test(model, data)
        print(f"epoch:{epoch}\tdev acc:{dev_acc}\tloss:{loss.item()}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model = copy.deepcopy(model)
    test_acc = test(best_model, data, "test")
    print(f"best_dev_acc:{best_dev_acc}\ttest_acc:{test_acc}")

def test(model, data, mode="dev"):
    """[summary]
    
    Args:
        model ([type]): [description]
        data ([type]): [description]
        mode (str, optional): [description]. Defaults to "dev".
        
    """

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    model.eval()

    pred = []
    for i in range(0, len(x), hyperparameter.TEST_BATCH_SIZE):
        batch_size = min(hyperparameter.TEST_BATCH_SIZE, len(x)-i)

        batch_x = []
        for sent in x[i:i+batch_size]:
            sent = sent if len(sent) < hyperparameter.MAX_SENT_LEN else sent[:hyperparameter.MAX_SENT_LEN]
            batch_x.append([data["word_to_idx"][w] if w in data["word_to_idx"]
                    else hyperparameter.VOCAB_SIZE for w in sent] + 
                    (hyperparameter.MAX_SENT_LEN-len(sent))*[hyperparameter.VOCAB_SIZE+1])
        batch_x = torch.LongTensor(batch_x).cuda(hyperparameter.GPU)
        pred += np.argmax(model(batch_x).cpu().data.numpy(), axis=1).tolist()
    
    y = [data["classes"].index(c) for c in y]
    assert len(y) == len(pred), "the dev_y is different from pred"

    return accuracy_score(y, pred)
            

def main():

    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_predict", action="store_true", help="whether to predict")
    parser.add_argument("--build_vocab", action="store_true", help="whether to build a new vocab")
    
    args = parser.parse_args()

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    
    if hyperparameter.data_dir == None:
        raise ValueError("The path of the data_dir is None")
    data = read_data_cn(hyperparameter.data_dir)
    
    # build a vocab file from train_data if there is no vocab file
    if os.path.exists(hyperparameter.vocab_file) and not args.build_vocab:
        with open(hyperparameter.vocab_file, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f.readlines()]
    else:
        vocab = sorted(list(set([w for sent in data["train_x"] for w in sent])))
        with open(hyperparameter.vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab))
        print(f"vocab file write in {hyperparameter.vocab_file}")
    
    hyperparameter.VOCAB_SIZE = len(vocab)
    data["word_to_idx"] = {w:i for i,w in enumerate(vocab)}
    data["classes"] = sorted(list(set(data["train_y"])))
    hyperparameter.CLASS_SIZE = len(data["classes"])

    print("="*20 + "INFORMATION" + "="*20)
    print("EPOCH:", hyperparameter.EPOCH)
    print("BATCH_SIZE:", hyperparameter.BATCH_SIZE)
    print("LEARNING_RATE:", hyperparameter.LEARNING_RATE)
    print("VOCAB_SIZE:", hyperparameter.VOCAB_SIZE)
    print("CLASS_NUM:", hyperparameter.CLASS_SIZE)
    print("=" * 20 + "INFORMATION" + "=" * 20)

    
    if args.do_train:
        print("="*20 + "TRAINING STARTED" + "="*20)
        model = train(data)
    # if args.do_predict:
    #     print("="*20 + "PREDICTING STARTED" + "="*20)
    #     model = test(data, params)


if __name__ == "__main__":
    main()
    pass



    


    

    

    


