# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2019/06/12
@Author  :   peiji
@Contact :   peiji.yang@foxmail.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparameter

class CNN(nn.Module):

    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        
        # import the hyperparameter
        self.BATCH_SIZE     = hyperparameter.BATCH_SIZE
        self.MAX_SENT_LEN   = hyperparameter.MAX_SENT_LEN
        self.WORD_DIM       = hyperparameter.WORD_DIM
        self.VOCAB_SIZE     = hyperparameter.VOCAB_SIZE
        self.CLASS_SIZE     = hyperparameter.CLASS_SIZE
        self.FILTERS        = hyperparameter.FILTERS
        self.FILTER_NUM     = hyperparameter.FILTER_NUM
        self.DROPOUT_PROB   = hyperparameter.DROPOUT_PROB
        self.IN_CHANNEL     = 1

        # the number of the FILTERS must be same as FILTER_NUM
        assert len(self.FILTERS) == len(self.FILTER_NUM)

        # build the word embedding 
        self.embedding = nn.Embedding(self.VOCAB_SIZE+2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE+1)

        # convolutional layer
        conv_dict = {}
        for i in range(len(self.FILTERS)):
            conv_dict[str(i)] = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM*self.FILTERS[i],
                            stride=self.WORD_DIM)
        self.conv = nn.ModuleDict(conv_dict)

        self.drop = nn.Dropout(p=self.DROPOUT_PROB)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
    
    def forward(self, inp):
        
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)        
        z = []

        # max pooling layer 
        for i in range(len(self.FILTERS)):
            c = F.relu(self.conv[str(i)](x))
            c_hat = F.max_pool1d(c, self.MAX_SENT_LEN-self.FILTERS[i]+1).view(-1, self.FILTER_NUM[i])
            z.append(c_hat)

        x = torch.cat(z, 1)
        x = self.drop(x)

        return self.fc(x)
        