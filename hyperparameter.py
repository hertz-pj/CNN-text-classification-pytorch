# -*- encoding: utf-8 -*-
'''
@File    :   hyperparameter.py
@Time    :   2019/06/13
@Author  :   peiji
@Contact :   peiji.yang@foxmail.com
'''
EPOCH           = 100
BATCH_SIZE      = 100
MAX_SENT_LEN    = 512
WORD_DIM        = 300
VOCAB_SIZE      = -1
CLASS_SIZE      = -1
FILTERS         = [3,4,5]
FILTER_NUM      = [100, 100, 100]
DROPOUT_PROB    = 0.5
IN_CHANNEL      = 1
LEARNING_RATE   = 1
GPU             = 1
NORM_LIMIT      = 3
TEST_BATCH_SIZE = 1000

# the file
data_dir        = "./data/label_v1"
model_save_dir  = "./save_models"
vocab_file      = "./vocab.txt"