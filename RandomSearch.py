import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
import gc
gc.enable()
import random as rand

print("Reading data sets....")
train_df = pd.read_csv("input/train.csv")

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']

from sklearn.model_selection import RandomizedSearchCV
from random import choice
import random as rand
from sklearn.model_selection import train_test_split
n_iter = 60
best_model_score = 0 
best_params = 0
def get_random(min_value,nr_range):
    return rand.random()*nr_range + min_value
for model_nr in range(n_iter):
    param = {
        'bagging_freq': rand.randint(0,12),
        'bagging_fraction': get_random(0.2,0.3),
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': get_random(0.01,0.2),
        'learning_rate': get_random(0.005,0.01),
        'max_depth':choice([-1,3,6,5,10,13]),  
        'metric':'auc',
        'min_data_in_leaf': rand.randint(10,130),
        'min_sum_hessian_in_leaf': get_random(5,5),
        'num_leaves': rand.randint(3,15),
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': -1,
    }
    print("checking model {} with param \n {} ".format(model_nr,param))
    
    X_t, X_valid,y_t , y_valid = train_test_split(train_df[features].values, target.values, test_size=0.2, random_state=42)
    
    trn_data = lgb.Dataset(X_t, label=y_t)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=2000, early_stopping_rounds = 3000)
    
    model_score = clf.best_score['valid_1']['auc']
    if model_score > best_model_score:
        print("Updating score from {} to {}".format(best_model_score,model_score))
        best_params = param
        best_model_score = model_score
    else:
        print("Not updating {} - model score {}".format(best_model_score,model_score))
    gc.collect()   
        
print("Best score {} for model {}".format(best_model_score,best_params))

