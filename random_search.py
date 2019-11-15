import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import datetime as dt
import pickle
from sklearn.model_selection import GridSearchCV
import random
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

with open('./features/ssy_fea.pickle', 'rb') as f:
    ssy_fea = pickle.load(f)
    
with open('ssy_df.pickle', 'rb') as f:
    ssy_df = pickle.load(f)

train_x = ssy_df[ssy_fea]
train_y = ssy_df['收缩压']

params = {
#     'learin_rate':my_iter(0.01,0.15)
    'learning_rate': sp_uniform(0.01,0.14),
    'colsample_bytree':sp_uniform(0.6,0.4),
    'subsample':sp_uniform(0.6,0.4),
    'subsample_freq': sp_randint(1,8),
    'num_leaves':sp_randint(35,85),
    'min_child_samples':sp_randint(5,20)
    
}

def RandomSearch(clf,params,x,y):
    rscv = RandomizedSearchCV(clf,params,scoring='neg_mean_squared_error',cv=5,n_jobs=-1,verbose=10,n_iter=5000)
    rscv.fit(x,y)
    
    print(rscv.cv_results_)
    print(rscv.best_params_)


gbm = lgbm.LGBMRegressor(boosting_type='gbdt',
                        objective='mse',
                        n_estimators=100
#                         early_stopping_rounds=100
                        )


start = time.time()
RandomSearch(gbm,params,train_x,train_y)
print('RandomSearch一共用时：{}'.format(time.time()-start))
