import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import datetime as dt
import os
# os.chdir(os.path.dirname(os.getcwd()))
import pickle
from sklearn.model_selection import GridSearchCV
import pickle

# 读取数据
ssy_df = pd.read_csv('./data_new/train_data_收缩压.csv',low_memory=False)


# 特征
with open('./features/szy_fea.pickle', 'rb') as f:
    szy_fea = pickle.load(f)


train_x = ssy_df[ssy_fea]
train_y = ssy_df['收缩压']


params = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'feature_fraction':[0.6, 0.7, 0.8, 0.9, 0.95],
    'bagging_fraction':[0.6, 0.7, 0.8, 0.9, 0.95],
    'bagging_freq': [1, 3, 5, 7, 8],
    'num_leaves':[35,45,55,65,75,85],
    'min_data_in_leaf':[5,8,11,14,17,20]
}



gbm = lgbm.LGBMRegressor(boosting_type='gbdt',
                        objective='mse',
                        n_estimators=80
#                         early_stopping_rounds=100
                        )


start = time.time()
gsearch = GridSearchCV(gbm,param_grid=params,scoring='neg_mean_squared_error',
	cv=5,verbose=20,n_jobs=-1)
gsearch.fit(train_x,train_y)
print('GridCV一共用时：',time.time()-start)