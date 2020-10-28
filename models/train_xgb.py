# enconding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
import time
import warnings

warnings.filterwarnings("ignore")

train_file = './pre_data/training.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('id')
feature_list.remove('label')

training = data_set[feature_list].values

test_data = pickle.load(open('./pre_data/eval.pkl','rb'))
test_data.fillna(0.,inplace=True)
sub_df = test_data['id'].copy()

del test_data['id']
test_data = test_data.values

##### xgb
xgb_params = {'eta': 0.005, 
              'max_depth': 10, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': 4}#xgb的参数，可以自己改
folds = KFold(n_splits=5, shuffle=True, random_state=2018)#5折交叉验证
oof_xgb = np.zeros(len(label))#用于存放训练集的预测
predictions_xgb = np.zeros(len(sub_df))#用于存放测试集的预测

X_train = training
y_train = label
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])#训练集的80%
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])#训练集的20%，验证集
 
    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)#80%用于训练过程
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)#预测20%的验证集
    predictions_xgb += clf.predict(xgb.DMatrix(test_data), ntree_limit=clf.best_ntree_limit) / folds.n_splits#预测测试集，并且取平均
    
    print("===========oof_xgb Result============")
    print(oof_xgb)
    print("===========predictions_xgb Result============")
    print(predictions_xgb)

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, label)))