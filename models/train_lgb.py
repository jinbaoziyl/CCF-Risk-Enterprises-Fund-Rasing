# enconding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
# import xgboost as xgb
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

##### lgb
# param = {'num_leaves': 120,
#          'min_data_in_leaf': 30, 
#          'objective':'regression',
#          'max_depth': -1,
#          'learning_rate': 0.01,
#          "min_child_samples": 30,
#          "boosting": "gbdt",
#          "feature_fraction": 0.9,
#          "bagging_freq": 1,
#          "bagging_fraction": 0.9 ,
#          "bagging_seed": 11,
#          "metric": 'mse',
#          "lambda_l1": 0.1,
#          "verbosity": -1}#模型参数，可以修改
# folds = KFold(n_splits=5, shuffle=True, random_state=2018)#5折交叉验证
# oof_lgb = np.zeros(len(label))#存放训练集的预测结果
# predictions_lgb = np.zeros(len(sub_df))#存放测试集的预测结果

# for fold_, (trn_idx, val_idx) in enumerate(folds.split(training, label)):
#     X_train, y_train, X_val, y_val = training[trn_idx], label[trn_idx],training[val_idx],label[val_idx]

#     trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])#80%的训练集用于训练
#     val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])#20%的训练集做验证集

#     num_round = 10000
#     clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)#训练过程
#     oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)#对验证集得到预测结果
#     predictions_lgb += clf.predict(test_data, num_iteration=clf.best_iteration) / folds.n_splits#对测试集5次取平均值
    
#     lgb_pred =pd.DataFrame(predictions_lgb)
#     lgb_pred.to_csv('submission_lgb.csv',sep=',',header=False,index=False,encoding='utf8')

# print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, label)))


##### lgb
param = {'num_leaves': 120,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}#模型参数，可以修改
folds = KFold(n_splits=5, shuffle=True, random_state=2018)#5折交叉验证
oof_lgb = np.zeros(len(label))#存放训练集的预测结果
predictions_lgb = np.zeros(len(sub_df))#存放测试集的预测结果

X_train = training
y_train = label
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])#80%的训练集用于训练
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])#20%的训练集做验证集

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)#训练过程
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)#对验证集得到预测结果
    predictions_lgb += clf.predict(test_data, num_iteration=clf.best_iteration) / folds.n_splits#对测试集5次取平均值

    print("===========oof_lgb Result============")
    print(oof_lgb)
    print("===========predictions_lgb Result============")
    print(predictions_lgb)

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, label)))