# enconding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold,RepeatedKFold
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline

import time
import warnings

warnings.filterwarnings("ignore")

train_file = '../pre_data/training.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('id')
feature_list.remove('label')

training = data_set[feature_list].values

test_data = pickle.load(open('../pre_data/eval.pkl','rb'))
test_data.fillna(0.,inplace=True)
sub_df = test_data['id'].copy()

del test_data['id']
test_data = test_data.values


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
folds = KFold(n_splits=5, shuffle=True, random_state=2020)#5折交叉验证
oof_lgb = np.zeros(len(label))#存放训练集的预测结果
predictions_lgb = np.zeros(len(sub_df))#存放测试集的预测结果

X_train = training
y_train = label
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("lgb fold n°{}".format(fold_+1))
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

print("lgb CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, label)))

## xgb
xgb_params = {'eta': 0.005, 
              'max_depth': 10, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': 4}#xgb的参数，可以自己改
folds = KFold(n_splits=5, shuffle=True, random_state=2020)#5折交叉验证
oof_xgb = np.zeros(len(label))#用于存放训练集的预测
predictions_xgb = np.zeros(len(sub_df))#用于存放测试集的预测

X_train = training
y_train = label
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("xgb fold n°{}".format(fold_+1))
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

print("xgb CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, label)))

## logic regression
folds = KFold(n_splits=5, shuffle=True, random_state=2020)#5折交叉验证
oof_lr = np.zeros(len(label))#用于存放训练集的预测
predictions_lr = np.zeros(len(sub_df))#用于存放测试集的预测

X_train = training
y_train = label
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("xgb fold n°{}".format(fold_+1))
    trn_data_x = X_train[trn_idx]
    trn_data_y = y_train[trn_idx]
    val_data_x = X_train[val_idx]
    val_data_y = y_train[val_idx]

    clf = LinearRegression()
    clf.fit(trn_data_x, trn_data_y)
    
    oof_lr[val_idx] = clf.predict(val_data_x)
    predictions_lr += clf.predict(test_data) / folds.n_splits
    
    print("===========oof_lr Result============")
    print(oof_lr)
    print("===========predictions_lr Result============")
    print(predictions_lr)

print("lr CV score: {:<8.8f}".format(mean_squared_error(oof_lr, label)))


# 将lgb和xgb的结果进行stacking（叠加）
train_stack = np.vstack([oof_lgb,oof_xgb, oof_lr]).transpose()#训练集2列特征
test_stack = np.vstack([predictions_lgb, predictions_xgb, predictions_lr]).transpose()#测试集2列特征

#贝叶斯分类器也使用交叉验证的方法，5折，重复2次，主要是避免过拟合
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2020)
oof_stack = np.zeros(train_stack.shape[0])#存放训练集中验证集的预测结果
predictions = np.zeros(test_stack.shape[0])#存放测试集的预测结果

#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,label)):#label就是每一行样本的标签值
    print("stacking fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], label[trn_idx]#划分训练集的80%
    val_data, val_y = train_stack[val_idx], label[val_idx]#划分训练集的20%做验证集
    
<<<<<<< HEAD
    clf_3 = LinearRegression()
=======
    clf_3 = BayesianRidge()
    # from sklearn.linear_model import LogisticRegression

>>>>>>> 5db1492b2de5bf0d07ee941f019a8cc57921177e
    clf_3.fit(trn_data, trn_y)#贝叶斯训练过程，sklearn中的。
    
    oof_stack[val_idx] = clf_3.predict(val_data)#对验证集有一个预测，用于后面计算模型的偏差
    predictions += clf_3.predict(test_stack) / 10#对测试集的预测，除以10是因为5折交叉验证重复了2次

    print("===========stacking oof_stack Result============")
    print(oof_stack)
    print("===========stacking predictions Result============")
    print(predictions)


mean_squared_error(label, oof_stack)#计算出模型在训练集上的均方误差
print("CV score: {:<8.8f}".format(mean_squared_error(label, oof_stack)))


print(predictions.shape)
print(predictions)
# pred = np.mean(predictions,axis=0)
pcol =pd.DataFrame(list(predictions))
sub_df = pd.concat([sub_df, pcol], axis=1)
sub_df.to_csv('submission.csv',sep=',',header=False,index=False,encoding='utf8')