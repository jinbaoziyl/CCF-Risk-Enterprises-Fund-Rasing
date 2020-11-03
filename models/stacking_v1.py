# -*- coding: utf-8 -*-
from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
import pandas as pd
import xgboost as xgb
import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

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

def xgb_feature(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'reg:linear',
              'eval_metric' : 'rmse',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1111,
            #   'silent': True, 
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def xgb_feature2(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'reg:linear',
              'eval_metric' : 'rmse',
              'eta': 0.015,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 11,
            #   'silent': True, 
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=1200)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def xgb_feature3(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'reg:linear',
              'eval_metric' : 'rmse',
              'eta': 0.01,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1,
            #   'silent': True, 
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=2000)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)


def et_model(X_train, y_train, X_test, y_test=None):
    model = ExtraTreesClassifier(max_features = 'log2', n_estimators = 1000 , n_jobs = -1).fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def gbdt_model(X_train, y_train, X_test, y_test=None):
    model = GradientBoostingClassifier(learning_rate = 0.02, max_features = 0.7, n_estimators = 700 , max_depth = 5).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def logistic_model(X_train, y_train, X_test, y_test=None):
    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def lgb_feature(X_train, y_train, X_test, y_test=None):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric':'mse',
        'num_leaves': 25,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf':5,
        'max_bin':200,
        'verbose': 0,
    }
    gbm = lgb.train(params,
    lgb_train,
    num_boost_round=2000)
    predict = gbm.predict(X_test)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

lgb_dataset = Dataset(training,label,test_data,use_cache=False)
xgb_dataset = Dataset(X_train=training,y_train=label,X_test=test_data,y_test=None,use_cache=False)
#heamy
model_xgb = Regressor(dataset=xgb_dataset, estimator=xgb_feature,name='xgb',use_cache=False)
model_xgb2 = Regressor(dataset=xgb_dataset, estimator=xgb_feature2,name='xgb2',use_cache=False)
model_xgb3 = Regressor(dataset=xgb_dataset, estimator=xgb_feature3,name='xgb3',use_cache=False)
model_lgb = Regressor(dataset=lgb_dataset, estimator=lgb_feature,name='lgb',use_cache=False)
model_gbdt = Regressor(dataset=xgb_dataset, estimator=gbdt_model,name='gbdt',use_cache=False)
pipeline = ModelsPipeline(model_xgb,model_xgb2,model_xgb3,model_lgb,model_gbdt)
stack_ds = pipeline.stack(k=5, seed=111, add_diff=False, full_test=True)
stacker = Regressor(dataset=stack_ds, estimator=LinearRegression,parameters={'fit_intercept': False})
predict_result = stacker.predict()

print(predict_result)
pcol =pd.DataFrame(list(predict_result))
sub_df = pd.concat([sub_df, pcol], axis=1)
sub_df.to_csv('submission.csv',sep=',',header=False,index=False,encoding='utf8')

