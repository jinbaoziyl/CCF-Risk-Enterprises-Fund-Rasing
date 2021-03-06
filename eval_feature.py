import pandas as pd 
import numpy as np 
import pickle
import os
from util import *
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder

t_base_info = "./dataset/train/base_info.csv"
t_annual_report_info = "./dataset/train/annual_report_info.csv"
t_tax_info = "./dataset/train/tax_info.csv"
t_change_info = "./dataset/train/change_info.csv"
t_news_info = "./dataset/train/news_info.csv"
t_other_info = "./dataset/train/other_info.csv"
t_entprise_info = "./dataset/entprise_evaluate.csv" # train labeled data

is_update = True

if not os.path.exists("./pre_data"):
    os.mkdir("./pre_data")

def gen_base_feat_v1():
    dump_path = "./pre_data/train_base_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        base_info = pickle.load(open(dump_path, 'rb'))
    else:
        base_info = pd.read_csv(t_base_info, header=0)

        del base_info['ptbusscope']
        del base_info['midpreindcode']
        del base_info['dom']
        del base_info['opscope']  # 后面可以分词后用 svd 抽取特征, 但感觉效果不大

        base_info['opfrom'] = pd.to_datetime(base_info.opfrom)
        base_info['opfrom_year'] = base_info['opfrom'].dt.year.astype('int')

        base_info['opto'] = pd.to_datetime(base_info.opto)
        base_info['opto_year'] = base_info['opto'].dt.year.fillna(-1).astype('int')

        del base_info['opfrom']
        del base_info['opto']

        for col in tqdm(['industryco', 'enttypeitem', 'compform', 'venind', 
                        'enttypeminu', 'protype']):
            base_info[col] = base_info[col].fillna(-1).astype('int')

        for col in tqdm(base_info.select_dtypes(['float64']).columns):
            base_info[col] = base_info[col].fillna(base_info[col].median())

        base_info['opform'] = base_info['opform'].replace('01', '01-以个人财产出资').replace('02', '02-以家庭共有财产作为个人出资')

        # 数据比较长尾, label encoding 和 freq 处理
        for col in tqdm(['industryphy', 'opform', 'oploc', 'orgid', 'jobid', 'oplocdistrict',
                        'enttypegb', 'industryco', 'enttype', 'enttypeitem']):
            lbl = LabelEncoder()
            base_info[col] = lbl.fit_transform(base_info[col].astype(str))
            vc = base_info[col].value_counts(dropna=True, normalize=True).to_dict()
            base_info[f'{col}_freq'] = base_info[col].map(vc)
        del base_info['forreccap']
        del base_info['forregcap']
        del base_info['protype']
        del base_info['congro']
    return base_info

def gen_base_feat():
    dump_path = "./pre_data/eval_base_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_base_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_base_info = pd.read_csv(t_base_info, header=0)
        df_base_info['oplocdistrict'].value_counts() 
        dict_oplocdistrict = {}
        cnt = 0
        for e in df_base_info['oplocdistrict'].unique():
            if e in dict_oplocdistrict: continue
            else:
                dict_oplocdistrict[e] = cnt 
                cnt += 1
        df_base_info['oplocdistrict'] = df_base_info['oplocdistrict'].map(dict_oplocdistrict)

        df_base_info['industryphy'].value_counts() 
        dict_industryphy = {}
        cnt = 0
        for e in df_base_info['industryphy'].unique():
            if e in dict_industryphy: continue
            else:
                dict_industryphy[e] = cnt 
                cnt += 1
    
        df_base_info['industryphy'] = df_base_info['industryphy'].map(dict_industryphy)
        df_base_info['industryco'].value_counts() 
        dict_industryco = {}
        cnt = 0
        for e in df_base_info['industryco'].unique():
            if e in dict_industryco: continue
            else:
                dict_industryco[e] = cnt 
                cnt += 1
        df_base_info['industryco'] = df_base_info['industryco'].map(dict_industryco)

        df_base_info['dom'].value_counts() 
        del df_base_info['dom']
        del df_base_info['opscope']

        df_base_info['enttype'].value_counts() 
        dict_enttype = {}
        cnt = 0
        for e in df_base_info['enttype'].unique():
            if e in dict_enttype: continue
            else:
                dict_enttype[e] = cnt 
                cnt += 1
        df_base_info['enttype'] = df_base_info['enttype'].map(dict_enttype)
     
        df_base_info['enttypeitem'].value_counts() 
        dict_enttypeitem = {}
        cnt = 0
        for e in df_base_info['enttypeitem'].unique():
            if e in dict_enttypeitem: continue
            else:
                dict_enttypeitem[e] = cnt 
                cnt += 1
        df_base_info['enttypeitem'] = df_base_info['enttypeitem'].map(dict_enttypeitem)

        del df_base_info['opfrom']
        del df_base_info['opto']

        df_base_info['state'].value_counts()
        dict_state = {}
        cnt = 0
        for e in df_base_info['state'].unique():
            if e in dict_state: continue
            else:
                dict_state[e] = cnt 
                cnt += 1
        df_base_info['state'] = df_base_info['state'].map(dict_state)

        del df_base_info['orgid']
        del df_base_info['jobid']

        dict_opform = {}
        cnt = 0
        for e in df_base_info['opform'].unique():
            if e in dict_opform: continue
            else:
                dict_opform[e] = cnt 
                cnt += 1
        df_base_info['opform'] = df_base_info['opform'].map(dict_opform)
        del df_base_info['ptbusscope']
        dict_enttypeminu = {}
        cnt = 0
        for e in df_base_info['enttypeminu'].unique():
            if e in dict_enttypeminu: continue
            else:
                dict_enttypeminu[e] = cnt 
                cnt += 1

        df_base_info['enttypeminu'] = df_base_info['enttypeminu'].map(dict_enttypeminu)
        del df_base_info['midpreindcode']
        del df_base_info['protype']
        del df_base_info['oploc']
        del df_base_info['enttypegb']

        pickle.dump(df_base_info, open(dump_path, 'wb'))
    return df_base_info

def gen_anreport_feat():
    dump_path = "./pre_data/eval_anreport_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_anreport_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_anreport_info = pd.read_csv(t_annual_report_info, header=0)
        dict_year = {"2015.0": 2015, "2016.0": 2016, "2017.0": 2017, "2018.0": 2018}
        df_anreport_info['ANCHEYEAR'] = df_anreport_info['ANCHEYEAR'].map(lambda x : dict_year[str(x)])
        # del df_anreport_info['MEMNUM']
        # del df_anreport_info['FARNUM']
        # del df_anreport_info['ANNNEWMEMNUM']
        # del df_anreport_info['ANNREDMEMNUM']

        # df_anreport_info['BUSSTNAME'].value_counts() 
        # dict_bsnm = { "开业": 1, "歇业": 2, "停业": 3, "清算": 4}
        # df_anreport_info['BUSSTNAME'] = df_anreport_info['BUSSTNAME'].map(dict_bsnm)
        # df_anreport_info['BUSSTNAME'].fillna(0, inplace=True)
        df_af = pd.get_dummies(df_anreport_info['BUSSTNAME'], prefix='busstname')  
        df_anreport_info = pd.concat([df_anreport_info, df_af], axis=1)
        del df_anreport_info['BUSSTNAME']

        df_anreport_info = df_anreport_info.groupby(['id'], as_index=False).sum()

        pickle.dump(df_anreport_info, open(dump_path, 'wb'))
    return df_anreport_info

def gen_tax_feat_v1():
    dump_path = "./pre_data/train_tax_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_tax_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_tax_info = pd.read_csv(t_tax_info, header=0)
        df_tax_info['TAX_DAYS'] = df_tax_info[['START_DATE', 'END_DATE']].apply(lambda x : days_v1(x['END_DATE'], x['START_DATE']), axis=1)

        del df_tax_info['START_DATE']
        del df_tax_info['END_DATE']

        df_tax_info_cp = df_tax_info.copy()
        df_tax_amount_cp = df_tax_info_cp[['id', 'TAX_DAYS', 'TAX_RATE', 'DEDUCTION', 'TAX_AMOUNT', 'TAXATION_BASIS']].groupby(['id'], as_index=False).sum()

        # 加上平均
        df_tax_amount_cp['arg_tax_rate'] =  df_tax_amount_cp['TAX_RATE'] / df_tax_amount_cp['TAX_DAYS']
        df_tax_amount_cp['arg_tax_deduction'] =  df_tax_amount_cp['DEDUCTION'] / df_tax_amount_cp['TAX_DAYS']
        df_tax_amount_cp['arg_tax_amount'] =  df_tax_amount_cp['TAX_AMOUNT'] / df_tax_amount_cp['TAX_DAYS']
        df_tax_amount_cp['arg_tax_basis'] =  df_tax_amount_cp['TAXATION_BASIS'] / df_tax_amount_cp['TAX_DAYS']

        del df_tax_amount_cp['TAX_DAYS']

        pickle.dump(df_tax_amount_cp, open(dump_path, 'wb'))
    return df_tax_amount_cp

def gen_tax_feat():
    dump_path = "./pre_data/eval_tax_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_tax_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_tax_info = pd.read_csv(t_tax_info, header=0)
        df_tax_info['TAX_DAYS'] = df_tax_info[['START_DATE', 'END_DATE']].apply(lambda x : days_v1(x['END_DATE'], x['START_DATE']), axis=1)

        del df_tax_info['START_DATE']
        del df_tax_info['END_DATE']

        tax_cg_dict = {}
        cnt = 0
        for e in df_tax_info['TAX_CATEGORIES'].unique():
            if e in tax_cg_dict: continue
            else:
                tax_cg_dict[e] = cnt 
                cnt += 1
        # print("categories size, ", cnt)
        # print(tax_cg_dict)

        df_tax_info['TAX_CATEGORIES'] = df_tax_info['TAX_CATEGORIES'].map(tax_cg_dict)

        tax_it_dict = {}
        cnt = 0
        for e in df_tax_info['TAX_ITEMS'].unique():
            if e in tax_it_dict: continue
            else:
                tax_it_dict[e] = cnt 
                cnt += 1
 
        df_tax_info['TAX_ITEMS'] = df_tax_info['TAX_ITEMS'].map(tax_it_dict)
        # df_tax_info['TAXATION_BASIS'] = df_tax_info['TAXATION_BASIS'].apply(np.log)
        # del df_tax_info['TAXATION_BASIS']

        df_tax_extra = df_tax_info.copy()
        df_tax_amount_extra = df_tax_extra[['id', 'TAX_DAYS', 'TAX_RATE', 'DEDUCTION', 'TAX_AMOUNT', 'TAXATION_BASIS']].groupby(['id'], as_index=False).sum()


        df_tax_info_cp = df_tax_info.copy()
        df_tax_amount_cp = df_tax_info_cp[['id', 'DEDUCTION', 'TAX_AMOUNT', 'TAXATION_BASIS']].groupby(['id'], as_index=False).sum()

        df_tax_cg_info_cp = df_tax_info.copy()
        df_tax_cg_amount_cp = df_tax_cg_info_cp[['id','TAX_CATEGORIES', 'DEDUCTION', 'TAX_AMOUNT']].groupby(['id','TAX_CATEGORIES'], as_index=False).sum()

        df_tax_cg_amount_cp = df_tax_cg_amount_cp.pivot(index='id', columns='TAX_CATEGORIES', values=['DEDUCTION', 'TAX_AMOUNT']).reset_index()
        len_deduction = len(df_tax_cg_amount_cp['DEDUCTION'].columns)
        len_tax_amount = len(df_tax_cg_amount_cp['TAX_AMOUNT'].columns)

        for i in range(len_deduction):
            df_tmp = pd.DataFrame()
            df_tmp.loc[:,'cg_deduction'+str(i)] = df_tax_cg_amount_cp['DEDUCTION', i]
            df_tax_amount_cp = pd.concat([df_tax_amount_cp, df_tmp], axis=1)

        for i in range(len_tax_amount):
            df_tmp = pd.DataFrame()
            df_tmp.loc[:,'cg_tax_amount'+str(i)] = df_tax_cg_amount_cp['TAX_AMOUNT', i]
            df_tax_amount_cp = pd.concat([df_tax_amount_cp, df_tmp], axis=1)


        df_tax_it_info_cp = df_tax_info.copy()
        df_tax_it_amount_cp = df_tax_cg_info_cp[['id','TAX_ITEMS', 'DEDUCTION', 'TAX_AMOUNT']].groupby(['id','TAX_ITEMS'], as_index=False).sum()

        df_tax_it_amount_cp = df_tax_it_amount_cp.pivot(index='id', columns='TAX_ITEMS', values=['DEDUCTION', 'TAX_AMOUNT']).reset_index()
        len_deduction = len(df_tax_it_amount_cp['DEDUCTION'].columns)
        len_tax_amount = len(df_tax_it_amount_cp['TAX_AMOUNT'].columns)

        for i in range(len_deduction):
            df_tmp = pd.DataFrame()
            df_tmp.loc[:,'it_deduction'+str(i)] = df_tax_it_amount_cp['DEDUCTION', i]
            df_tax_amount_cp = pd.concat([df_tax_amount_cp, df_tmp], axis=1)

        for i in range(len_tax_amount):
            df_tmp = pd.DataFrame()
            df_tmp.loc[:,'it_tax_amount'+str(i)] = df_tax_it_amount_cp['TAX_AMOUNT', i]
            df_tax_amount_cp = pd.concat([df_tax_amount_cp, df_tmp], axis=1)

       # 加上平均
        df_tax_amount_cp['arg_tax_rate'] =  df_tax_amount_extra['TAX_RATE'] / df_tax_amount_extra['TAX_DAYS']
        df_tax_amount_cp['arg_tax_deduction'] =  df_tax_amount_extra['DEDUCTION'] / df_tax_amount_extra['TAX_DAYS']
        df_tax_amount_cp['arg_tax_amount'] =  df_tax_amount_extra['TAX_AMOUNT'] / df_tax_amount_extra['TAX_DAYS']
        df_tax_amount_cp['arg_tax_basis'] =  df_tax_amount_extra['TAXATION_BASIS'] / df_tax_amount_extra['TAX_DAYS']

        pickle.dump(df_tax_amount_cp, open(dump_path, 'wb'))
    return df_tax_amount_cp

def gen_change_feat():
    dump_path = "./pre_data/eval_change_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_change_info = pickle.load(open(dump_path, 'rb'))
    else:
        # TODO: 变更信息如何处理  这里直接删除 只保留变更时间 #
        # df_change_info = pd.read_csv(t_change_info, header=0)
        # # del df_change_info['bgxmdm']
        # del df_change_info['bgq']
        # del df_change_info['bgh']
        # del df_change_info['bgrq']

        # df_change_info['bgcnt'] = 1
        # df_change_info = df_change_info.groupby(['id'], as_index=False).sum()
        df_change_info = pd.read_csv(t_change_info, header=0)
        del df_change_info['bgq']
        del df_change_info['bgh']

        df_change_info['bgcnt'] = 1
        df_tmp = df_change_info[['id', 'bgcnt']]
        df_tmp = df_tmp.groupby(['id'], as_index=False).sum()

        def get_last_time(x):
            return x.sort_values(ascending = False)[:1]
        df_bgrq = df_change_info['bgrq'].groupby(df_change_info['id']).apply(get_last_time)

        df_change_info = pd.merge(df_tmp, df_bgrq, how='left', on='id')
        df_change_info['bgrq'] = df_change_info['bgrq'].apply(lambda x: str(x)[:4])

        pickle.dump(df_change_info, open(dump_path, 'wb'))
    return df_change_info

def gen_news_feat():
    dump_path = "./pre_data/eval_news_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_news_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_news_info = pd.read_csv(t_news_info, header=0)
        dict_atitude = {"积极": 1, "中立": 0.1, "消极": -1}
        df_news_info['positive_negtive'] = df_news_info['positive_negtive'].map(lambda x : dict_atitude[x])
        # public date 转变成迄今为止发生时间
        # 处理"xx小时前" 数据统一为昨天更新 
        cmp_date = "2020-10-09"
        def handle_public_date(str):
            if(check_date(str) is False):
                return 1
            else:
                return days(cmp_date, str)
        df_news_info['public_date'] = df_news_info['public_date'].map(lambda x: handle_public_date(x))

        # 新闻发布时间，越久远 因子越小
        df_news_info['news_factor'] = 1 - (df_news_info['public_date'] / df_news_info['public_date'].max())
        df_news_info['news_factor'] = df_news_info[['news_factor', 'positive_negtive']].apply(lambda x: x['news_factor'] * x['positive_negtive'], axis=1)

        del df_news_info['positive_negtive']
        del df_news_info['public_date']
        df_news_info = df_news_info.groupby(['id'], as_index=False).sum()

        pickle.dump(df_news_info, open(dump_path, 'wb'))
    return df_news_info

def gen_other_feat():
    dump_path = "./pre_data/eval_other_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_other_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_other_info = pd.read_csv(t_other_info, header=0)
        # 特征提取

        pickle.dump(df_other_info, open(dump_path, 'wb'))
    return df_other_info

def gen_eval_feat():
    dump_path = "./pre_data/eval_eva_info.pkl"
    if os.path.exists(dump_path) and is_update is not True:
        df_eval_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_eval_info = pd.read_csv(t_entprise_info, header=0)
        # 特征提取
        del df_eval_info['score']

        pickle.dump(df_eval_info, open(dump_path, 'wb'))
    return df_eval_info

def making_eval_data():
    dump_path = "./pre_data/eval.pkl"
    if is_update is not True:
        training_set = pickle.load(open(dump_path, 'rb'))
    else:
        print("making data beginning!")
        label_feat = gen_eval_feat()
        other_feat = gen_other_feat()

        training_set = pd.merge(label_feat, other_feat, how='left', on='id')
        training_set = training_set.groupby(['id'], as_index=False).mean()
        print("begin shape", training_set.shape)
        news_feat = gen_news_feat()
        training_set = pd.merge(training_set, news_feat, how='left', on='id')
        print("news shape", training_set.shape)

        change_feat = gen_change_feat()
        training_set = pd.merge(training_set, change_feat, how='left', on='id')
        print("change shape", training_set.shape)

        df_tax_info = gen_tax_feat()
        training_set = pd.merge(training_set, df_tax_info, how='left', on='id')
        print("tax shape", training_set.shape)

        anreport_feat = gen_anreport_feat()
        training_set = pd.merge(training_set, anreport_feat, how='left', on='id')
        print("anreport shape", training_set.shape)

        base_feat = gen_base_feat_v1()
        training_set = pd.merge(training_set, base_feat, how='left', on='id')
        print("base shape", training_set.shape)

        pickle.dump(training_set, open(dump_path, 'wb'))

    return training_set

if __name__ == '__main__':
    df_train = making_eval_data()
    print(df_train.info())
    print(df_train.values.shape)