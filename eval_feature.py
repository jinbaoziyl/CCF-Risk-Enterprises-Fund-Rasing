import pandas as pd 
import numpy as ny 
import pickle
import os

t_base_info = "./dataset/train/base_info.csv"
t_annual_report_info = "./dataset/train/annual_report_info.csv"
t_tax_info = "./dataset/train/tax_info.csv"
t_change_info = "./dataset/train/change_info.csv"
t_news_info = "./dataset/train/news_info.csv"
t_other_info = "./dataset/train/other_info.csv"
t_evaluate_info = "./dataset/entprise_evaluate.csv" # evaluate data

is_update = True

if not os.path.exists("./pre_data"):
    os.mkdir("./pre_data")

def gen_base_feat():
    dump_path = "./pre_data/eval_base_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_base_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_base_info = pd.read_csv(t_base_info, header=0)
        # 特征提取
        del df_base_info['industryphy']
        del df_base_info['opscope']
        del df_base_info['dom']
        del df_base_info['protype']
        del df_base_info['oploc']
        del df_base_info['opfrom']
        del df_base_info['opto']

        pickle.dump(df_base_info, open(dump_path, 'wb'))
    return df_base_info

def gen_anreport_feat():
    dump_path = "./pre_data/eval_anreport_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_anreport_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_anreport_info = pd.read_csv(t_annual_report_info, header=0)
        # 特征提取
        del df_anreport_info['BUSSTNAME']
        del df_anreport_info['MEMNUM']
        del df_anreport_info['FARNUM']
        del df_anreport_info['ANNNEWMEMNUM']
        del df_anreport_info['ANNREDMEMNUM']
        del df_anreport_info['ANCHEYEAR']

        pickle.dump(df_anreport_info, open(dump_path, 'wb'))
    return df_anreport_info

def gen_tax_feat():
    dump_path = "./pre_data/eval_tax_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_tax_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_tax_info = pd.read_csv(t_tax_info, header=0)
        # 特征提取
        del df_tax_info['START_DATE']
        del df_tax_info['END_DATE']
        del df_tax_info['TAX_CATEGORIES']

        pickle.dump(df_tax_info, open(dump_path, 'wb'))
    return df_tax_info

def gen_change_feat():
    dump_path = "./pre_data/eval_change_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_change_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_change_info = pd.read_csv(t_change_info, header=0)
        # 特征提取
        del df_change_info['bgq']
        del df_change_info['bgh']
        del df_change_info['bgrq']

        pickle.dump(df_change_info, open(dump_path, 'wb'))
    return df_change_info

def gen_news_feat():
    dump_path = "./pre_data/eval_news_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_news_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_news_info = pd.read_csv(t_news_info, header=0)
        # 特征提取
        del df_news_info['positive_negtive']
        del df_news_info['public_date']

        pickle.dump(df_news_info, open(dump_path, 'wb'))
    return df_news_info

def gen_other_feat():
    dump_path = "./pre_data/eval_other_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_other_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_other_info = pd.read_csv(t_other_info, header=0)
        # 特征提取

        pickle.dump(df_other_info, open(dump_path, 'wb'))
    return df_other_info

def gen_eval_feat():
    dump_path = "./pre_data/eval_label_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_label_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_label_info = pd.read_csv(t_evaluate_info, header=0)
        # 特征提取

        pickle.dump(df_label_info, open(dump_path, 'wb'))
    return df_label_info

def making_eval_data():
    dump_path = "./pre_data/eval.pkl"
    if is_update == False:
        eval_set = pickle.load(open(dump_path, 'rb'))
    else:
        eval_feat = gen_eval_feat()
        other_feat = gen_other_feat()
        news_feat = gen_news_feat()
        change_feat = gen_change_feat()
        # tax_feat = gen_tax_feat()
        anreport_feat = gen_anreport_feat()
        base_feat = gen_base_feat()

        # 合成训练集
        eval_set = pd.merge(eval_feat, other_feat, how='left', on='id')
        eval_set = pd.merge(eval_set, news_feat, how='left', on='id')
        eval_set = pd.merge(eval_set, change_feat, how='left', on='id')
        # eval_set = pd.merge(eval_set, tax_feat, how='left', on='id')
        eval_set = pd.merge(eval_set, anreport_feat, how='left', on='id')
        eval_set = pd.merge(eval_set, base_feat, how='left', on='id')

        # del eval_set['score']
        pickle.dump(eval_set, open(dump_path, 'wb'))

        feat_id = {i:fea for i ,fea in enumerate(list(eval_set.columns))}
        print(feat_id)

    return eval_set

if __name__ == '__main__':
    df_eval = making_eval_data()
    print(df_eval.info())
    print(df_eval.head(10))
    print(df_eval.values.shape)