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
t_entprise_info = "./dataset/train/entprise_info.csv" # train labeled data

is_update = True

if not os.path.exists("./pre_data"):
    os.mkdir("./pre_data")

def gen_base_feat():
    dump_path = "./pre_data/train_base_info.pkl"
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
    dump_path = "./pre_data/train_anreport_info.pkl"
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
    dump_path = "./pre_data/train_tax_info.pkl"
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
    dump_path = "./pre_data/train_change_info.pkl"
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
    dump_path = "./pre_data/train_news_info.pkl"
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
    dump_path = "./pre_data/train_other_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_other_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_other_info = pd.read_csv(t_other_info, header=0)
        # 特征提取

        pickle.dump(df_other_info, open(dump_path, 'wb'))
    return df_other_info

def gen_label_feat():
    dump_path = "./pre_data/train_label_info.pkl"
    if os.path.exists(dump_path) and is_update == False:
        df_label_info = pickle.load(open(dump_path, 'rb'))
    else:
        df_label_info = pd.read_csv(t_entprise_info, header=0)
        # 特征提取

        pickle.dump(df_label_info, open(dump_path, 'wb'))
    return df_label_info

def making_training_data():
    dump_path = "./pre_data/training.pkl"
    dump_path1 = "./pre_data/training1.pkl"
    dump_path2 = "./pre_data/training2.pkl"
    dump_path3 = "./pre_data/training3.pkl"
    dump_path4 = "./pre_data/training4.pkl"
    if is_update == False:
        training_set = pickle.load(open(dump_path1, 'rb'))
    else:
        label_feat = gen_label_feat()
        print(label_feat.shape)
        other_feat = gen_other_feat()
        news_feat = gen_news_feat()
        change_feat = gen_change_feat()
        # tax_feat = gen_tax_feat()
        anreport_feat = gen_anreport_feat()
        base_feat = gen_base_feat()

        # 合成训练集
        training_set = pd.merge(label_feat, other_feat, how='left', on='id')
        training_set = pd.merge(training_set, news_feat, how='left', on='id')
        training_set = pd.merge(training_set, change_feat, how='left', on='id')
        # training_set = pd.merge(training_set, tax_feat, how='left', on='id')
        training_set = pd.merge(training_set, anreport_feat, how='left', on='id')
        training_set = pd.merge(training_set, base_feat, how='left', on='id')
        print(training_set.shape)

        # print(training_set.shape[0], training_set.shape[1])
        # print(training_set.tail(2))
        # section_size = training_set.shape[0] // 4
        pickle.dump(training_set, open(dump_path, 'wb'))
        # pickle.dump(training_set[0:section_size], open(dump_path1, 'wb'))
        # pickle.dump(training_set[section_size:2*section_size], open(dump_path2, 'wb'))
        # pickle.dump(training_set[2*section_size:3*section_size], open(dump_path3, 'wb'))
        # pickle.dump(training_set[3*section_size:-1], open(dump_path4, 'wb'))

        feat_id = {i:fea for i ,fea in enumerate(list(training_set.columns))}
        print(feat_id)

    return training_set

if __name__ == '__main__':
    df_train = making_training_data()
    print(df_train.info())
    # print(df_train.head(10))
    print(df_train.values.shape)