# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: TrainUtils.py
# time: 2020/3/16 10:36
# doc: 
"""

from sklearn.metrics import (classification_report,average_precision_score)
import pandas as pd
import joblib,os
from lightgbm.sklearn import LGBMClassifier
import gc
from talib import LINEARREG_SLOPE
import numpy as np
import datetime


def get_all_NaN_one_cols(csv_path,header='infer',sep=','):
    '''
    获取csv_path这个DataFrame中，所有值都是全部缺失值，全部某个单一值的列名
    :param csv_path:
    :param header:
    :param sep:
    :return:
    '''
    df=pd.read_csv(csv_path,header=header,sep=sep) if isinstance(csv_path,str) else csv_path
    remove_cols=[]
    for col in df.columns:
        # if df[col].isnull().sum()==df.shape[0] or df[col].nunique()==1:
        if df[col].isnull().sum()==df.shape[0] or (df[col].nunique()==1 and df[col].isnull().sum() == 0):
            remove_cols.append(col)
    use_cols=list(set(df.columns)-set(remove_cols))
    print('All cols num: {},remove cols num: {}, use cols num: {}'.format(df.shape[1],len(remove_cols),len(use_cols)))
    return use_cols,remove_cols


def put_all_labels(src_folder, tag_path, save_folder):
    '''
    对src_folder中每个csv打标签，标签文件为tag_path，标记完之后的csv保存到save_folder中
    :param src_folder:
    :param tag_path:
    :param save_folder:
    :return:
    '''
    os.makedirs(save_folder, exist_ok=True)

    def put_label(csv_path, label_df, save_folder):
        ##### train_df
        cols = ['dt', 'serial_number', 'model', 'smart_1_normalized', 'smart_1raw', 'smart_3_normalized',
                'smart_4_normalized', 'smart_4raw',
                'smart_5_normalized', 'smart_5raw', 'smart_7_normalized', 'smart_7raw', 'smart_9_normalized',
                'smart_9raw',
                'smart_10_normalized', 'smart_10raw', 'smart_12_normalized', 'smart_12raw', 'smart_184_normalized',
                'smart_184raw',
                'smart_187_normalized', 'smart_187raw', 'smart_188_normalized', 'smart_188raw', 'smart_189_normalized',
                'smart_189raw',
                'smart_190_normalized', 'smart_190raw', 'smart_192_normalized', 'smart_192raw', 'smart_193_normalized',
                'smart_193raw',
                'smart_194_normalized', 'smart_194raw', 'smart_195_normalized', 'smart_195raw', 'smart_197_normalized',
                'smart_197raw',
                'smart_198_normalized', 'smart_198raw', 'smart_199_normalized', 'smart_199raw', 'smart_240_normalized',
                'smart_240raw',
                'smart_241_normalized', 'smart_241raw', 'smart_242_normalized', 'smart_242raw']

        df = pd.read_csv(csv_path, usecols=cols)
        df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d')

        df = df.merge(label_df[['serial_number', 'model', 'fault_time']], how='left', on=['serial_number', 'model'])
        df['diff_day'] = (df['fault_time'] - df['dt']).dt.days
        df['label'] = 0
        df.loc[df['diff_day'] <= 30, 'label'] = 1
        df.to_csv(os.path.join(save_folder, os.path.split(csv_path)[-1]), index=False)
        print('finished: ', csv_path)

    #### prepare tag
    label_df = pd.read_csv(tag_path)
    label_df['fault_time'] = pd.to_datetime(label_df['fault_time'], format='%Y-%m-%d')
    label_df = label_df.drop_duplicates(['serial_number', 'model'])

    #####################
    all_csvs = os.listdir(src_folder)
    for csv_name in all_csvs:
        put_label(os.path.join(src_folder, csv_name), label_df, save_folder)



def valid_folder_dates(src_folder, dst_folder):
    '''
    对src_folder中的每个csv文件，读取df内容，按天排序，删除天数重复的行，
    填充缺失的天数，然后保存到dst-folder中
    :param src_folder:
    :param dst_folder:
    :return:
    '''
    os.makedirs(dst_folder, exist_ok=True)

    def valid_df_date(csv_path, save_path):
        df = pd.read_csv(csv_path)
        df=df.drop_duplicates('dt')
        df.reset_index(drop=True, inplace=True)
        df_dates=pd.to_datetime(df['dt'])
        min_d = df_dates.min()
        max_d = df_dates.max()
        diff = (max_d - min_d).days + 1
        raw_len = len(df)
        if raw_len<diff:
            raw_cols = df.shape[1]
            missing_days = set(pd.date_range(min_d, max_d)) - set(df_dates)
            nan_col = [np.nan] * raw_cols
            dt_idx = df.columns.tolist().index('dt')
            for idx, day in enumerate(missing_days):
                nan_col[dt_idx] = day.strftime('%Y-%m-%d')
                df.loc[raw_len + idx + 1] = nan_col
        df = df.sort_values(by='dt', ascending=True)
        df['serial_number'] = df['serial_number'].fillna(method='ffill')
        df['model'] = df['model'].fillna(method='ffill')
        df['label'] = df['label'].fillna(method='ffill')
        df = df.interpolate()
        df.fillna(-1,inplace=True)
        df.to_csv(save_path[:-4]+'_'+str(min_d.strftime('%Y%m%d'))+'_'+str(max_d.strftime('%Y%m%d'))+'.csv', index=False)

    all_csvs = os.listdir(src_folder)
    for idx, csv_file in enumerate(all_csvs):
        valid_df_date(os.path.join(src_folder, csv_file), os.path.join(dst_folder, csv_file))
        print('\r {}/{} finished...'.format(idx + 1, len(all_csvs)), end=' ')
    print('\nDONE')


def split_save_disk_df(src_folder,save_folder):
    '''
    将src_folder中的所有df，按照serial_number+model的方式取出来并保存到save_folder中
    每一个盘保存到一个csv中。
    :param src_folder:
    :param save_folder:
    :return:
    '''
    os.makedirs(save_folder,exist_ok=True)
    for csv_name in os.listdir(src_folder):
        df=pd.read_csv(os.path.join(src_folder,csv_name))
        for name,tmp_df in df.groupby(['serial_number','model']):
            save_path=os.path.join(save_folder,name[0]+'_'+str(name[1])+'.csv')
            if os.path.exists(save_path):
                tmp_df.to_csv(save_path,index=False,mode='a',header=False)
            else:
                tmp_df.to_csv(save_path,index=False,mode='w',header=True)
        print('finished: ',csv_name)
    print('DONE')


def load_dataset_Month(raw_csv_folder,first_dt_path,save_path,dates): # need test
    '''
    从raw_csv_folder中选择满足一定日期条件的csv，加载cols列，计算raw列的斜率，将最终所有列都整合到一起，组建数据集。
    :param raw_csv_folder: 所有盘的数据都存在这个文件夹中
    :param first_dt_path: 每个硬盘的第一天使用的日期信息存放在这个路径中
    :param save_path: 最终的df保存的路径
    :param dates 选择的范围：eg: dates=[20180401,20180630] # 只提取这三个月的数据
    :return:
    '''
    ###first_dt_path
    first_df = pd.read_csv(first_dt_path)  # columns: 'serial_number','dt_first','model'
    first_df['dt_first'] = pd.to_datetime(first_df['dt_first'], format='%Y%m%d')

    cols=['serial_number', 'model','dt', 'smart_1_normalized', 'smart_1raw', 'smart_4_normalized', 'smart_4raw',
 'smart_5_normalized', 'smart_5raw', 'smart_7_normalized', 'smart_7raw', 'smart_9_normalized', 'smart_9raw',
 'smart_12_normalized', 'smart_12raw',
  'smart_188_normalized', 'smart_188raw', 'smart_190_normalized', 'smart_190raw', 'smart_192_normalized',
               'smart_192raw', 'smart_193_normalized', 'smart_193raw',
 'smart_197_normalized', 'smart_197raw',
  'smart_199_normalized', 'smart_199raw', 'smart_241_normalized', 'smart_241raw',
               'smart_242_normalized', 'smart_242raw', 'label']
    cnt=0
    for csv_name in os.listdir(raw_csv_folder):
        csv_start,csv_end=csv_name.split('_')[3:5]
        if int(csv_end[:-4])<dates[0] or int(csv_start)>dates[1]:
            pass
        else:
            # print(csv_name)
            df=pd.read_csv(os.path.join(raw_csv_folder,csv_name),usecols=cols)
            df['dt']=df['dt'].apply(lambda x: int(x.replace('-','')))
            for col in cols:
                if col.endswith('raw'):
                    df[col+'_s3']=LINEARREG_SLOPE(df[col],timeperiod=3)
            df=df[(df['dt']>=dates[0]) & (df['dt']<=dates[1])]
            df = df.merge(first_df, how='left', on=['serial_number', 'model'])
            df['dt']=pd.to_datetime(df['dt'],format='%Y%m%d')
            df['days'] = (df['dt'] - df['dt_first']).dt.days
            df.dropna(inplace=True)
            if len(df)==0:continue
            if cnt==0:
                df.to_csv(save_path, index=False, mode='w', header=True)
            else:
                df.to_csv(save_path, index=False, mode='a', header=False)
            cnt+=1
        print('\rcnt: {}..'.format(cnt),end=' ')
    print('DONE')


def prepare_Train_Val_set(src_folder,train_months,val_months,cols=None):
    '''
    从src_folder中准备trainset, valset
    :param src_folder: 所有的Month数据都存放在这个文件夹中，比如以Set_开头，eg: Set_201802.csv
    :param cols: 需要使用的cols
    :param train_months: 一个list，eg: [201801,201802]，这几个月份的数据会整合作为trainset
    :param val_months:  一个list，eg: [201803，201804], 这几个月份的数据会整合作为valset
    :return:
    '''
    cols=cols or ['model', 'smart_1_normalized', 'smart_1raw', 'smart_4_normalized', 'smart_4raw',
                  'smart_5_normalized', 'smart_5raw',
     'smart_7_normalized', 'smart_7raw', 'smart_9_normalized', 'smart_9raw', 'smart_12_normalized', 'smart_12raw',
     'smart_188_normalized','smart_188raw','smart_190_normalized','smart_190raw','smart_192_normalized','smart_192raw',
     'smart_193_normalized','smart_193raw','smart_197_normalized', 'smart_197raw','smart_199_normalized','smart_199raw',
     'smart_241_normalized','smart_241raw','smart_242_normalized','smart_242raw',
     'label',
     'smart_1raw_s3', 'smart_4raw_s3','smart_5raw_s3','smart_7raw_s3','smart_9raw_s3','smart_12raw_s3',
     'smart_188raw_s3','smart_190raw_s3','smart_192raw_s3','smart_193raw_s3','smart_197raw_s3',
     'smart_199raw_s3','smart_241raw_s3','smart_242raw_s3','days']
    train_df=None
    for train_month in train_months:
        df=pd.read_csv(os.path.join(src_folder,'Set_'+str(train_month)+'.csv'),usecols=cols)
        # print('train_month: {}, value_counts: {}'.format(train_month,df['label'].value_counts()))
        train_df=df if train_df is None else pd.concat([train_df,df])
        del df
        gc.collect()
    train_df=train_df.sample(frac=1.0)
    trainy = train_df['label']
    trainX = train_df.drop(['label'], axis=1)
    del train_df

    val_df=None
    for val_month in val_months:
        df=pd.read_csv(os.path.join(src_folder,'Set_'+str(val_month)+'.csv'),usecols=cols)
        val_df=df if val_df is None else pd.concat([val_df,df])
        del df
        gc.collect()
    val_df=val_df.sample(frac=1.0)
    valy=val_df['label']
    valX=val_df.drop(['label'],axis=1)
    del val_df

    gc.collect()
    return trainX, trainy, valX, valy


def train_LGBM(src_folder,cols,model_save_folder,lr):
    '''
    从src_folder中加载trainset,valset, 并训练LGBM，将训练后的模型保存到model_save_folder中
    :param src_folder:
    :param cols:
    :param model_save_folder:
    :param lr:
    :return:
    '''
    os.makedirs(model_save_folder,exist_ok=True)

    def auc_prc(y_true, y_pred):
        return 'AUC_PRC', average_precision_score(y_true, y_pred), True

    train_months_li=[[201803,201804]]
    val_months_li=[[201806]]
    for train_months,val_months in zip(train_months_li,val_months_li):
        print('************** train Months: {}, val Months: {}**************'
              .format(', '.join([str(i) for i in train_months]), ', '.join([str(i) for i in val_months])))
        trainX, trainy, valX, valy=prepare_Train_Val_set(src_folder,train_months,val_months,cols)
        print('trainX shape: {}, valX shape: {}'.format(trainX.shape, valX.shape))
        print('trainy value_counts: {}'.format(trainy.value_counts()))
        print('valy value_counts: {}'.format(valy.value_counts()))
        clf=LGBMClassifier(num_leaves=127,learning_rate=lr,n_estimators=10000,objective='binary',
                           is_unbalance=True,
                           subsample=0.8,colsample_bytree=0.8,
                           device_type='gpu', gpu_platform_id=1, gpu_device_id=0
                           )
        t0 = time.time()
        clf.fit(
            trainX, trainy,
            eval_set=[(valX, valy)],
            eval_metric=auc_prc,
            early_stopping_rounds=50,
            verbose=100)
        print('fit time: {:.4f}'.format(time.time() - t0))
        save_name='LGBM_'+'Val_M'+', '.join([str(i) for i in val_months])+ datetime.now().strftime('%Y%m%d_%H%M%S')
        joblib.dump(clf, os.path.join(model_save_folder,save_name))
        print('model is saved to {}'.format(save_name))
        gc.collect()


def prepare_models_prob_Set(save_path, src_folder, models_path_li=None):
    '''
    从src_folder中的月份原始数据中加载所有正样本，并随机获取一定比例的负样本，使用多个模型来预测，得到每个模型的概率。
    并将每个样本每个模型的概率数值保存到save_path对应的路径中
    :param save_path:
    :param src_folder:
    :param models_path_li:
    :return:
    '''
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    # src_folder=src_folder or r'D:\DataSet\DiskError\DataSet\DataSet'
    models_path_li = models_path_li or ["../user_data/model_data/Model0_20200311_184904",
                                      "../user_data/model_data/Model1_20200311_223537",
                                      "../user_data/model_data/Model2_20200312_095621",
                                      "../user_data/model_data/Model3_20200312_135026",
                                      "../user_data/model_data/Model4_20200312_185358",
                                      "../user_data/model_data/Model5_20200313_085224",
                                      "../user_data/model_data/Model6_20200313_111017",
                                      "../user_data/model_data/Model7_20200313_173538"]
    cols2 = ['serial_number','dt',
        'model', 'smart_1_normalized', 'smart_1raw', 'smart_4_normalized', 'smart_4raw', 'smart_5_normalized',
             'smart_5raw',
             'smart_7_normalized', 'smart_7raw', 'smart_9_normalized', 'smart_9raw', 'smart_12_normalized',
             'smart_12raw',
             'smart_188_normalized', 'smart_188raw', 'smart_190_normalized', 'smart_190raw', 'smart_192_normalized',
             'smart_192raw',
             'smart_193_normalized', 'smart_193raw', 'smart_197_normalized', 'smart_197raw', 'smart_199_normalized',
             'smart_199raw',
             'smart_241_normalized', 'smart_241raw', 'smart_242_normalized', 'smart_242raw',
             'label',
             'smart_1raw_s3', 'smart_4raw_s3', 'smart_5raw_s3', 'smart_7raw_s3', 'smart_9raw_s3', 'smart_12raw_s3',
             'smart_188raw_s3', 'smart_190raw_s3', 'smart_192raw_s3', 'smart_193raw_s3', 'smart_197raw_s3',
             'smart_199raw_s3', 'smart_241raw_s3', 'smart_242raw_s3', 'days']

    all_df=None
    B=list(range(201707,201713)) + list(range(201801,201807))
    for month in B:
        csv_name='Set_'+str(month)+'.csv'
        df=pd.read_csv(os.path.join(src_folder,csv_name),usecols=cols2)
        pos_df=df[df['label']==1]
        all_df=pos_df if all_df is None else pd.concat([all_df,pos_df])
        neg_df=df[df['label']==0].sample(n=len(pos_df)*9)
        all_df=pd.concat([all_df,neg_df])
    all_df=all_df.sample(frac=1.0)
    result_df=all_df[['serial_number','model','dt','days','label']]
    all_df.drop(['serial_number','dt','label'],axis=1,inplace=True)
    for idx,model_path in enumerate(models_path_li):
        model=joblib.load(model_path)
        result_df['p_'+str(idx)]=model.predict_proba(all_df,num_iteration=model.best_iteration_)[:,1]
        print('\r model {} finished..'.format(idx),end=' ')
    result_df.to_csv(save_path,index=False)
    print('\nDONE')


def split_DataSet(raw_set_path,save_folder, val_ratio=0.2):
    '''
    对不平衡数据集raw_set_path进行划分，trainset,valset都保存到save_folder中，其中val_set占比val ratio
    :param raw_set_path:
    :param save_folder:
    :param val_ratio:
    :return:
    '''
    os.makedirs(save_folder,exist_ok=True)
    all_df=pd.read_csv(raw_set_path)
    pos_df=all_df[all_df['label']==1]
    pos_df=pos_df.sample(frac=1.0)
    neg_df=all_df[all_df['label']==0]
    neg_df=neg_df.sample(frac=1.0)
    val_pos_num=int(val_ratio*len(pos_df))
    val_neg_num=int(val_ratio*len(neg_df))
    val_df=pd.concat([pos_df[:val_pos_num],neg_df[:val_neg_num]])
    val_df=val_df.sample(frac=1.0)
    print('val_df shape: ',val_df.shape)
    print('val_df labels: ',val_df['label'].value_counts())
    val_df.to_csv(os.path.join(save_folder,'MergeSet2_Val.csv'),index=False)
    del val_df

    train_df=pd.concat([pos_df[val_pos_num:],neg_df[val_neg_num:]])
    train_df=train_df.sample(frac=1.0)
    print('train_df shape: ',train_df.shape)
    print('train_df labels: ',train_df['label'].value_counts())
    train_df.to_csv(os.path.join(save_folder,'MergeSet2_Train.csv'),index=False)
    del train_df
    print('DONE')


from sklearn.ensemble import RandomForestClassifier
import time
from datetime import datetime
def train_merged_Model(trainset_path,valset_path,model_save_folder,lr=0.001,isLGB=True):
    '''
    训练融合后的模型
    :param trainset_path:
    :param valset_path:
    :param model_save_folder:
    :param lr:
    :param isLGB:
    :return:
    '''
    os.makedirs(model_save_folder,exist_ok=True)
    cols=['model', 'days','label']
    cols+=['p_'+str(i) for i in range(8)]
    def load_set(set_path):
        df=pd.read_csv(set_path,usecols=cols)
        df['model_1']=df['model'].apply(lambda x: int(x==1))
        df['model_2']=df['model'].apply(lambda x: int(x==2))
        sety=df['label']
        setX=df.drop(['label','model'],axis=1)
        del df
        return setX,sety

    def auc_prc(y_true, y_pred):
        return 'AUC_PRC', average_precision_score(y_true, y_pred), True

    trainX,trainy=load_set(trainset_path)
    valX,valy=load_set(valset_path)
    print('trainset info, shape: {},value_counts: {}'.format(trainX.shape,trainy.value_counts()))
    print('valset info, shape: {},value_counts: {}'.format(valX.shape,valy.value_counts()))

    ##########LGBMClassifier
    clf = LGBMClassifier(num_leaves=127, learning_rate=lr, n_estimators=10000, objective='binary',
                         is_unbalance=True,
                         subsample=0.8, colsample_bytree=0.8,
                         ) if isLGB else RandomForestClassifier()
    t0 = time.time()
    if isLGB:
        clf.fit(
            trainX, trainy,
            eval_set=[(valX, valy)],
            eval_metric=auc_prc,
            early_stopping_rounds=50,
            verbose=100)
    else:
        clf.fit(trainX, trainy)
    print('fit time: {:.4f}'.format(time.time() - t0))
    save_name='LGBM_Merged_'+ datetime.now().strftime('%Y%m%d_%H%M%S') if isLGB else \
        'RF_Merged'+datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(clf, os.path.join(model_save_folder,save_name))
    print('Merged model is saved to {}'.format(save_name))


def evaluate_Merged_model(model_save_path, set_save_path, thresh=0.5,isLGB=True):
    '''
    评估融合模型在setX上的性能表现
    :param model_save_path:
    :param set_save_path:
    :param thresh:
    :param isLGB:
    :return:
    '''
    cols = ['model', 'days', 'label']
    cols += ['p_' + str(i) for i in range(8)]

    def load_set(set_path):
        df = pd.read_csv(set_path, usecols=cols)
        df['model_1'] = df['model'].apply(lambda x: int(x == 1))
        df['model_2'] = df['model'].apply(lambda x: int(x == 2))
        sety = df['label']
        setX = df.drop(['label', 'model'], axis=1)
        del df
        return setX, sety

    X_test, y_test = load_set(set_save_path)
    clf = joblib.load(model_save_path)
    pred_prob = clf.predict_proba(X_test,num_iteration=clf.best_iteration_)[:, 1] if isLGB else \
        clf.predict_proba(X_test)[:, 1]
    print('AUCPRC: ', average_precision_score(y_test, pred_prob))
    print('classification report: ')
    y_pred = (pred_prob > thresh).astype(int)
    print(classification_report(y_test, y_pred))


import lightgbm as lgb
def plot_feature_importance(models_path_li=None):
    '''
    绘制多个模型的feature_importance图
    :param models_path_li:
    :return:
    '''
    models_path_li = models_path_li or ["../user_data/model_data/Model0_20200311_184904",
                                      "../user_data/model_data/Model1_20200311_223537",
                                      "../user_data/model_data/Model2_20200312_095621",
                                      "../user_data/model_data/Model3_20200312_135026",
                                      "../user_data/model_data/Model4_20200312_185358",
                                      "../user_data/model_data/Model5_20200313_085224",
                                      "../user_data/model_data/Model6_20200313_111017",
                                      "../user_data/model_data/Model7_20200313_173538"]
    for model_path in models_path_li:
        clf=joblib.load(model_path)
        lgb.plot_importance(clf,title=os.path.split(model_path)[-1],figsize=(10,8))


def prepare_overlap_setB(raw_csv_folder, testA_path, subTest_path, cols, result_save_path):
    all_train_csvs = set(os.listdir(raw_csv_folder))
    testA_df = pd.read_csv(testA_path, usecols=cols)
    testA_df['model'] = testA_df['model'].astype(str)
    testA_df['disk'] = testA_df['serial_number'] + '_' + testA_df['model']
    all_disks = np.unique(testA_df['disk'])

    sub_df = pd.read_csv(subTest_path, usecols=cols)
    cnt = 0
    overlap_df = None
    for name, tmp_df in sub_df.groupby(['serial_number', 'model']):
        disk_id = name[0] + '_' + str(name[1]) + '.csv'
        if disk_id in all_train_csvs:
            # print('found overlap disk: ',disk_id)
            before_df = pd.read_csv(os.path.join(raw_csv_folder, disk_id), usecols=cols)
            before_df['dt'] = before_df['dt'].apply(lambda x: int(x.replace('-', '')))
            before_df = before_df.sort_values('dt').drop_duplicates('dt')
            before_df = before_df[-10:]
            overlap_df = before_df if overlap_df is None else pd.concat([overlap_df, before_df])
            cnt += 1
        disk_name = name[0] + '_' + str(name[1])
        if disk_name in all_disks:
            part_df = testA_df[testA_df['disk'] == disk_name]
            part_df = part_df.sort_values('dt').drop_duplicates('dt')
            part_df = part_df[-10:]
            part_df = part_df.drop(['disk'], axis=1)
            overlap_df = part_df if overlap_df is None else pd.concat([overlap_df, part_df])
            cnt += 1
        print('\rfound {} overlap disk...'.format(cnt), end=' ')
    overlap_df.to_csv(result_save_path, index=False)
    print('DONE')

