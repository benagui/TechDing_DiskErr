# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: Utils.py
# time: 2020/3/15 17:02
# doc: 
"""
import numpy as np
import pandas as pd
import joblib,gc

COLS=['smart_1_normalized', 'smart_1raw', 'smart_4_normalized',
     'smart_4raw', 'smart_5_normalized', 'smart_5raw', 'smart_7_normalized',
     'smart_7raw', 'smart_9_normalized', 'smart_9raw', 'smart_12_normalized',
     'smart_12raw', 'smart_188_normalized', 'smart_188raw',
     'smart_190_normalized', 'smart_190raw', 'smart_192_normalized',
     'smart_192raw', 'smart_193_normalized', 'smart_193raw',
     'smart_197_normalized', 'smart_197raw', 'smart_199_normalized',
     'smart_199raw', 'smart_241_normalized', 'smart_241raw',
     'smart_242_normalized', 'smart_242raw']

def calc_slope(serial, timeperiod=3):
    return serial.rolling(timeperiod).apply(lambda x: np.polyfit(range(timeperiod), x, 1)[0], raw=False)

from talib import LINEARREG_SLOPE
def valid_df_date(overlap_df,df):  # OK
    '''
    对单个硬盘df，填充缺失日期的信息，并计算raw列的斜率，返回一个修之后的df，适合于trainset，testset
    :param df:
    :return: 返回的df中dt为int
    '''
    df = pd.concat([overlap_df,df])
    df = df.drop_duplicates('dt')
    df.reset_index(drop=True, inplace=True)
    df_dates = pd.to_datetime(df['dt'], format='%Y%m%d')
    min_d = df_dates.min()
    max_d = df_dates.max()
    diff = (max_d - min_d).days + 1
    raw_len = len(df)
    if raw_len < diff:
        raw_cols = df.shape[1]
        missing_days = set(pd.date_range(min_d, max_d)) - set(df_dates)
        nan_col = [np.nan] * raw_cols
        dt_idx = df.columns.tolist().index('dt')
        for idx, day in enumerate(missing_days):
            nan_col[dt_idx] = int(day.strftime('%Y%m%d'))
            df.loc[raw_len + idx + 1] = nan_col
    df = df.sort_values(by='dt', ascending=True)
    df['serial_number'] = df['serial_number'].fillna(method='ffill')
    df['model'] = df['model'].fillna(method='ffill')
    if 'label' in df.columns:
        df['label'] = df['label'].fillna(method='ffill')
    df = df.interpolate()
    df.fillna(-1, inplace=True)
    for col in df.columns:
        if col.endswith('raw'):
            df[col + '_s3'] = LINEARREG_SLOPE(df[col], timeperiod=3) # faster
    df = df[(df['dt'] >= 20180801) & (df['dt'] <= 20180831)]
    return df

def prepare_subTest(subTest_path,overlap_path):
    '''
    准备subTest数据集
    :param subTest_path: 原始的test文件路径
    :param overlap_path: 重叠数据所在文件路径
    :return:
    '''
    cols = ['serial_number', 'model', 'dt'] + COLS
    test_df = pd.read_csv(subTest_path, usecols=cols)
    all_df = None
    overlap_df = pd.read_csv(overlap_path)
    print('calculating df_date...')
    for idx, (name, tmp_df) in enumerate(test_df.groupby(['serial_number', 'model'])):
        overlap_df_tmp = overlap_df[(overlap_df['serial_number'] == name[0]) & (overlap_df['model'] == name[1])]
        date_df = valid_df_date(overlap_df_tmp, tmp_df)
        all_df = date_df if all_df is None else pd.concat([all_df, date_df])
        print('\r {} finished... all about 11500'.format(idx + 1), end=' ')
    del test_df
    gc.collect()
    return all_df


def add_days(df):
    '''
    在整个df中添加days列，并返回该df
    :param df: dt为int
    :return: dt 为datetime
    '''
    first_dt_path='../user_data/tmp_data/all_first_dt.csv'
    first_dt_df=pd.read_csv(first_dt_path)
    tmp=df[['serial_number','model','dt']]
    tmp=tmp.sort_values('dt').drop_duplicates(['serial_number','model']) # dt format 20181002
    tmp.rename(columns={'dt':'dt_first'},inplace=True)
    df_all = pd.concat([first_dt_df,tmp])
    df_all = df_all.sort_values('dt_first').drop_duplicates(['serial_number', 'model'])
    df = df.merge(df_all, how='left', on=['serial_number', 'model'])
    df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d')
    df['dt_first']=pd.to_datetime(df['dt_first'], format='%Y%m%d')
    df['days'] = (df['dt'] - df['dt_first']).dt.days
    df.dropna(inplace=True)
    del tmp,df_all
    gc.collect()
    return df


def models_predict_SetX(setX_df): # OK
    '''
    用多个模型models_path_li来预测setX，并返回预测之后的df
    :param models_path_li:
    :param setX_df: dt 为datetime
    :return: 各个模型预测后的probs
    '''
    models_path_li=["../user_data/model_data/Model0_20200311_184904",
                      "../user_data/model_data/Model1_20200311_223537",
                      "../user_data/model_data/Model2_20200312_095621",
                      "../user_data/model_data/Model3_20200312_135026",
                      "../user_data/model_data/Model4_20200312_185358",
                      "../user_data/model_data/Model5_20200313_085224",
                      "../user_data/model_data/Model6_20200313_111017",
                      "../user_data/model_data/Model7_20200313_173538"]
    s3_cols=[ 'smart_1raw_s3',
     'smart_4raw_s3', 'smart_5raw_s3', 'smart_7raw_s3', 'smart_9raw_s3',
     'smart_12raw_s3', 'smart_188raw_s3', 'smart_190raw_s3',
     'smart_192raw_s3', 'smart_193raw_s3', 'smart_197raw_s3',
     'smart_199raw_s3', 'smart_241raw_s3', 'smart_242raw_s3']
    cols=['model'] + COLS + s3_cols + ['days']
    result_df=setX_df[['serial_number', 'model', 'dt','days']]
    setX_df=setX_df[cols] # OK
    print('testX shape: ', setX_df.shape)
    for idx, model_path in enumerate(models_path_li):
        clf = joblib.load(model_path)
        result_df['p_' + str(idx)] = clf.predict_proba(setX_df, num_iteration=clf.best_iteration_)[:, 1]
    return result_df


def mergedModel_predict_SubTest(df, result_save_path, isLGB=True,thresh=0.5): # OK
    '''
    用融合模型来预测subTest_path对应的dataframe，并将预测结果保存到result_save_path中，用于提交
    :param df:
    :param result_save_path:
    :param isLGB:
    :param thresh:
    :return:
    '''
    setX=df.copy()
    setX['model_1'] = setX['model'].apply(lambda x: int(x == 1))
    setX['model_2'] = setX['model'].apply(lambda x: int(x == 2))
    setX = setX.drop(['serial_number', 'dt'], axis=1)
    cols = ['days'] + ['p_' + str(i) for i in range(8)] + ['model_1', 'model_2']
    setX = setX[cols]
    model_path='../user_data/model_data/Merged_20200313_215828' if isLGB else \
        '../user_data/model_data/Merged_20200313_220602'
    clf = joblib.load(model_path)
    df['merged_p'] = clf.predict_proba(setX, num_iteration=clf.best_iteration_)[:, 1] if isLGB else clf.predict_proba(
        setX)[:, 1]
    pos_df = df[df['merged_p'] >= thresh]
    pos_df = pos_df.sort_values('merged_p', ascending=False).drop_duplicates(['serial_number', 'model'])
    pos_df['manufacturer'] = 'A'
    pos_df['dt'] = pos_df['dt'].astype(str)
    pos_df['model'] = pos_df['model'].astype(int)
    print('Abnormal disk num: ', pos_df.shape[0])
    pos_df[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(result_save_path, index=False, header=False)
    print('DONE')

