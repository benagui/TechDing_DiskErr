# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: main.py
# time: 2020/3/15 14:01
# doc: 
"""

from model.Utils import (prepare_subTest,add_days,models_predict_SetX,mergedModel_predict_SubTest)
import pandas as pd


def main():
    test_type='B' # or B
    raw_path='../data/round1_testA/disk_sample_smart_log_test_a.csv' if test_type=='A' else \
            '../data/round1_testB/disk_sample_smart_log_test_b.csv'
    overlap_path = '../user_data/tmp_data/testA_overlap.csv' if test_type == 'A' \
        else '../user_data/tmp_data/testB_overlap.csv'
    # 1. 准备subTestX
    all_df=prepare_subTest(raw_path,overlap_path)
    # all_df.to_csv(r'../user_data/need_delete/testB/testB_preparedSubTest.csv',index=False)

    # all_df=pd.read_csv(r'../user_data/need_delete/testB/testB_preparedSubTest.csv')
    # 2. 添加days列特征
    print('\nstart to add days col')
    all_df=add_days(all_df)
    # all_df.to_csv(r'../user_data/need_delete/testB/testB_add_days.csv',index=False)

    # all_df=pd.read_csv(r'../user_data/need_delete/testB/testB_add_days.csv')
    # 3. 模型预测1
    print('round1 models will predict setX')
    result_df=models_predict_SetX(all_df)
    # result_df.to_csv(r'../user_data/need_delete/testB/testB_models_predictSetX.csv',index=False)

    # result_df=pd.read_csv(r'../user_data/need_delete/testB/testB_models_predictSetX.csv')
    # 4. 模型预测2
    print('round2 model will predict setX')
    mergedModel_predict_SubTest(result_df,'../prediction_result/predictions.csv',False,0.55)


if __name__ == '__main__':
    main()

