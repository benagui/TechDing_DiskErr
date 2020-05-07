# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: train.py
# time: 2020/3/16 10:30
# doc: 
"""

from model.TrainUtils import *

def main():
    trainset_folder='../data/round1_train'
    testA_path='../data/round1_testA/disk_sample_smart_log_test_a.csv'
    tag_path='../data/disk_sample_fault_tag.csv'
    all_train_files=os.listdir(trainset_folder)

    ## 1. 统计有意义的列
    use_cols, _ =get_all_NaN_one_cols(os.path.join(trainset_folder,all_train_files[0]))

    ## 2. 对每个train文件打上label, 并存放到tmp_data中
    put_all_labels(trainset_folder, tag_path, '../user_data/tmp_data')

    ## 3. 内存不够，故将每个硬盘的数据单独保存一个csv
    split_save_disk_df('../user_data/tmp_data', '../user_data/tmp_data2')

    ## 4. 对每个硬盘的数据进行日期排序，确保不重复，缺失日期进行填充
    valid_folder_dates('../user_data/tmp_data2', '../user_data/tmp_data3')

    ## 5. 准备数据集，按照月份来整合,作为示例，如下只整合201801这一个月份
    # 如果内存足够，上面的2，3，4,5步骤可以合并为一个
    first_dt_path = '../user_data/tmp_data/all_first_dt.csv'
    load_dataset_Month('../user_data/tmp_data3', first_dt_path,
                       '../user_data/tmp_data4/Set_201801.csv', [20180101,20180131])

    ## 6. 使用LGBM训练单个模型,以下训练一个单模型，使用201803，201804月份的数据作为训练集，201806的为valset
    ## 此处换用其他月份的训练集和验证集，一共训练了8个基本模型
    train_LGBM('../user_data/tmp_data4', None, '../user_data/model_data', lr=0.01)

    ## 7. 模型融合，为了训练一个融合模型，此处准备训练集：从每个月份中随机抽取一定量的正样本和负样本，组成新的数据集
    prepare_models_prob_Set('../user_data/tmp_data5/All_probs_set.csv', '../user_data/tmp_data4', models_path_li=None)

    ## 8. 划分新数据集为train set, valset
    split_DataSet('../user_data/tmp_data5/All_probs_set.csv', '../user_data/tmp_data5', val_ratio=0.2)

    ## 9.训练融合模型：
    train_merged_Model('../user_data/tmp_data5/MergeSet2_Train.csv',
                       '../user_data/tmp_data5/MergeSet2_Val.csv',
                       '../user_data/model_data', lr=0.01, isLGB=False)

    ## 10. 评估融合模型的好坏：
    evaluate_Merged_model('../user_data/model_data/final_merged_model', '../user_data/tmp_data5/MergeSet2_Val.csv',
                          thresh=0.5, isLGB=False)

