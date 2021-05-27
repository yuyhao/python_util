# -*- coding: utf-8 -*-
# @Time: 2021/5/20 下午4:38
# @Author: yuyinghao
# @FileName: data_explore.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time: 2021/5/17 下午5:31
# @Author: yuyinghao
# @FileName: data_exploer.py
# @Software: PyCharm

import pandas as pd
import numpy as np

"""
数据获取
"""
def get_data():
    data = pd.read_csv('datas/data_sub.csv')

    return data

"""
缺失值情况
"""
def na_exp(data):
    res = data.isnull().sum().to_frame('na_count').reset_index()
    res = res.rename(columns = {'index':'col_name'})
    res['count'] = data.shape[0]
    res['na_rate'] = (res['na_count'] / res['count'] * 100).round(1).map(str).str.cat(['%'] * data.shape[1])
    res['type'] = data.dtypes.to_list()

    return res

"""
分布情况
"""
def distribute_exp(data):
    res = data.describe().T.apply(lambda x: round(x, 3), axis = 0).reset_index()
    res = res.rename(columns={'index': 'col_name'})

    return res

def distribute_detail_exp(data, method = 'percentage'):
    res = pd.DataFrame()

    # 等频分箱
    def drop_duplicate_by_val(data, col_by, col_val, by='min'):
        def drop_data(data, by):
            if by == 'min':
                val = min(data[col_val])  # 最小值
            elif by == 'max':
                val = max(data[col_val])  # 最大值

            data_need = data[data[col_val] == val].head(1)  # 取第一条数据
            return data_need

        data_sub = data.groupby(col_by).apply(drop_data, by=by).reset_index(drop=True)

        return data_sub


    if method == 'percentage':
        for index, value in data.iteritems():
            val = value.dropna().map(lambda x: round(x, 2))
            val = val.sort_values().to_frame('value') # 正序排序
            val['count'] = 1
            val['cum_sum'] = (val['count'].cumsum()/val.shape[0] * 100).map(int)
            val = val[(val['cum_sum'] % 5 == 0)]

            val_index = drop_duplicate_by_val(val, 'cum_sum', 'value', by='min')

            value_idx = []
            for idx, value in enumerate(val_index['value']):
                if idx == 0:
                    t = str(value)
                else:
                    t = t + '-' + str(value)
                    value_idx.append(t)
                    t = str(value)

            df = pd.DataFrame({index:value_idx})
            res = pd.concat([res, df], axis = 1)
        res['rate'] = 0.05
        res['cum_rate'] = res['rate'].cumsum()
    return res

"""
结果输出
"""
def get_data_explore_result():
    data = get_data()
    dic = {}
    res_1 = na_exp(data)
    res_2 = distribute_exp(data)
    res_3 = distribute_detail_exp(data[list(res_2['col_name'])])

    with pd.ExcelWriter('datas/driver_data_explore_2.xlsx') as writer:
        sheet_name = ['缺失值情况', '特征分布', '详细特征分布']
        res_1.to_excel(writer, index=False, sheet_name=sheet_name[0])
        res_2.to_excel(writer, sheet_name=sheet_name[1])
        res_3.to_excel(writer, sheet_name=sheet_name[2])
    writer.save()

if __name__ == '__main__':
    get_data_explore_result()
