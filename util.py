from exception_util import *
import pandas as pd
import numpy as np

"""
归一化
"""
def normalization(elm):
    # 列表或者np.ndarray或者pd.core.series.Series
    if isinstance(elm, list) or isinstance(elm, np.ndarray) or isinstance(elm, pd.core.series.Series):
        val_min = min(elm)
        val_max = max(elm)
        if val_max - val_min != 0:
            val_normalization = [(i - val_min)/(val_max - val_min) for i in elm]
        else:
            val_normalization = list(np.repeat(0, len(elm)))
            print('warning: {} is constant series, transform to 0 series'.format(elm))
        return val_normalization
    else:
        raise TypeError(type(elm))

    # 2 sklearn
    # scaler = MinMaxScaler((0,1))
    # data_driver_norm = scaler.fit_transform(data_driver.drop('Id', axis = 1))

"""
分段赋值 
"""
def cut_score(elm, cut):
    result = []

    # vfunc
    def get_score(x, val, min, max, tp):
        if tp == 0:
            if min <= x < max:
                return val
            else:
                return np.nan
        elif tp == 1:
            if x >= max:
                return val
            else:
                return np.nan
        elif tp == 2:
            if x < min:
                return val
            else:
                return np.nan
        elif tp == 3:
            if x == min:
                return val
            else:
                return np.nan

    vfunc_get_score = np.vectorize(get_score, otypes = [float])


    for key,value in cut.items():
        if ',' in key:
            tp = 0
            key_value = key.split(',')
            val_min = float(key_value[0])
            val_max = float(key_value[1])
        elif '>=' in key:
            tp= 1
            key_value = key.split('>=')
            val_min = np.nan
            val_max = float(key_value[1])
        elif '<' in key:
            tp = 2
            key_value = key.split('<')
            val_min = float(key_value[1])
            val_max = np.nan
        else:
            tp = 3
            key_value = key
            val_min = float(key_value[0])
            val_max = float(key_value[0])

        result_tmp = vfunc_get_score(elm, value, val_min, val_max, tp)
        result_tmp = list(result_tmp)
        result.append(result_tmp)

    res_array = np.array(result)
    res = np.apply_along_axis(func1d = np.nanmax, axis = 0, arr = res_array)

    return res

"""
缺失值填补-数值型
"""
def fill_na(elm, tp):
    # 列表或者np.ndarray或者pd.core.series.Series
    if  isinstance(elm, pd.core.series.Series):
        pass
    elif isinstance(elm, list) or isinstance(elm, np.ndarray):
        elm = pd.Series(elm)
    else:
        raise TypeError(type(elm))

    # 缺失值填补
    res = None
    if tp.isdigit():
        val = float(tp)
        res = elm.fillna(val)
    elif tp == 'mean':
        val = elm.mean()
        res = elm.fillna(val)
    elif tp == 'mode':
        val = elm.mode()[0] # 众数可能是多个,取第一个值进行填补
        res = elm.fillna(val)
    elif tp == 'min':
        val = elm.min()
        res = elm.fillna(val)
    elif tp == 'max':
        val = elm.max()
        res = elm.fillna(val)
    elif tp == 'median':
        val = elm.median()
        res = elm.fillna(val)
    return res

"""
计算woe值 IV值
"""
def cal_woe(feature, label):
    data = pd.DataFrame({'fea':feature, 'la':label})

    subdata = data.groupby('fea')['fea'].count()
    suby = data.groupby('fea')['la'].sum()
    data_2 = pd.merge(subdata, suby, how="left", left_index=True, right_index=True)

    g_total = data_2['la'].sum() # 1 的总数量
    total = data_2['fea'].sum() # 0 + 1 的总数量
    b_total = total - g_total # 0 的总数量

    data_2["good"] = data_2.apply(lambda x: x['la'] / (g_total + 1), axis = 1)
    data_2["bad"] = data_2.apply(lambda x: (x['fea'] - x['la']) / (b_total + 1), axis = 1)

    # 计算WOE值
    data_2['WOE'] = data_2.apply(lambda x:round(np.log(x['good']/(x['bad'] + 1e-2) + 1e-2), 3), axis = 1)  # 处理分子或者分母为0的问题

    # 计算IV值
    data_2["IV"] = data_2.apply(lambda x: (x['good'] - x['bad']) * x['WOE'], axis = 1)
    IV = round(sum(data_2["IV"]), 3)

    return data_2, IV



"""
分箱
"""
def bin_box(data, tp, bins):
    if tp == '0': # 等距分箱
        data_cut = pd.cut(data, bins = bins)
        df = pd.DataFrame({'data':data, 'cut':data_cut})

    if tp == '1': # 等频分箱
        data_cut = pd.qcut(data, q = bins)
        df = pd.DataFrame({'data':data, 'cut':data_cut})

    # 统计分布
    df_table = df['cut'].value_counts().to_frame('count')
    df_table['min'] = list(map(lambda x: x.left, df_table.index))

    # 计算占比
    df_table['rate'] = (df_table['count'] / df_table['count'].sum()).round(2)  # 占比

    # 计算累计占比
    df_table = df_table.sort_values('min').reset_index()
    df_table = df_table.drop('min', axis=1)
    df_table['cum_rate'] = (df_table['count'].cumsum() / df_table['count'].sum()).round(2)

    return df,df_table

if __name__ == '__main__':
    data = np.random.randint(0, 100, 20)
    bin_box(data, '1', 5)
