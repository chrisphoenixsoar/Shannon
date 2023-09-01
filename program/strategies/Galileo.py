# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:37:03 2022

@author: w
"""

import pandas as pd

name = 'Galileo'  # 策略名称（必填）

period_list = ['W']  # 持仓周期（必填）

select_count_list = [5]  # 选股数量（必填）

flow_fin_cols = ['R_np','C_ncf_from_oa']  # 流量型财务字段

cross_fin_cols = ['B_total_liab']  # 截面型财务字段

add_fin_cols = ['R_np','B_total_liab','C_ncf_from_oa']  # 最终需要加到数据上的财务字段

factor_rank_dict={
    '总市值':True,
    '成交额std_5':True,
    'bias_5':True,    
    '收盘价': True,
    '量价相关系数_20': True,  
   
    }

# def special_data():
#     """
#     处理策略需要的专属数据，非必要。
#     :return:
#     """
#     return


# def before_merge_index(data, exg_dict, fill_0_list):
#     """
#     合并指数数据之前的处理流程，非必要。
#     :param data: 传入的数据
#     :param exg_dict: resample规则
#     :param fill_0_list: 合并指数时需要填充为0的数据
#     :return:
#     """
#     return data, exg_dict, fill_0_list


# def merge_single_stock_file(data, exg_dict):
#     """
#     合并策略需要的单个的数据，非必要。
#     :param data:传入的数据
#     :param exg_dict:resample规则
#     :return:
#     """
#     return data, exg_dict


def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    合并数据后计算策略需要的因子，非必要
    :param data:传入的数据
    :param fin_data:财报数据（去除废弃研报)
    :param fin_raw_data:财报数据（未去除废弃研报）
    :param exg_dict:resample规则
    :return:
    """
    
    
    data['成交额std_5']=data['成交额'].rolling(5).std()
    data['bias_5']=data['收盘价_复权']/data['收盘价_复权'].rolling(5).mean()-1
    
    data['量价相关系数_20']=data['复权因子'].rolling(20).corr(data['换手率'])
    data['经营现金流量总负债比'] = data['C_ncf_from_oa'] / data['B_total_liab']    

    

    exg_dict['成交额std_5'] = 'last'
    exg_dict['bias_5'] = 'last'
    exg_dict['总市值'] = 'last'
    exg_dict['收盘价'] = 'last'
    exg_dict['量价相关系数_20'] = 'last'
    exg_dict['经营现金流量总负债比'] = 'last'
    
    return data, exg_dict


def after_resample(data):
    """
    数据降采样之后的处理流程，非必要
    :param data: 传入的数据
    :return:
    """
    return data


def filter_stock(all_data):
    """
    过滤函数，在选股前过滤，必要
    :param all_data: 截面数据
    :return:
    """
    all_data = all_data[all_data['交易日期'] >= '20070101']
    all_data = all_data[all_data['股票代码'].str.contains('sh68|bj') == False]

  # =删除不能交易的周期数
    all_data['经营现金流量总负债比_分位数'] =  all_data.groupby(['交易日期'])['经营现金流量总负债比'].rank(ascending=True, pct=True)
    all_data = all_data[all_data['经营现金流量总负债比_分位数'] > 0.05]
    all_data['散户卖出占比排名'] = all_data.groupby(['交易日期'])\
        ['散户资金卖出占比'].rank(ascending=False,pct=True)
    all_data=all_data[all_data['散户卖出占比排名']>0.2]   
     
    # 删除月末为st状态的周期数
    all_data = all_data[all_data['股票名称'].str.contains('ST') == False]
    # 删除月末为s状态的周期数
    all_data = all_data[all_data['股票名称'].str.contains('S') == False]
    # 删除月末有退市风险的周期数
    all_data = all_data[all_data['股票名称'].str.contains('退') == False]
    # 删除交易天数过少的周期数
    all_data = all_data[all_data['交易天数'] / all_data['市场交易天数'] >= 0.8]

    all_data = all_data[all_data['下日_是否交易'] == 1]
    all_data = all_data[all_data['下日_开盘涨停'] == False]
    all_data = all_data[all_data['下日_是否ST'] == False]
    all_data = all_data[all_data['下日_是否退市'] == False]
    
    all_data = all_data[all_data['lower'] <= all_data['收盘价_复权']]           #lower在选股数据整理.py中已经计算
    
  

    
    return all_data


def select_stock(all_data, count):
    """
    选股函数，必要
    :param all_data: 截面数据
    :param count: 选股数量
    :return:
    """
    # 定义合并需要的list
    merge_factor_list = []
    for factor in factor_rank_dict:
        
    
        all_data[factor+'_排名'] = all_data.groupby('交易日期')[factor].rank(ascending=factor_rank_dict[factor], method='first')
        # 将计算好的因子rank添加到list中
        merge_factor_list.append(factor + '_排名')
        
    all_data['因子'] = all_data[merge_factor_list].mean(axis=1)
    # 对因子进行排名
    all_data['选股排名'] = all_data.groupby('交易日期')['因子'].rank(method='first')
    # 选取排名靠前的股票
    all_data = all_data[all_data['选股排名'] <= count]
    return all_data
