# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:36:25 2022

@author: w
"""

import pandas as pd

name = 'Shannon'  # 策略名称
period_list = ['W']  # 持仓周期,W表示周频换仓,M表示月换仓
select_count_list = [5]  # 选股数量
# 流量型财务字段：列表中对应的指标分别是：净利率、营业总成本、经营性现金流、经营活动现金流入小计
flow_fin_cols = ['R_np','R_operating_total_cost',
                 'C_ncf_from_oa','C_sub_total_of_ci_from_oa']  
# 截面型财务字段：列表中对应的指标分别是：存货、总负债、资产减值损失
cross_fin_cols = ['B_inventory','B_total_liab','R_asset_impairment_loss']  
# 最终需要加到数据上的财务字段
add_fin_cols = [
    'R_np','R_np','B_inventory','B_total_liab',
    'R_operating_total_cost','C_ncf_from_oa','C_sub_total_of_ci_from_oa',
    'R_asset_impairment_loss'
    ]  

#排序因子
factor_rank_dict={
    '总市值':True,
    }




def cal_factors(data, fin_data, fin_raw_data, exg_dict):
    """
    合并数据后计算策略需要的因子，将要在后面的函数filter_stock中使用
    :param data:传入的数据
    :param fin_data:财报数据  
    :param fin_raw_data:财报数据
    :param exg_dict:resample规则
    :return:
    """
    
    data['归母PE'] = data['总市值'] / data['R_np']
    data['归母EP'] = 1 / data['归母PE']        
    data['存货周转'] = 365 * data['B_inventory'] / (data['R_operating_total_cost'] - data['R_asset_impairment_loss'])    
    data['经营现金流量总负债比'] = data['C_ncf_from_oa'] / data['B_total_liab']    
    data['经营活动现金流入小计'] = data['C_sub_total_of_ci_from_oa']
      
    exg_dict['总市值'] = 'last'
    exg_dict['归母EP'] = 'last'
    exg_dict['存货周转'] = 'last'
    exg_dict['经营现金流量总负债比'] = 'last'
    exg_dict['经营活动现金流入小计'] = 'last'
    
    
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
    #过滤逻辑：2010年以后，去除科创板和北交所；选取存货周转、经营现金流量总负债比、归母EP_二级行业
    # 经营活动现金流入小计、总市值、散户卖出占比等指标在一定范围内的股票
    all_data = all_data[all_data['交易日期'] >= '20100101']
    all_data = all_data[all_data['股票代码'].str.contains('sh68|bj') == False]

    all_data['归母EP_二级行业分位数'] =  all_data.groupby(['交易日期', '申万二级行业名称'])['归母EP'].rank(ascending=True, pct=True)
    all_data['存货周转_分位数'] =  all_data.groupby(['交易日期'])['存货周转'].rank(ascending=False, pct=True)
    all_data['经营现金流量总负债比_分位数'] =  all_data.groupby(['交易日期'])['经营现金流量总负债比'].rank(ascending=True, pct=True)
    all_data['经营活动现金流入小计_分位数'] =  all_data.groupby(['交易日期'])['经营活动现金流入小计'].rank(ascending=True, pct=True)
    all_data['总市值分位数'] =  all_data.groupby('交易日期')['总市值'].rank(ascending=True, pct=True)



    all_data =all_data[(all_data['存货周转_分位数'] >= 0.3) & (all_data['存货周转_分位数'] <= 0.62)]
    all_data =all_data[(all_data['经营现金流量总负债比_分位数'] >= 0.06) & (all_data['经营现金流量总负债比_分位数'] <= 0.99)]
    all_data =all_data[(all_data['归母EP_二级行业分位数'] >= 0.03) & (all_data['归母EP_二级行业分位数'] <= 0.93)]
    all_data =all_data[(all_data['经营活动现金流入小计_分位数'] >= 0) & (all_data['经营活动现金流入小计_分位数'] <= 0.44)]

    all_data=all_data[all_data['总市值分位数']>=0.01]
    all_data['散户卖出占比排名'] = all_data.groupby(['交易日期'])\
        ['散户资金卖出占比'].rank(ascending=False,pct=True)
    all_data=all_data[all_data['散户卖出占比排名']>0.2 ]  
  
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
        # 对量价因子进行等权合并，生成新的因子
    all_data['因子'] = all_data[merge_factor_list].mean(axis=1)
    # 对因子进行排名
    all_data['选股排名'] = all_data.groupby('交易日期')['因子'].rank(method='first')
    # 选取排名靠前的股票
    all_data = all_data[all_data['选股排名'] <= count]
    return all_data
