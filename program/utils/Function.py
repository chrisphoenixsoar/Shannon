import itertools
import os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


 

def get_file_in_folder(path, file_type, contains=None, filters=[], drop_type=False):
    """
    获取指定文件夹下的文件
    :param path: 文件夹路径
    :param file_type: 文件类型
    :param contains: 需要包含的字符串，默认不含
    :param filters: 字符串中需要过滤掉的内容
    :param drop_type: 是否要保存文件类型
    :return:
    """
    file_list = os.listdir(path)
    file_list = [file for file in file_list if file_type in file]
    if contains:
        file_list = [file for file in file_list if contains in file]
    for con in filters:
        file_list = [file for file in file_list if con not in file]
    if drop_type:
        file_list = [file[:file.rfind('.')] for file in file_list]

    return file_list


def import_index_data(path, start=None, end=None,switch=0):
    """
    从指定位置读入指数数据。指数数据来自于：program/构建自己的股票数据库/案例_获取股票最近日K线数据.py
    :param end: 结束时间
    :param start: 开始时间
    :param path:指数数据的路径
    :return:
    """
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'])
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    
    ############新增，用于指数对冲功能###############
    df_index['指数开盘买入涨跌幅'] = df_index['close'] / df_index['open'] - 1 
    if switch==0:
        
        df_index = df_index[['candle_end_time', '指数涨跌幅','指数开盘买入涨跌幅']]
    else:
        pass
    ################################################
    
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)
    # 截取数据开始时间和结束时间
    if start:
        df_index = df_index[df_index['交易日期'] >= pd.to_datetime(start)]
    if end:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(end)]
    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


def merge_with_index_data(df, index_data, extra_fill_0_list=[]):
    """
    原始股票数据在不交易的时候没有数据。
    将原始股票数据和指数数据合并，可以补全原始股票数据没有交易的日期。
    :param df: 股票数据
    :param index_data: 指数数据
    :param extra_fill_0_list: 合并时需要填充为0的字段
    :return:
    """
    # ===将股票数据和上证指数合并，结果已经排序
    df = pd.merge(left=df, right=index_data, on='交易日期', how='right', sort=True, indicator=True)

    # ===对开、高、收、低、前收盘价价格进行补全处理
    # 用前一天的收盘价，补全收盘价的空值
    df['收盘价'].fillna(method='ffill', inplace=True)
    # 用收盘价补全开盘价、最高价、最低价的空值
    df['开盘价'].fillna(value=df['收盘价'], inplace=True)
    df['最高价'].fillna(value=df['收盘价'], inplace=True)
    df['最低价'].fillna(value=df['收盘价'], inplace=True)
    # 补全前收盘价
    df['前收盘价'].fillna(value=df['收盘价'].shift(), inplace=True)

    # ===将停盘时间的某些列，数据填补为0
    fill_0_list = ['成交量', '成交额', '涨跌幅', '开盘买入涨跌幅'] + extra_fill_0_list
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    # ===用前一天的数据，补全其余空值
    df.fillna(method='ffill', inplace=True)

    # ===去除上市之前的数据
    df = df[df['股票代码'].notnull()]

    # ===判断计算当天是否交易
    df['是否交易'] = 1
    df.loc[df['_merge'] == 'right_only', '是否交易'] = 0
    del df['_merge']
    df.reset_index(drop=True, inplace=True)
    return df


def cal_zdt_price(df):
    """
    计算股票当天的涨跌停价格。在计算涨跌停价格的时候，按照严格的四舍五入。
    包含st股，但是不包含新股
    涨跌停制度规则:
        ---2020年8月23日
        非ST股票 10%
        ST股票 5%

        ---2020年8月24日至今
        普通非ST股票 10%
        普通ST股票 5%

        科创板（sh68） 20%（一直是20%，不受时间限制）
        创业板（sz30） 20%
        科创板和创业板即使ST，涨跌幅限制也是20%

        北交所（bj） 30%

    :param df: 必须得是日线数据。必须包含的字段：前收盘价，开盘价，最高价，最低价
    :return:
    """
    # 计算涨停价格
    # 普通股票
    cond = df['股票名称'].str.contains('ST')
    df['涨停价'] = df['前收盘价'] * 1.1
    df['跌停价'] = df['前收盘价'] * 0.9
    df.loc[cond, '涨停价'] = df['前收盘价'] * 1.05
    df.loc[cond, '跌停价'] = df['前收盘价'] * 0.95

    # 科创板 20%
    rule_kcb = df['股票代码'].str.contains('sh68')
    # 2020年8月23日之后涨跌停规则有所改变
    # 新规的创业板
    new_rule_cyb = (df['交易日期'] > pd.to_datetime('2020-08-23')) & df['股票代码'].str.contains('sz30')
    # 北交所条件
    cond_bj = df['股票代码'].str.contains('bj')

    # 科创板 & 创业板
    df.loc[rule_kcb | new_rule_cyb, '涨停价'] = df['前收盘价'] * 1.2
    df.loc[rule_kcb | new_rule_cyb, '跌停价'] = df['前收盘价'] * 0.8

    # 北交所
    df.loc[cond_bj, '涨停价'] = df['前收盘价'] * 1.3
    df.loc[cond_bj, '跌停价'] = df['前收盘价'] * 0.7

    # 四舍五入
    df['涨停价'] = df['涨停价'].apply(lambda x: float(Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))
    df['跌停价'] = df['跌停价'].apply(lambda x: float(Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))

    # 判断是否一字涨停
    df['一字涨停'] = False
    df.loc[df['最低价'] >= df['涨停价'], '一字涨停'] = True
    # 判断是否一字跌停
    df['一字跌停'] = False
    df.loc[df['最高价'] <= df['跌停价'], '一字跌停'] = True
    # 判断是否开盘涨停
    df['开盘涨停'] = False
    df.loc[df['开盘价'] >= df['涨停价'], '开盘涨停'] = True
    # 判断是否开盘跌停
    df['开盘跌停'] = False
    df.loc[df['开盘价'] <= df['跌停价'], '开盘跌停'] = True

    return df


def _factors_linear_regression(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法外部不可直接调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """
    # print(data['交易日期'].to_list()[0])
    lrm = LinearRegression(fit_intercept=True)  # 创建线性回归模型
    if industry:  # 如果需要对行业进行中性化，将行业的列名加入到neutralize_list中
        industry_cols = [col for col in data.columns if '所属行业' in col]
        for col in industry_cols:
            if col not in neutralize_list:
                neutralize_list.append(col)
    train = data[neutralize_list].copy()  # 输入变量
    label = data[[factor]].copy()  # 预测变量
    lrm.fit(train, label)  # 线性拟合
    predict = lrm.predict(train)  # 输入变量进行预测
    data[factor + '_中性'] = label.values - predict  # 计算残差
    return data


def factor_neutralization(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法可以被外部调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor] + neutralize_list, how='any')
    if industry:  # 果需要对行业进行中性化，先构建行业哑变量
        # 剔除中性化所涉及的字段中，包含inf、-inf、nan的部分
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[industry], how='any')
        # 对行业进行哑变量处理
        ind = df[industry]
        ind = pd.get_dummies(ind, columns=[industry], prefix='所属行业',
                             prefix_sep="_", dummy_na=False, drop_first=True)
        """
        drop_first=True会导致某一行业的的哑变量被删除，这样的做的目的是为了消除行业间的多重共线性
        详见：https://www.learndatasci.com/glossary/dummy-variable-trap/
        """
    else:
        ind = pd.DataFrame()
    df = pd.concat([df, ind], axis=1)
    df = df.groupby(['交易日期']).apply(_factors_linear_regression, factor=factor,
                                    neutralize_list=neutralize_list, industry=industry)
    # df[factor + '_中性'] = df.groupby(['交易日期']).apply(_factors_linear_regression, factor=factor,
    #                                                 neutralize_list=neutralize_list, industry=industry)
    df.sort_values(by=['交易日期', '股票代码'], inplace=True)
    return df




def transfer_to_period_data(df, period_type='m', extra_agg_dict={}):
    """
    将日线数据转换为相应的周期数据
    :param df:原始数据
    :param period_type:转换周期
    :param extra_agg_dict:
    :return:
    """

    # 将交易日期设置为index
    df['周期最后交易日'] = df['交易日期']
    df.set_index('交易日期', inplace=True)

    agg_dict = {
        # 必须列
        '周期最后交易日': 'last',
        '股票代码': 'last',
        '股票名称': 'last',
        '是否交易': 'last',

        '开盘价': 'first',
        '最高价': 'max',
        '最低价': 'min',
        '收盘价': 'last',
        '成交额': 'sum',
        '流通市值': 'last',
        '总市值': 'last',
        '上市至今交易天数': 'last',

        '下日_是否交易': 'last',
        '下日_开盘涨停': 'last',
        '下日_是否ST': 'last',
        '下日_是否S': 'last',
        '下日_是否退市': 'last',
        '下日_开盘买入涨跌幅': 'last',
    }
    agg_dict = dict(agg_dict, **extra_agg_dict)
    period_df = df.resample(rule=period_type).agg(agg_dict)

    # 计算必须额外数据
    period_df['交易天数'] = df['是否交易'].resample(period_type).sum()
    period_df['市场交易天数'] = df['股票代码'].resample(period_type).size()
    period_df = period_df[period_df['市场交易天数'] > 0]  # 有的时候整个周期不交易（例如春节、国庆假期），需要将这一周期删除

    # 计算其他因子
    # 计算周期资金曲线
    period_df['每天涨跌幅'] = df['涨跌幅'].resample(period_type).apply(lambda x: list(x))
    period_df['涨跌幅'] = df['涨跌幅'].resample(period_type).apply(lambda x: (x + 1).prod() - 1)
    
 
    # 重新设定index
    period_df.reset_index(inplace=True)
    period_df['交易日期'] = period_df['周期最后交易日']
    del period_df['周期最后交易日']

    return period_df


def create_empty_data(index_data, period):
    """
    创建一个空的持仓数据
    :param index_data: 指数数据
    :param period: 时间周期
    :return:
    """
    empty_df = index_data[['交易日期']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['周期最后交易日'] = empty_df['交易日期']
    empty_df.set_index('交易日期', inplace=True)
    agg_dict = {'周期最后交易日': 'last'}
    empty_period_df = empty_df.resample(rule=period).agg(agg_dict)
    empty_period_df['每天涨跌幅'] = empty_df['涨跌幅'].resample(period).apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['周期最后交易日'], inplace=True)
    empty_period_df['选股下周期每天涨跌幅'] = empty_period_df['每天涨跌幅'].shift(-1)
    empty_period_df.dropna(subset=['选股下周期每天涨跌幅'], inplace=True)

    # 填仓其他列
    empty_period_df['股票数量'] = 0
    empty_period_df['买入股票代码'] = 'empty'
    empty_period_df['买入股票名称'] = 'empty'
    empty_period_df['选股下周期涨跌幅'] = 0.0
   
    empty_period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)
    empty_period_df.set_index('交易日期', inplace=True)
    empty_period_df = empty_period_df[['股票数量', '买入股票代码', '买入股票名称', \
                '选股下周期涨跌幅', '选股下周期每天涨跌幅']]
    return empty_period_df


def equity_to_csv(equity, strategy_name, period, select_stock_num, folder_path):
    """
    输出策略轮动对应的文件
    :param equity: 策略资金曲线
    :param strategy_name: 策略名称
    :param period: 周期
    :param select_stock_num: 选股数
    :param folder_path: 输出路径
    :return:
    """
    period_dict = {  # 周期对应的字典，暂时不兼容工作月的策略
        'W': 'week',
        'M': 'natural_month'
    }
    to_csv_path = folder_path + f'/{strategy_name}_{period_dict[period]}_{select_stock_num}.csv'
    equity['策略名称'] = strategy_name + '_' + period_dict[period] + '_' + str(select_stock_num)
   
    equity = equity[['交易日期', '策略名称', '持有股票代码', '涨跌幅', 'equity_curve', '指数涨跌幅', 'benchmark']]
    equity.to_csv(to_csv_path, encoding='gbk', index=False, mode='a')


def save_select_result(path, new_res, name, period, count):
    """
    保存选股数据的最新结果
    :param path: 保存结果的文件夹路劲
    :param new_res: 选股数据
    :param name: 策略名称
    :param period: 持仓周期
    :param count: 选股数据
    :param signal: 择时信号
    :return:
    """
    period_dict = {  # 周期对应的字典，暂时不兼容工作月的策略
        'W': 'week',
        'M': 'natural_month'
    }
    # 构建保存的路径
    file_path = path + '/%s_%s_%s.csv' % (name, period_dict[period], count)
    # 申明历史选股结果的变量
    res_df = pd.DataFrame()
    # 如果有历史结果，则读取历史结果
    if os.path.exists(file_path):
        res_df = pd.read_csv(file_path, encoding='gbk', parse_dates=['交易日期'])
    new_res = new_res[['交易日期', '股票代码', '股票名称', '选股排名']]
    # if signal != 1:
    #     new_res = pd.DataFrame(columns=['交易日期', '股票代码', '股票名称', '选股排名'])

    # 将历史选股结果与最新选股结果合并
    res_df = pd.concat([res_df, new_res], ignore_index=True)
    # 清洗数据，保存结果
    res_df.drop_duplicates(subset=['交易日期', '选股排名'], keep='last', inplace=True)
    res_df.sort_values(by=['交易日期', '选股排名'], inplace=True)
    res_df.to_csv(file_path, encoding='gbk', index=False)



def create_empty_data_for_strategy(bench_data, trans_period):
    """
    根据基准策略创建空的周期数据，用于填充不交易的日期
    :param bench_data: 基准策略
    :param trans_period: 数据转换周期
    :return:
    """
    empty_df = bench_data[['交易日期']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['周期最后交易日'] = empty_df['交易日期']
    empty_df.set_index('交易日期', inplace=True)
    agg_dict = {'周期最后交易日': 'last'}
    empty_period_df = empty_df.resample(rule=trans_period).agg(agg_dict)

    empty_period_df['每天涨跌幅'] = empty_df['涨跌幅'].resample(trans_period).apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['周期最后交易日'], inplace=True)

    empty_period_df['下周期每天涨跌幅'] = empty_period_df['每天涨跌幅'].shift(-1)
    empty_period_df.dropna(subset=['下周期每天涨跌幅'], inplace=True)

    # 填仓其他列
    empty_period_df['策略数量'] = 0
    empty_period_df['策略名称'] = 'empty'
    empty_period_df['持有股票代码'] = 'empty'
    empty_period_df['选股下周期涨跌幅'] = 0.0
    empty_period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)
    empty_period_df.set_index('交易日期', inplace=True)

    empty_period_df = empty_period_df[['策略名称', '持有股票代码', '策略数量', '选股下周期涨跌幅', '下周期每天涨跌幅']]
    return empty_period_df


def transfer_strategy_data(df, period_type='m', extra_agg_dict={}):
    """
    将日线数据转换为相应的周期数据
    :param df:数据
    :param period_type:持仓周期
    :param extra_agg_dict:合并规则
    :return:
    """
    # 将交易日期设置为index
    df['周期最后交易日'] = df['交易日期']
    df.set_index('交易日期', inplace=True)

    agg_dict = {
        # 必须列
        '周期最后交易日': 'last',
        '策略名称': 'last',
        '持有股票代码': 'last',
    }
    agg_dict = dict(agg_dict, **extra_agg_dict)
    period_df = df.resample(rule=period_type).agg(agg_dict)

    # 剔除不交易的数据
    period_df.dropna(subset=['策略名称'], inplace=True)

    # 计算周期资金曲线
    period_df['每天涨跌幅'] = df['涨跌幅'].resample(period_type).apply(lambda x: list(x))

    # 重新设定index
    period_df.reset_index(inplace=True)
    period_df['交易日期'] = period_df['周期最后交易日']
    del period_df['周期最后交易日']
    return period_df


