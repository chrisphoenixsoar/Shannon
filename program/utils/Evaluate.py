import math
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots


# 通用评价函数

def draw_equity_curve_mat(df, data_dict, date_col=None, right_axis=None, pic_size=[16, 9], dpi=72, font_size=25,
                          log=False, chg=False, title=None, y_label='净值'):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param dpi: 图片的dpi
    :param font_size: 字体大小
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param log: 是都要算对数收益率
    :param title: 标题
    :param y_label: Y轴的标签
    :return:
    """
    # 复制数据
    draw_df = df.copy()
    
    # draw_df['净值_max']=draw_df['净值'].expanding().max()
    # draw_df['dd']=draw_df['净值']/draw_df['净值_max']-1
    
    # 模块基础设置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.style.use('dark_background')

    plt.figure(num=1, figsize=(pic_size[0], pic_size[1]), dpi=dpi)
    # 获取时间轴
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index
    # 绘制左轴数据
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        if log:
            draw_df[data_dict[key]] = np.log(draw_df[data_dict[key]])
        plt.plot(time_data, draw_df[data_dict[key]], linewidth=2, label=str(key))
    # 设置坐标轴信息等
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(loc=0, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.grid()
    if title:
        plt.title(title, fontsize=font_size)

    # 绘制右轴数据
    if right_axis:
        # 生成右轴
        ax_r = plt.twinx()
        # 获取数据
        key = list(right_axis.keys())[0]
        ax_r.plot(time_data, draw_df[right_axis[key]], 'y', linewidth=1, label=str(key))
        # 设置坐标轴信息等
        ax_r.set_ylabel(key, fontsize=font_size)
        ax_r.legend(loc=1, fontsize=font_size)
        ax_r.tick_params(labelsize=font_size)
    plt.show()


def draw_equity_curve_plotly(df, data_dict, date_col=None, right_axis=None, pic_size=[1500, 800], log=False, chg=False,
                             title=None, path='./pic.html', show=True):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param log: 是都要算对数收益率
    :param title: 标题
    :param path: 图片路径
    :param show: 是否打开图片
    :return:
    """
    draw_df = df.copy()
    
    draw_df['equity_curve_max']=draw_df['equity_curve'].expanding().max()
    draw_df['dd']=draw_df['equity_curve']/draw_df['equity_curve_max']-1

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 绘制左轴数据
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[data_dict[key]], name=key, ))

    # 绘制右轴数据
    if right_axis:
        # for key in list(right_axis.keys()):
        key = list(right_axis.keys())[0]
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                 marker=dict(color='rgba(220, 220, 220, 0.8)'), yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title, hovermode='x')
    # 是否转为log坐标系
    if log:
        fig.update_layout(yaxis_type="log")
    plot(figure_or_data=fig, filename=path, auto_open=False)

    # 打开图片的html文件，需要判断系统的类型
    if show:
        res = os.system('start ' + path)
        if res != 0:
            os.system('open ' + path)


#  选股策略评价函数

def strategy_evaluate(equity, select_stock):
    """
    :param equity:  每天的资金曲线
    :param select_stock: 每周期选出的股票
    :return:
    """

    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity['equity_curve'].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity['equity_curve'].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['交易日期'].iloc[-1] - equity['交易日期'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity['max2here'] = equity['equity_curve'].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['交易日期'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['交易日期']
    # 将无关的变量删除
    equity.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)

    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)

    # ===统计每个周期
    results.loc[0, '盈利周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0])  # 盈利笔数
    results.loc[0, '亏损周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0])  # 亏损笔数
    results.loc[0, '胜率'] = format(results.loc[0, '盈利周期数'] / len(select_stock), '.2%')  # 胜率
    results.loc[0, '每周期平均收益'] = format(select_stock['选股下周期涨跌幅'].mean(), '.2%')  # 每笔交易平均盈亏
    results.loc[0, '盈亏收益比'] = round(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0]['选股下周期涨跌幅'].mean() / \
                                    select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0]['选股下周期涨跌幅'].mean() * (-1), 2)  # 盈亏比
    results.loc[0, '单周期最大盈利'] = format(select_stock['选股下周期涨跌幅'].max(), '.2%')  # 单笔最大盈利
    results.loc[0, '单周期大亏损'] = format(select_stock['选股下周期涨跌幅'].min(), '.2%')  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, '最大连续盈利周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] > 0, 1, np.nan))])  # 最大连续盈利次数
    results.loc[0, '最大连续亏损周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===每年、每月收益率
    equity.set_index('交易日期', inplace=True)
    year_return = equity[['涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    monthly_return = equity[['涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)

    return results.T, year_return, monthly_return


def strategies_describe(equity, period):
    """
    分析所有子策略在轮动中的表现
    :param equity: 回测结果
    :param period: 轮动周期
    :return:
    """
    res = pd.DataFrame()
    resample_dict = {'策略名称': 'last', 'equity_curve': 'last', 'benchmark': 'last'}
    data = equity.resample(rule=period).agg(resample_dict)
    data.dropna(subset=['策略名称'], inplace=True)
    data['equity_pct'] = data['equity_curve'].pct_change()
    data['equity_pct'].fillna(value=data['equity_curve'] - 1, inplace=True)
    data['benchmark_pct'] = data['benchmark'].pct_change()
    data['benchmark_pct'].fillna(value=data['benchmark'] - 1, inplace=True)
    data['超额收益'] = data['equity_pct'] - data['benchmark_pct']

    groups = data.groupby(['策略名称'])
    for t, g in groups:
        max_inx = res.index.max()
        new_inx = 0 if math.isnan(max_inx) else max_inx + 1
        res.loc[new_inx, '策略名称'] = t
        res.loc[new_inx, '入选次数'] = str(int(g.shape[0]))
        res.loc[new_inx, '平均涨幅'] = str(round(g['equity_pct'].mean() * 100, 2)) + '%'
        res.loc[new_inx, '涨幅中值'] = str(round(g['equity_pct'].median() * 100, 2)) + '%'
        res.loc[new_inx, '最大涨幅'] = str(round(g['equity_pct'].max() * 100, 2)) + '%'
        res.loc[new_inx, '最小涨幅'] = str(round(g['equity_pct'].min() * 100, 2)) + '%'

        res.loc[new_inx, '平均超额'] = str(round(g['超额收益'].mean() * 100, 2)) + '%'
        res.loc[new_inx, '超额中值'] = str(round(g['超额收益'].median() * 100, 2)) + '%'

        win_rate = g[g['equity_pct'] > 0].shape[0] / g['equity_pct'].shape[0]
        res.loc[new_inx, '涨幅胜率'] = str(round(win_rate * 100, 2)) + '%'

        win_rate = g[g['benchmark_pct'] > 0].shape[0] / g['equity_pct'].shape[0]
        res.loc[new_inx, '超额胜率'] = str(round(win_rate * 100, 2)) + '%'

    return res



def _get_hold_net_value(zdf_list, hold_period):
    """
    输入涨跌幅数据，根据持有期输出净值数据。（这是个内部函数）
    :param zdf_list: 涨跌幅数据
    :param hold_period: 持有期
    :return:
    """
    zdf_count = len(zdf_list)
    if zdf_count < hold_period:
        zdf_list = zdf_list
    else:
        zdf_list = zdf_list[:hold_period]
    net_value = np.cumprod(np.array(list(zdf_list)) + 1)
    return net_value


def get_max_trade(all_stock_data, df, hold_period, stock_num=5, date_col='交易日期'):
    """
    获取回测结果中盈利（亏损）最大的几次交易
    :param all_stock_data: 所有数据的集合
    :param df: 回测的结果（back_test）
    :param hold_period: 持有期数据
    :param stock_num: 需要查看的最大盈利（亏损）的个数
    :param date_col: 时间列的名字
    :return:
    """
    # 先从回测结果中取出真正下单的数据
    buy_date = df[df['投出资金'] > 0][date_col].to_list()
    # 从all_data中保留真正下单买入的数据
    real_buy = all_stock_data[all_stock_data[date_col].isin(buy_date)].copy()

    # 计算每笔交易的净值
    real_buy['持仓每日净值'] = real_buy['未来N日涨跌幅'].apply(_get_hold_net_value, hold_period=hold_period)
    real_buy['最终涨跌幅'] = real_buy['持仓每日净值'].apply(lambda x: x[-1] - 1)

    # 只保留必要的列
    real_buy = real_buy[[date_col, '股票代码', '股票名称', '最终涨跌幅']]

    # 获取盈利最多的数据
    profit_max = real_buy.sort_values(by='最终涨跌幅', ascending=False).head(stock_num).reset_index(drop=True)
    # 获取亏算最大的数据
    loss_max = real_buy.sort_values(by='最终涨跌幅', ascending=True).head(stock_num).reset_index(drop=True)

    return profit_max, loss_max


