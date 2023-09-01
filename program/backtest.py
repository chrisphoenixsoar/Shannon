import sys
import os
#sys.path.append() #若因路径报错，则要在此处加入Shannon文件夹的绝对路径
from tqdm import tqdm
import pkgutil
import warnings
import quantstats as qs
from program.config import *
from program.utils.Function import *
from program.utils.Evaluate import *

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
# 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


# ===导入数据
# 导入指数数据
index_data = import_index_data(os.path.join(index_path, f'{index_code}.csv'))

for period in ['W']:
    # 获取所有的选股策略
    stg_list = []
    # for _, file, _ in pkgutil.iter_modules([os.path.join(root_path, 'program', 'strategies', 'select')]):
    for file in stg_test_list:
        cls = __import__('program.strategies.%s' % file, fromlist=('',))
        if period in cls.period_list:
            stg_list.append(cls)
    if len(stg_list) == 0:
        continue

    # 从pickle文件中读取整理好的所有股票数据
    all_df = pd.read_pickle(os.path.dirname(current_path) + \
                            '/data/preprocess/all_data_' + period + '.pkl')
    # 创建空的df
    # 创造空的事件周期表，用于填充不选股的周期
    ini_empty_df = create_empty_data(index_data.copy(), period)
    for strategy in stg_list:
        strategy_df = all_df.copy()  # 复制基础数据
        # 过滤股票数据
        strategy_df = strategy.filter_stock(strategy_df)
        for count in strategy.select_count_list:
            print('\n', '=' * 10, '正在回测：%s  %s  %s' % (strategy.name, period, count), '=' * 10, )
            df = strategy_df.copy()
            empty_df = ini_empty_df.copy()  # 复制空数据
            # 选股
            df = strategy.select_stock(df, count)

            # 记录一下最新的选股结果
            new_result = df[df['下周期每天涨跌幅'].isna() & (df['交易日期'] == df['交易日期'].max())].copy()
            # 删除数据
            df.dropna(subset=['下周期每天涨跌幅'], inplace=True)

            # ===选股
            # ===按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
            # 即将下周期每天的涨跌幅中第一天的涨跌幅，改成由开盘买入的涨跌幅
            df['下日_开盘买入涨跌幅'] = df['下日_开盘买入涨跌幅'].apply(lambda x: [x])
            df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: x[1:])
            df['下周期每天涨跌幅'] = df['下日_开盘买入涨跌幅'] + df['下周期每天涨跌幅']
            
            # ===整理选中股票数据
            # 挑选出选中股票
            df['股票代码'] += ' '
            df['股票名称'] += ' '
            group = df.groupby('交易日期')
            select_stock = pd.DataFrame()
            select_stock['股票数量'] = group['股票名称'].size()
            select_stock['买入股票代码'] = group['股票代码'].sum()
            select_stock['买入股票名称'] = group['股票名称'].sum()

            # 计算下周期每天的资金曲线
            select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(
                lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))

            # 扣除买入手续费
            select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
            # 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
            select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(
                lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])

            # 计算下周期整体涨跌幅
            select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
            # 计算下周期每天的涨跌幅
            select_stock['选股下周期每天涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(
                lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
            del select_stock['选股下周期每天资金曲线']
            

            # 将选股结果更新到empty_df上
            empty_df.update(select_stock)
            select_stock = empty_df
            # 剔除策略没选到股票的情况
            select_stock = select_stock[select_stock['股票数量'].expanding().sum() > 1]

            # 计算整体资金曲线
            select_stock.reset_index(inplace=True)
            select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()

            # ===计算选中股票每天的资金曲线
            # 计算每日资金曲线
            equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                              how='left', sort=True)  # 将选股结果和大盘指数合并

            equity['持有股票代码'] = equity['买入股票代码'].shift()
            equity['持有股票代码'].fillna(method='ffill', inplace=True)
            equity.dropna(subset=['持有股票代码'], inplace=True)
            del equity['买入股票代码']
            equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()        
            equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()
            equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
            # 输出策略回测结果
            folder_path = os.path.dirname(current_path) + '/data/backtest_result/'
            equity_to_csv(equity, strategy.name, period, count, folder_path)

            # ===计算策略评价指标
            rtn, year_return, month_return = strategy_evaluate(equity, select_stock)
            print(rtn)
            print(year_return)

            # ===生成回测报告
            equity = equity.reset_index()
            # pic_title = '策略名称：%s  选股周期：%s  选股数：%s' % (strategy.name, period, count)
            # curve_dict = {'策略表现': 'equity_curve', '基准涨跌幅': 'benchmark'}
            # draw_equity_curve_plotly(equity, data_dict=curve_dict, date_col='交易日期',right_axis={'回撤':'dd'}, title=pic_title)
            benchmark= pd.read_csv(os.path.join(index_path, f'{index_code}.csv'),encoding='gbk')
            benchmark.index=pd.to_datetime(benchmark['candle_end_time'])
            benchmark=benchmark[(benchmark.index>=equity.loc[0,'交易日期']) &\
                                (benchmark.index<=equity.loc[len(equity)-1,'交易日期'])]
            benchmark[f'{index_code}']=benchmark['close']
            equity.index=equity['交易日期']
            qs.reports.html(returns=equity['equity_curve'],benchmark=benchmark[f'{index_code}'],\
                            output=report_save_path+fr'/report_{strategy.name}_{period}_{count}.html',\
                   title=f"{strategy.name}")    
