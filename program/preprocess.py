import sys
import os
#sys.path.append() #若因路径报错，则要在此处加入Shannon文件夹的绝对路径
from tqdm import tqdm
import pkgutil
import warnings
from joblib import Parallel, delayed
from program.config import *
from program.utils.Function_fin import *
from program.utils.Function import *


warnings.filterwarnings('ignore')

def cal_stock(inx, stock):
    try:
        # stock = 'sh600092.csv'
        print(stock)
        # =读入股票数据
        path = candle_path + '/%s' % stock
        df = pd.read_csv(path, encoding='gbk', skiprows=1, parse_dates=['交易日期'])
        # =计算涨跌幅
        df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
        df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1  # 为之后开盘买入做好准备

        # =计算复权价：计算所有因子当中用到的价格，都使用复权价
        df['复权因子'] = (1 + df['涨跌幅']).cumprod()
        df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
        df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']
        df['最高价_复权'] = df['最高价'] / df['收盘价'] * df['收盘价_复权']
        df['最低价_复权'] = df['最低价'] / df['收盘价'] * df['收盘价_复权']
        df['换手率'] = df['成交额'] / df['流通市值']
        df['换手率_ma'] = df['换手率'].rolling(20).mean()
        df['换手率_std'] = df['换手率'].rolling(20,min_periods=1).std(ddof=0)
        df['换手率变化率'] =df['换手率']/ df['换手率'].shift(1)-1
        df['换手率变化率_std'] = df['换手率变化率'].rolling(20,min_periods=1).std(ddof=0)
        
        df['ma_20']=df['收盘价_复权'].rolling(20).mean()
        df['std_20']=df['收盘价_复权'].rolling(20).std()
        df['z']=abs((df['收盘价_复权']-df['ma_20'])/df['std_20'])    
        df['m']=df['z'].rolling(20).mean().shift(1)
        df['lower']=df['ma_20']-df['std_20']*df['m']    #计算布林线下轨
        df['散户资金卖出占比'] = df['散户资金卖出额']*10000 / df['成交额']

        exg_dict = {
                   '收盘价_复权': 'last',
                   # '沪深300成分股': 'last',
                   # '上证50成分股': 'last',
                   # '创业板指成分股': 'last',
                   # '中证500成分股': 'last',
                   # '中证1000成分股': 'last',
                   '申万一级行业名称': 'last',
                   '申万二级行业名称': 'last',
                   '申万三级行业名称': 'last',
                   '上市至今交易天数': 'last',             
                   'lower': 'last',
                   '换手率': 'last',
                   '换手率_ma': 'last',
                   '换手率_std': 'last',
                   '换手率变化率_std': 'last',
                   '散户资金卖出额' : 'last',
                   '散户资金卖出占比' : 'last',
                  
                   }

        df['上市至今交易天数'] = df.index.astype('int') + 1
        fill_0_list = ['换手率']  # 合并指数时需要填充为0的列

        # 合并指数之前
        # for strategy in stg_list:
        #     df, exg_dict, fill_0_list = strategy.before_merge_index(df, exg_dict, fill_0_list)
        fill_0_list = list(set(fill_0_list))
        # =将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
        df = merge_with_index_data(df, inx, fill_0_list)
        if df.empty:
            return pd.DataFrame()
        # =计算涨跌停价格
        df = cal_zdt_price(df)
       
        # 处理财务数据
        flow_fin_cols = []  # 流量型财务数据
        cross_fin_cols = []  # 截面型财务数据
        add_fin_cols = []  # 需要添加到最终数据上的列
        # 获取各个策略需要的财务字段合集
        for strategy in stg_list:
            flow_fin_cols += strategy.flow_fin_cols
            cross_fin_cols += strategy.cross_fin_cols
            add_fin_cols += strategy.add_fin_cols

        # 财务字段数据去重
        flow_fin_cols = list(set(flow_fin_cols))
        cross_fin_cols = list(set(cross_fin_cols))
        add_fin_cols = list(set(add_fin_cols))
        # 开始处理财务数据
        stock = stock.replace('.csv', '')
        df, fin_df, fin_raw_df = merge_with_finance_data(df, stock, fin_path, add_fin_cols, exg_dict, flow_fin_cols,
                                                         cross_fin_cols)

        # 合并单一数据 | 计算因子
        for strategy in stg_list:
            # df, exg_dict = strategy.merge_single_stock_file(df, exg_dict)
            df, exg_dict = strategy.cal_factors(df, fin_df, fin_raw_df, exg_dict)

        # ==== 计算下个交易的相关情况
        df['下日_是否交易'] = df['是否交易'].shift(-1)
        df['下日_一字涨停'] = df['一字涨停'].shift(-1)
        df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
        df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
        df['下日_是否S'] = df['股票名称'].str.contains('S').shift(-1)
        df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
        df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)
        
      
        
        # 处理最后一根K线的数据
        state_cols = ['下日_是否交易', '下日_是否ST', '下日_是否S', '下日_是否退市']
        df.loc[:, state_cols] = df.loc[:, state_cols].fillna(method='ffill')
        df.loc[:, ['下日_一字涨停', '下日_开盘涨停']] = df.loc[:, ['下日_一字涨停', '下日_开盘涨停']].fillna(value=False)

        # ==== 将日线数据转化为月线或者周线
        df = transfer_to_period_data(df, period, exg_dict)
        for strategy in stg_list:
            df = strategy.after_resample(df)

        # =对数据进行整理
        # 删除上市的第一个周期
        df.drop([0], axis=0, inplace=True)  # 删除第一行数据
        # 删除2017年之前的数据
        df = df[df['交易日期'] > pd.to_datetime('20061215')]
        # 计算下周期每天涨幅
        df['下周期每天涨跌幅'] = df['每天涨跌幅'].shift(-1)
        df['下周期涨跌幅'] = df['涨跌幅'].shift(-1)
        
     
        del df['每天涨跌幅']

        # 删除月末不交易的周期数
        df = df[df['是否交易'] == 1]
        return df
    except Exception as err:
        print('%s发生错误:%s' % (stock, err))
        return pd.DataFrame()


if __name__ == '__main__':
    # 导入指数数据
    index_df = import_index_data(os.path.join(index_path, index_code+'.csv'))

    # 股票列表
    stock_list = get_file_in_folder(candle_path, '.csv', filters=['bj','sh688'])#filters中去除了北交所和科创板

    # 按照周期划分
    for period in ['W']:
        # 获取所有的选股策略
        stg_list = []
        # for _, file, _ in pkgutil.iter_modules([os.path.join(root_path, 'program', 'strategies', 'select')]):
        for file in stg_test_list:
            cls = __import__('program.strategies.%s' % file, fromlist=('',))
            if period in cls.period_list:
                # print('正在运行%s.special_data()' % cls.name)
                # cls.special_data()
                stg_list.append(cls)

        if len(stg_list) == 0:
            continue

        multiply_process = True
        if multiply_process:
            df_list = Parallel(os.cpu_count())(delayed(cal_stock)(index_df, stock) for stock in tqdm(stock_list))
        else:
            df_list = []
            for stock in stock_list:
                data = cal_stock(index_df, stock)
                df_list.append(data)

        all_data = pd.concat(df_list, ignore_index=True)
        all_data.sort_values(by=['交易日期', '股票代码'], inplace=True)
        # print(all_data.tail(5))
        all_data.to_pickle(os.path.dirname(current_path) + '/data/preprocess/all_data_%s.pkl' % period)
