import os
import json


# 各个文件夹的路径
_ = os.path.dirname(os.path.dirname(os.path.abspath('')))
root_path = os.path.abspath(os.path.join(_, ''))
current_path = os.getcwd()  # 获取当前工作目录


data_path=''                                   #填写数据存放路径
report_save_path=''                            #填写回测报告存放路径

# 数据路径
candle_path = ''                               # 填写K线数据存放路径
fin_path = ''                                  # 填写财务数据存放路径
index_path =''                                 # 填写指数数据存放路径

#strategies文件夹中策略脚本名称列表
stg_test_list=['Shannon']

# ===手续费
c_rate = 1.2 / 10000  # 手续费
t_rate = 1 / 1000  # 印花税

#benchmark
index_code='sh000300'

