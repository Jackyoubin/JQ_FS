# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:16:49 2020

@author: 17854
"""
import pandas as pd
from factor_analyzer import detecte_factor_values
import jqdatasdk as jd
jd.auth('17854120489','shafajueduan28')
print(jd.get_query_count())



#%%
'''
获取中证500估值类因子数据

1、市盈率(PE, TTM)	每股市价为每股收益的倍数，反映投资人对每元净利润所愿支付的价格，
用来估计股票的投资报酬和风险	市盈率（PE，TTM）=（股票在指定交易日期的收盘价 * 截止当日公司总股本）/归属于母公司股东的净利润TTM。
'''

#获取数据主函数
#输入股票池、指标名称、开始日期、结束日期
#返回行标签为日期，列表签为股票名称的dataframe表格
'''
为了适应在jqfactor_analyzer上分析，保持股票种类不变。
'''
def get_factor_data(stockPool, factor,date_start, date_end):
    
    #获取股票池函数
    def get_stock(stockPool, begin_date):
        if stockPool == 'HS300':#用于获取沪深300股票池
            stockList = jd.get_index_stocks('000300.XSHG', begin_date)
        elif stockPool == 'ZZ500':#用于获取中证500股票池
            stockList = jd.get_index_stocks('399905.XSHE', begin_date)
        elif stockPool == 'ZZ800':#用于获取中证800股票池
            stockList = jd.get_index_stocks('399906.XSHE', begin_date)   
        elif stockPool == 'A':#用于获取全部A股股票池
            stockList = jd.get_index_stocks('000002.XSHG', begin_date) + jd.get_index_stocks('399107.XSHE', begin_date)
        else:#自定义输入股票池
            stockList = stockPool
        return stockList    
    
    #从财务库获取数据
    def get_factor_data1(factor,stock, date):
        if factor in val:
            q = jd.query(jd.valuation).filter(jd.valuation.code.in_(stock))
            df = jd.get_fundamentals(q, date)
        elif factor in bal:
            q = jd.query(jd.balance).filter(jd.balance.code.in_(stock))
            df = jd.get_fundamentals(q, date)
        elif factor in cf:
            q = jd.query(jd.cash_flow).filter(jd.cash_flow.code.in_(stock))
            df = jd.get_fundamentals(q, date)
        elif factor in inc:
            q = jd.query(jd.income).filter(jd.income.code.in_(stock))
            df = jd.get_fundamentals(q, date)
        elif factor in ind:
            q = jd.query(jd.indicator).filter(jd.indicator.code.in_(stock))
            df = jd.get_fundamentals(q, date)

        df.index = df['code']
        data = pd.DataFrame(index = df.index)
        data[date] = df[factor]  #date是函数的参数，转置索引=列名，使得date(时间)成为索引

        return data.T
    #获取日期列表
    date_list = jd.get_trade_days(start_date = date_start, end_date = date_end)
    #空df预备存储数据
    data = pd.DataFrame(columns = get_stock(stockPool,begin_date=date_list[0]))
    
    #获取五张财务基础所有指标名称
    val = jd.get_fundamentals(jd.query(jd.valuation).limit(1)).columns.tolist()
    bal = jd.get_fundamentals(jd.query(jd.balance).limit(1)).columns.tolist()
    cf = jd.get_fundamentals(jd.query(jd.cash_flow).limit(1)).columns.tolist()
    inc = jd.get_fundamentals(jd.query(jd.income).limit(1)).columns.tolist()
    ind = jd.get_fundamentals(jd.query(jd.indicator).limit(1)).columns.tolist()
    all_columns = val+bal+cf+inc+ind
    
    all_stocks = get_stock(stockPool, date_list[0])
    #循环时间列表获取指标数据
    for date in date_list:
        '''
        #获取股票池
        all_stocks = get_stock(stockPool, date)
        '''
        #获取因子数据
        if factor in all_columns: #可以从财务库直接取到因子值的因子
            data_temp = get_factor_data1(factor,all_stocks, date)
        else: #可以从因子库直接取到因子值的因子
            try:
                data_temp = jd.get_factor_values(all_stocks, [factor], end_date = date, count = 1)[factor]
            except:
                print('系统暂不能获取该因子，请获取其他因子')
                break
        data = pd.concat([data, data_temp], axis = 0, sorted=False) 
    return data

'''
df_pe=get_factor_data('ZZ500', 'pe_ratio','2016-01-01', '2020-05-01')
df_pe.to_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\原始因子pe.csv',\
          encoding='utf_8_sig')
'''
df_pe = pd.read_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\原始因子pe.csv',\
                index_col='Unnamed: 0')

df_1_pe = 1/df_pe.copy()











#%%
'''
因子数据分析之原始因子
'''

detectedactorvalues = detecte_factor_values.DetecteFactorValues(df_1_pe)
#%%
#个别股票在某些时间段内，存在连续的空值，可能是停牌了
null_value_situation = detectedactorvalues.detecte_null_value()
#%%
statistice_factor_value = detectedactorvalues.statistice_factor_value(
                 quantile=5,value=1,interval=(0,1))

#%%
industry_situation = detectedactorvalues.factor_value_industry(industry='sw_l1')
#%%
overall_factor_value = detectedactorvalues.overall_factor_value()
#%%
detectedactorvalues.plot_scatter()
#%%
detectedactorvalues.plot_hist()












#%%%
'''
因子数据处理
'''

#处理空值
factor = df_1_pe.fillna(0)
#%%
#去极值
factor = detecte_factor_values.winsorize_med(factor, scale=3, inclusive=True, inf2nan=True, axis=1)
#%%
#标准化
factor = detecte_factor_values.standardlize(factor, inf2nan=True, axis=1)

#%%
factor.to_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\处理后因子pe.csv',\
          encoding='utf_8_sig')













#%%
'''
因子数据分析之处理过后的因子
'''

factor_detectedactorvalues = detecte_factor_values.DetecteFactorValues(factor)
#%%
factor_null_value_situation = factor_detectedactorvalues.detecte_null_value()
#%%
factor_statistice_factor_value = factor_detectedactorvalues.statistice_factor_value(
                 quantile=5,value=1,interval=(0,1))

#%%
factor_industry_situation = factor_detectedactorvalues.factor_value_industry(industry='jq_l1')
#%%
factor_overall_factor_value = factor_detectedactorvalues.overall_factor_value()

#%%
factor_detectedactorvalues.plot_scatter()

#%%
factor_detectedactorvalues.plot_hist()

