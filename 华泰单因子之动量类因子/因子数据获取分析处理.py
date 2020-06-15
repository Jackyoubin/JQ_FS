# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:15:44 2020

@author: lenovo
"""
import pandas as pd
from factor_analyzer import detecte_factor_values
import jqdatasdk as jd
jd.auth('17854120489','shafajueduan28')
print(jd.get_query_count())





#%%

'''
函数get_rng(start_year=None,end_year=None)得到的是每次指数成分股改变时的开始日期
在函数get_interval(start_year,end_year)中得到一个完整的周期
'''

from datetime import datetime
def get_rng(start_year=None,end_year=None):
    '''
    输入起始结束年份，得到300，500，800指数成分股调整日期。
    
    '''
    y = pd.date_range(start_year,end_year,freq='Y')
    start_rng = pd.Series(index=pd.date_range(start_year,end_year,freq='WOM-2FRI'))
    rng = list()
    
    def june(i):
        June = '6-'+str(i.year)
        start_date = start_rng.loc[June].index[0]
        date = pd.to_datetime(jd.get_trade_days(start_date=str(i.year)+'-06-01', end_date=str(i.year)+'-06-30'))
        
        d = date[date > start_date]
        return d[0]
    
    def dec(i):
        Dec = '12-'+str(i.year)
        start_date = start_rng.loc[Dec].index[0]
        date = pd.to_datetime(jd.get_trade_days(start_date=str(i.year)+'-12-01', end_date=str(i.year)+'-12-30'))
        
        d = date[date > start_date]
        return d[0]
    
    for i in y:
        if datetime.now() > june(i):
            rng.append(june(i))
            
        if datetime.now() > dec(i):  
            rng.append(dec(i))
        
    return rng

rng = get_rng(start_year='2010',end_year='2021')

import numpy.random as npr
def get_interval(start_year,end_year):
    
    rng = get_rng(start_year=start_year,end_year=end_year)
    
    ind = npr.randint(0,len(rng))
    start_date = rng[ind].strftime('%Y-%m-%d')
    #适当处理得到结束日期
    end_date = jd.get_trade_days(end_date=rng[ind+1],count=2)[0]

    return (start_date,end_date)

start_date,end_date = get_interval(start_year='2010',end_year='2021')








#%%
import os
import shutil

path = 'D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子之动量类因子\\%s_%s'%(start_date,end_date)
if os.path.exists(path):                                                        
    shutil.rmtree(path)
if not os.path.exists(path):
    os.makedirs(path)







#%%

'''
中证800动量类因子wgt_return

个股N日内以每日换手率作为权重对每日收益率求算术平均值
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

        #获取股票池
        all_stocks = get_stock(stockPool, date)
   
        #获取因子数据
        if factor in all_columns: #可以从财务库直接取到因子值的因子
            data_temp = get_factor_data1(factor,all_stocks, date)
        else: #可以从因子库直接取到因子值的因子
            try:
                data_temp = jd.get_factor_values(all_stocks, [factor], end_date = date, count = 1)[factor]
            except:
                print('系统暂不能获取该因子，请获取其他因子')
                break
        data = pd.concat([data, data_temp], axis = 0) 
    return data


def get_wgt_return(stockPool,date_start, date_end):
    
    def get_stock(stockPool, date):
        if stockPool == 'HS300':#用于获取沪深300股票池
            stockList = jd.get_index_stocks('000300.XSHG', date)
        elif stockPool == 'ZZ500':#用于获取中证500股票池
            stockList = jd.get_index_stocks('399905.XSHE', date)
        elif stockPool == 'ZZ800':#用于获取中证800股票池
            stockList = jd.get_index_stocks('399906.XSHE', date)   
        elif stockPool == 'A':#用于获取全部A股股票池
            stockList = jd.get_index_stocks('000002.XSHG', date) + jd.get_index_stocks('399107.XSHE', date)
        else:#自定义输入股票池
            stockList = stockPool
        return stockList 
    
    turnover_ratio=get_factor_data(stockPool, 'turnover_ratio',date_start, date_end)
    
    all_stocks = get_stock('ZZ500',date_start)
    close = jd.get_price(all_stocks,start_date=date_start, end_date=date_end,\
                frequency='daily', fields=['close'], skip_paused=False, fq='pre').loc['close']
    
    earnings_ratio = (close - close.shift(-1))/close * 100

    aa = earnings_ratio * turnover_ratio
    bb = aa.rolling(10,min_periods=10,axis=0).sum()
    cc = turnover_ratio.rolling(10,min_periods=10,axis=0).sum()
    
    wgt_return = bb/cc
    return wgt_return.dropna(axis=0,how='all')

wgt_return = get_wgt_return('ZZ500',start_date, end_date)

wgt_return.to_csv(path+'\\原始因子wgt_return.csv')

wgt_return = pd.read_csv(path+'\\原始因子wgt_return.csv',index_col='Unnamed: 0')







#%%
'''
因子数据分析之原始因子
'''

detectedactorvalues = detecte_factor_values.DetecteFactorValues(wgt_return)
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
wr= wgt_return.copy()
#处理空值
factor = wr.fillna(0)
#%%
#去极值
factor = detecte_factor_values.winsorize_med(factor, scale=3, inclusive=True, inf2nan=True, axis=1)
#%%
#标准化
factor = detecte_factor_values.standardlize(factor, inf2nan=True, axis=1)

#%%
factor.to_csv(path+'\\处理后因子pe.csv',encoding='utf_8_sig')













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










#%%
from factor_analyzer import data,analyze,prepare,utils
import statsmodels.api as sm
import numpy as np






#%%
'''
获取数据，并处理一下
'''
factor_data = pd.read_csv(path + '\\处理后因子pe.csv',\
                index_col='Unnamed: 0')
factor_data.index = pd.to_datetime(factor_data.index)






#%%
api = data.DataApi(fq='pre', industry='jq_l1', weight_method='ln_mktcap')
api.auth('17854120489', 'shafajueduan28')
wgt_return_factor = analyze.FactorAnalyzer(factor_data,prices=api.get_prices,groupby=api.get_groupby,\
            weights=api.get_weights,bins=5,periods=(1,5,10))

clean_factor_data = wgt_return_factor._clean_factor_data







#%%
#绘制各分位数各周期的平均收益（收益数值不是重点，主要用于观察是否具有单调性）
wgt_return_factor.plot_quantile_returns_bar(by_group=False, demeaned=False, group_adjust=False)








#%%
#绘制各分位数的累计收益（收益数值不是重点，看层次是否分明）
wgt_return_factor.plot_cumulative_returns_by_quantile(period=5, demeaned=False, group_adjust=False)








#%%
#分维度获得 因子收益和标准差
mean_return_by_quantile = wgt_return_factor.calc_mean_return_by_quantile(by_date=1, by_group=0, \
                                                                 demeaned=0, group_adjust=0)









#%%
# 计算指定调仓周期的各分位数每日累积收益
return_by_quantile = wgt_return_factor.calc_cumulative_return_by_quantile(period=5,\
                        demeaned=False, group_adjust=False)


return_by_quantile.to_csv(path+'\\return_by_quantile_5.csv')









#%%
#相关指标计算
def rate_of_return(period_ret):
    """
    转换回报率为"每期"回报率：如果收益以稳定的速度增长, 则相当于每期的回报率
    """
    period = int(period_ret.name.replace('period_', ''))
    return period_ret.add(1).pow(1. / period).sub(1)

def calc_indicators(period,demeaned=False, group_adjust=False,\
               benchmark='000905.XSHG',rf=0.03):
    '''
    计算相关指标
    
    参数
    -------------
    period ： int ,远期
    benchmark ： str ,基准
    rf ： float, 无风险收益
    
    返回值
    -------------
    indicator : DataFrame
    
    '''
    return_by_quantile = wgt_return_factor.calc_cumulative_return_by_quantile(period=period,\
                        demeaned=demeaned, group_adjust=group_adjust)
    
    date_return_by_quantile = wgt_return_factor.calc_mean_return_by_quantile(
            by_date=True, by_group=False,
            demeaned=demeaned, group_adjust=group_adjust,
        )[0].apply(rate_of_return, axis=0)
    date_return_by_quantile.index.name=['factor_quantile','date']
    date_return_by_quantile = date_return_by_quantile.\
            unstack('factor_quantile')['period_'+str(period)]
    
    start_date = return_by_quantile.index[0]
    end_date = return_by_quantile.index[-1]
    h,l = return_by_quantile.shape                #行，列
    
    #策略年华收益率。return: series 
    def aa(x):
        return np.power(x[-1],252/h) - 1
    total_annualized_returns = return_by_quantile.apply(aa)
    total_annualized_returns.name = 'total_annualized_returns'
    
    #基准年华收益率。 return: series
    def bb():
        close = jd.get_price(benchmark, start_date=start_date, end_date=end_date,\
                  frequency='daily', fields=['close'], skip_paused=False, fq='pre')
        
        returns = np.power(close['close'][-1] /close['close'][0],252/h) - 1
  
        benchmark_annualized_returns = pd.Series(data = [returns]*l,\
                                                 index=total_annualized_returns.index)
        return benchmark_annualized_returns
    benchmark_annualized_returns = bb()
    benchmark_annualized_returns.name = 'benchmark_annualized_returns'
    
    #Beta贝塔
    def cc():
        close = jd.get_price(benchmark, start_date=start_date, end_date=end_date,\
                  frequency='daily', fields=['close'], skip_paused=False, fq='pre')
        
        benchmark_return = ((close-close.shift(-1))/close)['close'].fillna(0)
        benchmark_var = np.var((close-close.shift(-1))/close)
        return (benchmark_return,benchmark_var)
    benchmark_return,benchmark_var = cc()
    def dd(x):
        return np.cov(x,benchmark_return)[0,1]/benchmark_var
    
    beta = date_return_by_quantile.apply(dd)
    beta = beta.T.loc[:,'close']
    beta.index = total_annualized_returns.index
    beta.name = 'bata'
        
    #Alpha收益
    def dd():
        return total_annualized_returns-(rf+beta*(benchmark_annualized_returns-rf))
    alpha = dd()
    alpha.name = 'Alpha'
    
    #策略波动率
    def ee(x):
        return np.std(x)*np.sqrt(252)
    algorithm_volatility = date_return_by_quantile.apply(ee)
    algorithm_volatility.name = 'algorithm_volatility'
    
    #夏普比率
    sharpe = (total_annualized_returns - rf)/algorithm_volatility
    sharpe.name = 'sharpe'
    
    #最大回测
    def drawdown(return_by_quantile):
        max_d = pd.DataFrame(index = return_by_quantile.columns)
        for i in range(0,len(return_by_quantile)-2):
            drawdown = (return_by_quantile.iloc[:i+1,:].max() - return_by_quantile.iloc[i,:])/ return_by_quantile.iloc[:i+1,:].max()
            drawdown.name = return_by_quantile.index[i]
            max_d = pd.concat([max_d,drawdown],axis=1)
        return max_d.T.max()
    max_drawdown = drawdown(return_by_quantile)
    max_drawdown.name = 'max_drawdown'
    
    indicator = pd.concat([total_annualized_returns,benchmark_annualized_returns,beta,\
                           alpha,algorithm_volatility,sharpe,max_drawdown],axis=1,join='inner')
    return indicator


indicator = calc_indicators(5,demeaned=False, group_adjust=False,\
               benchmark='000905.XSHG',rf=0.03)











#%%
# 计算每日因子IC值
ic_date = wgt_return_factor.calc_factor_information_coefficient(group_adjust=1, by_group=0, method='rank')











#%%
'''
一般来说IC大于3%（因子反过来的时候就小于-3%）,则认为因子比较有效。   
IC.Std  
IR=IC.Mean\IC.Std  
p-value   p值，判断IC的统计分布，一般要求小于5%或1%。在假设检验中，如果p>0.05，则接受原假设H0。如果p<0.05或p<0.01,则拒绝原假设H0,接受备择假设H1。  
IC Skew   偏度  
IC Kurtosis  峰度

'''
# 打印信息比率（IC）相关表
wgt_return_factor.plot_information_table(group_adjust=True, method='rank')
wgt_return_factor.plot_ic_hist(group_adjust=True, method='rank')


'''
IC Skew              period_1 > 0    正偏态
                     period_10<0     负偏态
                     period_20<0     负偏态
             
IC Kurtosis          period_1<3      廋尾
                     period_10<3     廋尾
                     period_20<3     廋尾
                     
p-value(IC)          <0.05            在显著性水平0.05下拒接原假设，即认为均值不为零                     
'''










#%%
def regression_test(clean_factor_data):
    '''
    用回归法进行截面 规律统计
    
    a: 回归系数
    b: 截距
    t: t检验统计量
    '''    
    
    factor = prepare.demean_forward_returns(clean_factor_data)
    
    cols = utils.get_forward_returns_columns(factor.columns)
    grouper = factor.index.get_level_values('date')
    
    def aa(df):
        s = pd.Series()
        for i in utils.get_forward_returns_columns(factor.columns):
            y = df[i]
            x = df['factor']
            x = sm.add_constant(x)
            wls_model = sm.WLS(y,x)          #暂且设置为等权重
            results = wls_model.fit()
            b,a =results.params
            t = results.tvalues[1]
            s = s.append(pd.Series([a,b,t],index=[i+'_a',i+'_b',i+'_t']))
        return s

    rt = factor.groupby(grouper)[cols.append(pd.Index(['factor']))].apply(aa)
    
    return rt

rt = regression_test(wgt_return_factor._clean_factor_data)

rt.to_csv(path+'\\regression_test.csv')








