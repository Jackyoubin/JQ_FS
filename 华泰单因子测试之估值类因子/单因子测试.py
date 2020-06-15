# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:52:25 2020

@author: lenovo
"""

import pandas as pd
import numpy as np
from factor_analyzer import data,analyze,prepare,utils
from datetime import datetime
import statsmodels.api as sm
import jqdatasdk as jd
jd.auth('17854120489','shafajueduan28')
print(jd.get_query_count())


#%%
'''
获取数据，并处理一下
'''
factor_data = pd.read_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\处理后因子pe.csv',\
                index_col='Unnamed: 0')
factor_data.index = pd.to_datetime(factor_data.index)
#减少数据量
factor_data = factor_data.loc[datetime(2018,1,1):,]






#%%
api = data.DataApi(fq='pre', industry='jq_l1', weight_method='ln_mktcap')
api.auth('17854120489', 'shafajueduan28')
pe_factor = analyze.FactorAnalyzer(factor_data,prices=api.get_prices,groupby=api.get_groupby,\
            weights=api.get_weights,bins=5,periods=(1,10,20))

clean_factor_data = pe_factor._clean_factor_data






#%%
#绘制各分位数各周期的平均收益（收益数值不是重点，主要用于观察是否具有单调性）
pe_factor.plot_quantile_returns_bar(by_group=False, demeaned=False, group_adjust=False)





#%%
#绘制各分位数的累计收益（收益数值不是重点，看层次是否分明）
pe_factor.plot_cumulative_returns_by_quantile(period=20, demeaned=True, group_adjust=False)






#%%
#分维度获得 因子收益和标准差
mean_return_by_quantile = pe_factor.calc_mean_return_by_quantile(by_date=1, by_group=0, \
                                                                 demeaned=0, group_adjust=0)




#%%
# 计算指定调仓周期的各分位数每日累积收益
return_by_quantile = pe_factor.calc_cumulative_return_by_quantile(period=20,\
                        demeaned=False, group_adjust=False)




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
    return_by_quantile = pe_factor.calc_cumulative_return_by_quantile(period=period,\
                        demeaned=demeaned, group_adjust=group_adjust)
    
    date_return_by_quantile = pe_factor.calc_mean_return_by_quantile(
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



#%%
indicator = calc_indicators(20,demeaned=False, group_adjust=False,\
               benchmark='000905.XSHG',rf=0.03)






#%%
# 计算每日因子IC值
ic_date = pe_factor.calc_factor_information_coefficient(group_adjust=0, by_group=0, method='rank')







#%%
# 打印信息比率（IC）相关表
pe_factor.plot_information_table(group_adjust=0, method='rank')






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
pe_factor.plot_ic_hist(group_adjust=False, method='rank')


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
# 画信息比率(IC)时间序列图。要求持续大于0，若出现一段时间连续小于0，则需要进行分析研究。
pe_factor.plot_ic_ts(group_adjust=False, method='rank')












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

rt = regression_test(clean_factor_data)






