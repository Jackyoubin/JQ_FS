# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:05:55 2020

@author: 17854
"""

import jqdatasdk as jd
jd.auth('17854120489','shafajueduan28')
count=jd.get_query_count()
print(count)
import pandas as pd
import matplotlib.pyplot as plt

'''
申万指数在2014-02-21有一次大改,删除了6个一级行业，并增加了11个一级行业。故：
date < 2014-02-21   申万一级行业有23个
date = 2014-02-21   申万一级行业有34个
date > 2014-02-21   申万一级行业有28个

#date='2014-02-20'，23个
code1 = jd.get_industries(name='sw_l1', date='2013-01-20')

#date='2014-02-21'有34个
code2 = jd.get_industries(name='sw_l1', date='2014-02-21')

#date='2014-02-22'有28个
code3 = jd.get_industries(name='sw_l1', date='2015-02-22')
'''

def get_sw1_valuation(start_date=None, end_date=None):
    #2014-02-22之后申万一级行业28个
    code = jd.get_industries(name='sw_l1',date='2014-02-22').index.tolist()
    days = jd.get_trade_days(start_date,end_date)
    index = jd.finance.run_query(jd.query(jd.finance.SW1_DAILY_VALUATION).filter(
            jd.finance.SW1_DAILY_VALUATION.date=='2014-02-22'
            ).limit(1)).columns.tolist()
    data =  pd.DataFrame(columns = index)
    for day in days:
        df=jd.finance.run_query(jd.query(jd.finance.SW1_DAILY_VALUATION).filter(
            jd.finance.SW1_DAILY_VALUATION.code.in_(code),
            jd.finance.SW1_DAILY_VALUATION.date==day
            ))
        name1 = set(list(map(lambda x:x[:-1],jd.get_industries(name='sw_l1',date='2014-02-22').name.tolist())))
        name2 = set(df.name.tolist())
        if not name1-name2:
            data = pd.concat([data, df], axis = 0, sort=False)
    return data
'''
df = get_sw1_valuation(start_date='2015-01-01',end_date='2020-05-01')   
df = df.set_index(['date']).drop(['id'], axis=1)
df.to_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\申万一级行业估值数据.csv',\
          encoding='utf_8_sig')
'''

def plot_fig(factor):
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    index = data.unstack('name')[factor].index
    for i in index:
        fig, ax = plt.subplots(1,1,figsize=(14,6))
        x=data.unstack('name')[factor].loc[i,:].index
        height=data.unstack('name')[factor].loc[i,:].values
        ax.set_title(i)
        ax.bar(x=x,height=height)
        ax.plot(data.unstack('name')[factor].loc[i,:])
        ax.grid(True)
        plt.xticks(rotation=30)    # 设置x轴标签旋转角度
        fig.savefig('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子'\
                    +'\\%s_%s.png'%(factor,i.strftime('%Y-%m-%d')))
       
if __name__ == '__main__':    
    df = pd.read_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\申万一级行业估值数据.csv',\
                 index_col=['date'])
    df.index = pd.to_datetime(df.index)
    data = df[['pe','pb']].groupby(df['name']).resample('Y',how='mean')
    plot_fig('pe')



        
        


