# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:32:40 2019

@author: lenovo
"""

import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt 
import numpy as np
from functools import partial

class InputValueError(ValueError):
    pass

class DetecteFactorValues(object):
    
    def __init__(self,factor):
        self.factor=factor
        if not isinstance(self.factor,pd.DataFrame):
            raise ValueError(" Parameter 'factor' type error,should be 'pd.DataFrame'")
    
    def detecte_null_value(self,factor_data=None):
        '''
        统计空值情况
        
        参数
        -----------
        self.factor :  Series  
        
        返回值
        -----------
        null_value_situation : dict
            键名:  null_value_ratio ： int ,空值占比
                   null_value_data ：Series, 空值，
                      index 为日期 (level 0) 和资产(level 1) 的 MultiIndex，value为空值
                   null_value_row : DataFrame ,每天出现空值的情况
                      index 为日期，列 num 为空值数目，列 ratio 为空值占比
                   null_value_column ：DataFrame ,每种资产出现空值的情况
                      index 为资产，列 num 为空值数目，列 ratio 为空值占比
        '''
        def ratio_1(group,direction):
            if direction=='column':
                num=len(group)
                ratio=round(num/column,3)
                return pd.DataFrame({'num':[num],'ratio':[ratio]},index=['level_1'])
            if direction=='row':
                num=len(group)
                ratio=round(num/row,3)
                return pd.DataFrame({'num':[num],'ratio':[ratio]})
            
        if factor_data is None:     
            factor_data=self.factor.copy()            
            factor_data.index.name='date'
            factor_data.columns.name='asset'
            row,column=factor_data.shape
            factor_data = factor_data.stack('asset',dropna=False) 
            factor_data.name='value'
        elif isinstance(factor_data,pd.Series):
            row,column=factor_data.unstack().shape
            factor_data.name='value'
            factor_data.index.names=['date','asset']            
                    
        null_value_ratio=1-len(factor_data.dropna())/len(factor_data)
        
        null_value_data=factor_data[factor_data.isnull()]
        
        if len(null_value_data)==0:
            return ('factor中不含空值')
        
        null_value_row=null_value_data.groupby('date').apply(ratio_1,direction='column').\
                        reset_index().drop(['level_1'],axis=1).set_index('date')
                        
        null_value_column=null_value_data.groupby('asset').apply(ratio_1,direction='row').\
                        reset_index().drop(['level_1'],axis=1).set_index('asset')
                        
        null_value_situation={'null_value_ratio':round(null_value_ratio,3),'null_value_data':null_value_data,\
                        'null_value_row':null_value_row,'null_value_column':null_value_column}
        
        return null_value_situation
        
    def statistice_factor_value(self,factor_data=None,quantile=None,value=None,interval=None):
        '''
        统计因子值得分布情况
        
        参数
        -------------
        quantile : int,划分多少分位
        interval : tuple，因子值的某个区间
        
        返回值
        -------------
        statistice_situation : dict
            键名: 
                  statistice_cut : DataFrame , 均分
                  statistice_qcut : DataFrame , 等分
                  value_int ：int , 因子值为某一值的概率
                  value_int_column ：DataFrame ，列，因子值为某一值的概率
                  value_int_row ：DataFrame ，行，因子值为某一值的概率
                  interval_int : int, 因子值在某一范围(闭区间)的概率
                  interval_tuple_column ：DataFrame，列，因子值在某一范围(闭区间)的概率
                  interval_tuple_row ：DataFrame，行，因子值在某一范围(闭区间)的概率
        '''
        
        if factor_data is None:     
            factor_data=self.factor.copy()            
            factor_data.index.name='date'
            factor_data.columns.name='asset'
            row,column=factor_data.shape
            factor_data = factor_data.stack('asset',dropna=False) 
            factor_data.name='value'
            factor_data[np.isinf(factor_data)]=np.nan
        elif isinstance(factor_data,pd.Series):
            row,column=factor_data.unstack().shape
            factor_data.name='value'
            factor_data.index.names=['date','asset']     
            factor_data[np.isinf(factor_data)]=np.nan
            
        factor_value_statistice=dict()
         
        def calculate(group):
            num=len(group)
            ratio=round(len(group)/len(factor_data),3)
            return pd.Series([num,ratio],index=['num','ratio'])
        
        if isinstance(quantile,int):
            statistice_cut=pd.DataFrame(pd.cut(factor_data,quantile)).groupby('value').apply(calculate)
            statistice_qcut=pd.DataFrame(pd.qcut(factor_data,quantile)).groupby('value').apply(calculate)
            factor_value_statistice.update({'statistice_cut':statistice_cut, 'statistice_qcut':statistice_qcut})
        elif quantile is None:
            pass
        else:
            raise ValueError("The parameter 'quantile' types is not int")        
        
        def ratio_2(group,direction):
            if direction=='column':
                num=len(group)
                ratio=round(num/column,3)
                return pd.DataFrame({'num':[num],'ratio':[ratio]})
            if direction=='row':
                num=len(group)
                ratio=round(num/row,3)
                return pd.DataFrame({'num':[num],'ratio':[ratio]})
            
        if isinstance(value,int):
            
            statistice_value=factor_data[factor_data==value]
            if len(statistice_value) ==0:
                print('factor_data中没有这个值')
            else:
                value_int_row=statistice_value.groupby('date').apply(ratio_2,direction='column').\
                            reset_index().drop(['level_1'],axis=1).set_index('date')
                            
                value_int_column=statistice_value.groupby('asset').apply(ratio_2,direction='row').\
                                        reset_index().drop(['level_1'],axis=1).set_index('asset')
                                        
                value_int=(len(statistice_value),round(len(statistice_value)/len(factor_data),3))
                
                factor_value_statistice.update({'value_int':value_int,'value_int_row':value_int_row,\
                                                'value_int_column':value_int_column})
        elif value is None:
            pass
        else:
            raise ValueError("The parameter 'value' types is not int ")
            
        if isinstance(interval,tuple):
            left,right=interval
            
            if left < right:
                statistice_interval=factor_data[(left<=factor_data) & (factor_data<=right)]
                
                interval_tuple=(len(statistice_interval),round(len(statistice_interval)/len(factor_data),3))
                
                interval_tuple_row=statistice_interval.groupby('date').apply(ratio_2,direction='column').\
                        reset_index().drop(['level_1'],axis=1).set_index('date')
                
                interval_tuple_column=statistice_interval.groupby('asset').apply(ratio_2,direction='row').\
                                    reset_index().drop(['level_1'],axis=1).set_index('asset')
                                    
                factor_value_statistice.update({'interval_tuple':interval_tuple,'interval_tuple_row':interval_tuple_row,\
                                            'interval_tuple_column':interval_tuple_column})                    
            else:
                raise ValueError("参数 interval 左端值应小于右端值")
        elif interval is None:
            pass
        
        else:
            raise ValueError("The parameter 'interval' types is not tuple")
    
        return factor_value_statistice

    def auth(self, username='17854120489', password='shafajueduan28'):
        import jqdatasdk
        jqdatasdk.auth(username, password)
        self.api = jqdatasdk
       
    def factor_value_industry(self,factor_data=None,industry='sw_l1'):
        '''
        因子值得行业分布情况
        
        参数
        ----------
        industry : str ,行业分类标准
        
        返回值
        ----------
        industry_situation: dict,
        '''        
        self.auth()
        
        if factor_data is None:     
            factor_data=self.factor.copy()  
            factor_data.index = pd.to_datetime(factor_data.index)
            factor_data.index.name='date'
            factor_data.columns.name='asset'
            row,column=factor_data.shape
            factor_data = factor_data.stack('asset',dropna=False) 
            factor_data.name='value'
        elif isinstance(factor_data,pd.Series):
            row,column=factor_data.unstack().shape
            factor_data.name='value'
            factor_data.index.names=['date','asset']
        
        start_date=factor_data.index.get_level_values('date')[0]
        end_date=factor_data.index.get_level_values('date')[-1]
        trade_days=list(self.api.get_trade_days(start_date=start_date,end_date=end_date))
        
        securities=self.factor.columns.tolist()
        
        industries = map(partial(self.api.get_industry, securities), trade_days)
        
        industries = {
            d: {
                s: ind.get(s).get(industry, dict()).get('industry_name', 'NA')
                for s in securities
            }
            for d, ind in zip(trade_days, industries)   
        }
        
        merged_data=pd.DataFrame(factor_data)
        merged_data['group']=pd.DataFrame(industries).T.sort_index().stack().values
        
        industry_factor_value=merged_data.groupby('group')['value'].describe()
        industry_factor_value['ratio']=(industry_factor_value['count']/len(factor_data)).round(3)
        
        industry_df = pd.DataFrame(industries).T
        
        grouper = [merged_data.index.get_level_values('date')]
        grouper.append('group')
        industry_num = merged_data.groupby(grouper).count().unstack()
        
        industry_situation = dict()
        industry_situation.update({'industry_factor_value':industry_factor_value,\
                           'industry_df':industry_df,'industry_num':industry_num})
        return industry_situation
              
    def overall_factor_value(self,factor_data=None):
        '''
        因子值得整体特征
        
        返回值
        -----------
        overall_statistice ： DataFrame ,因子值得统计特征
        '''
        
        if factor_data is None:     
            factor_data=self.factor.copy()            
            factor_data.index.name='date'
            factor_data.columns.name='asset'
            row,column=factor_data.shape
            factor_data = factor_data.stack('asset',dropna=False) 
            factor_data.name='value'
            factor_data[np.isinf(factor_data)]=np.nan
        elif isinstance(factor_data,pd.Series):
            row,column=factor_data.unstack().shape
            factor_data.name='value'
            factor_data.index.names=['date','asset']        
            factor_data[np.isinf(factor_data)]=np.nan
            
        overall_statistice=pd.DataFrame(factor_data).describe().T
        overall_statistice["p-value"] = stats.normaltest(factor_data.dropna())[1]
        overall_statistice["Skew"] = stats.skew(factor_data.dropna())
        overall_statistice["Kurtosis"] = stats.kurtosis(factor_data.dropna())
        
        return overall_statistice.T
        
    def plot_hist(self,factor_data=None,bins=30): 
        
        if factor_data is None:     
            factor_data=self.factor.copy()            
            factor_data.index.name='date'
            factor_data.columns.name='asset'
            row,column=factor_data.shape
            factor_data = factor_data.stack('asset',dropna=False) 
            factor_data.name='value'
            factor_data[np.isinf(factor_data)]=np.nan
            
        elif isinstance(factor_data,pd.Series):
            row,column=factor_data.unstack().shape
            factor_data.name='value'
            factor_data.index.names=['date','asset']
            factor_data[np.isinf(factor_data)]=np.nan
            
        fig,axes=plt.subplots(figsize=(9, 4))
        axes.hist(factor_data.dropna().values,bins=bins)
        axes.grid(True)
        axes.set_xlabel('number')
        axes.set_ylabel('factor value')
         
    def plot_scatter(self,factor_data=None):
        
        if factor_data is None:     
            factor_data=self.factor.copy()            
            factor_data.index.name='date'
            factor_data.columns.name='asset'
            row,column=factor_data.shape
            factor_data = factor_data.stack('asset',dropna=False) 
            factor_data.name='value'
        elif isinstance(factor_data,pd.Series):
            row,column=factor_data.unstack().shape
            factor_data.name='value'
            factor_data.index.names=['date','asset']
            
        fig,axes=plt.subplots(figsize=(9, 4))
        x=np.arange(1,len(factor_data.dropna())+1)
        y=factor_data.dropna().values
        axes.scatter(x=x,y=y)
        axes.grid(True)
        axes.set_xlabel('number')
        axes.set_ylabel('factor value')      


# =============================================================================
# import jqdatasdk
# import pandas as pd
# jqdatasdk.auth('17854120489', 'shafajueduan28')
# factor=VOL5
# detectedactorvalues=DetecteFactorValues(factor)
# null_value_situation=detectedactorvalues.detecte_null_value()
# detectedactorvalues.plot_hist()
# detectedactorvalues.plot_scatter()
# 
# factor_value_statistice=detectedactorvalues.statistice_factor_value(quantile=4,value=0,interval=(0,0.5))
# 
# industry_factor_value=detectedactorvalues.factor_value_industry()
# 
# overall_statistice=detectedactorvalues.overall_factor_value()
# =============================================================================

class DisposeFactorValue(object):
    
    def __init__(self):
        pass
    
    def overall_factor_value(self,factor_data,factor_data_change):
        '''
        因子值得整体特征
        
        返回值
        -----------
        overall_statistice ： DataFrame ,因子值得统计特征
        '''
        
        if  (isinstance(factor_data,pd.Series) and isinstance(factor_data_change,pd.Series)):              
            overall_statistice01=pd.DataFrame(factor_data).describe().T
            overall_statistice01["p-value"] = stats.normaltest(factor_data.dropna())[1]
            overall_statistice01["Skew"] = stats.skew(factor_data.dropna())
            overall_statistice01["Kurtosis"] = stats.kurtosis(factor_data.dropna())
            
            overall_statistice02=pd.DataFrame(factor_data_change).describe().T
            overall_statistice02["p-value"] = stats.normaltest(factor_data_change.dropna())[1]
            overall_statistice02["Skew"] = stats.skew(factor_data_change.dropna())
            overall_statistice02["Kurtosis"] = stats.kurtosis(factor_data_change.dropna())
            
            overall_statistice02.rename(index={'value':'value_change'},inplace=True)
            overall_statistice=pd.concat([overall_statistice01,overall_statistice02])
        else:
            raise ValueError("Parameter type error")
            
        return overall_statistice.T
    
    def plot_hist(self,factor_data,factor_data_change,bins=50): 
        
        if  (isinstance(factor_data,pd.Series) and isinstance(factor_data_change,pd.Series)):            
            fig,axes=plt.subplots(2,1,figsize=(15,7))
            axes[0].hist(factor_data.dropna().values,bins=bins)
            axes[0].grid(True)
            axes[0].set_xlabel('number')
            axes[0].set_ylabel('factor value')      
        
            axes[1].hist(factor_data_change.dropna().values,bins=bins)
            axes[1].grid(True)
            axes[1].set_xlabel('number')
            axes[1].set_ylabel('factor value')            
        else:
            raise ValueError("Parameter type error")
                     
    def plot_scatter(self,factor_data,factor_data_change):
        
        if  (isinstance(factor_data,pd.Series) and isinstance(factor_data_change,pd.Series)):            
            fig,axes=plt.subplots(2,1,figsize=(15,7))
            x=np.arange(1,len(factor_data.dropna())+1)
            y=factor_data.dropna().values
            axes[0].scatter(x=x,y=y)
            axes[0].grid(True)
            axes[0].set_xlabel('number')
            axes[0].set_ylabel('factor value')      
            
            x=np.arange(1,len(factor_data_change.dropna())+1)
            y=factor_data_change.dropna().values
            axes[1].scatter(x=x,y=y)
            axes[1].grid(True)
            axes[1].set_xlabel('number')
            axes[1].set_ylabel('factor value')
            
        else:
            raise ValueError("Parameter type error")
            
'''
factor=VOL5
w= factor.stack(dropna=False)
w.name='value'
disposefactorvalue=DisposeFactorValue()
a=disposefactorvalue.overall_factor_value(w,w)
disposefactorvalue.plot_hist(w,w)
disposefactorvalue.plot_scatter(w,w)
'''

def winsorize_med(data, scale=1, inclusive=True, inf2nan=True, axis=1):
    '''
    参数
    ------------
    data: pd.Series/pd.DataFrame, 待缩尾的序列
    scale: 倍数，默认为 1.0。会将位于 [med - scale * distance, med + scale * distance] 边界之外的值替换为边界值/np.nan
    inclusive bool 是否将位于边界之外的值替换为边界值，默认为 True。 如果为 True，则将边界之外的值替换为边界值，否则则替换为 np.nan
    inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan，默认为 True。如果为 True，在缩尾之前会先将 np.inf 和 -np.inf 替换成 np.nan，缩尾的时候不会考虑 np.nan，否则 inf 被认为是在上界之上，-inf 被认为在下界之下
    axis: 在 data 为 pd.DataFrame 时使用，沿哪个方向做标准化，默认为 1。0 为对每列做缩尾，1 为对每行做缩尾

    返回
    ------------
    中位数去极值之后的因子数据
    '''
    if isinstance(data,pd.DataFrame):
        value = data.copy()        
        if axis==1:
            long = value.shape[0]
            for i in range(long):
                s = value.iloc[i,:]
                if inf2nan==True:
                    s[np.isinf(s)]=np.nan
                    med = np.median(s.dropna())
                    distance = np.median(np.abs(s-med).dropna())
                    up = med+scale*distance
                    down = med-scale*distance            
                    if inclusive==True:
                        s[s>up]=up
                        s[s<down]=down
                    else:
                        s[s>up]=np.nan
                        s[s<down]=np.nan            
                else:
                    med = np.median(s.dropna())
                    distance = np.median(np.abs(s-med).dropna())
                    up = med+scale*distance
                    down = med-scale*distance
                    if inclusive==True:
                        s[s>up]=up
                        s[s<down]=down
                    else:
                        s[s>up]=np.nan
                        s[s<down]=np.nan
            return value
        elif axis==0:
            width = value.shape[1]
            for j in range(width):
                s = value.iloc[:,j]
                if inf2nan==True:
                    s[np.isinf(s)]=np.nan
                    med = np.median(s.dropna())
                    distance = np.median(np.abs(s-med).dropna())
                    up = med+scale*distance
                    down = med-scale*distance            
                    if inclusive==True:
                        s[s>up]=up
                        s[s<down]=down
                    else:
                        s[s>up]=np.nan
                        s[s<down]=np.nan                
                else:
                    med = np.median(s.dropna())
                    distance = np.median(np.abs(s-med).dropna())
                    up = med+scale*distance
                    down = med-scale*distance
                    if inclusive==True:
                        s[s>up]=up
                        s[s<down]=down
                    else:
                        s[s>up]=np.nan
                        s[s<down]=np.nan
            return value
        else:
            return('axis值有误')
    elif isinstance(data,pd.Series):
        value = data.copy()
        if inf2nan==True:
            value[np.isinf(value)]=np.nan
            med = np.median(value.dropna())
            distance = np.median(np.abs(value-med).dropna())
            up = med+scale*distance
            down = med-scale*distance            
            if inclusive==True:
                value[value>up]=up
                value[value<down]=down
            else:
                value[value>up]=np.nan
                value[value<down]=np.nan  
            return value
        else:
            med = np.median(value.dropna())
            distance = np.median(np.abs(value-med).dropna())
            up = med+scale*distance
            down = med-scale*distance
            if inclusive==True:
                value[value>up]=up
                value[value<down]=down
            else:
                value[value>up]=np.nan
                value[value<down]=np.nan  
            return value
    else:
        print('不是pd.Series和pd.DataFrame类型')
        return

def standardlize(data, inf2nan=True, axis=1):
    '''
    参数
    -----------
    data: pd.Series/pd.DataFrame/np.array, 待标准化的序列
    inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan。默认为 True
    axis=1: 在 data 为 pd.DataFrame 时使用，如果 series 为 pd.DataFrame，沿哪个方向做标准化。0 为对每列做标准化，1 为对每行做标准化
    返回
    -----------
    标准化后的因子数据
    '''
    if isinstance(data,pd.DataFrame):
        value = data.copy()
        if axis==1:
            long = value.shape[0]
            for i in range(long):
                s = value.iloc[i,:]
                if inf2nan==True:
                    s[np.isinf(s)]=np.nan
                    mean = np.mean(s.dropna())
                    std = np.std(s.dropna(),ddof=1)
                    value.iloc[i,:] = (s-mean)/std            
                else: 
                    s1 = s[~np.isinf(s)]
                    mean = np.mean(s1)
                    std = np.std(s1,ddof=1)
                    value.iloc[i,:] = (s-mean)/std
            return value
        elif axis==0:
            width = value.shape[1]
            for j in range(width):
                s = value.iloc[:,j]
                if inf2nan==True:
                    s[np.isinf(s)]=np.nan
                    mean = np.mean(s.dropna())
                    std = np.std(s.dropna(),ddof=1)
                    value.iloc[:,j] = (s-mean)/std            
                else: 
                    s1 = s[~np.isinf(s)]
                    mean = np.mean(s1)
                    std = np.std(s1,ddof=1)
                    value.iloc[:,j] = (s-mean)/std
            return value
        else:
            return('axis值有误')
            
    elif isinstance(data,pd.Series):
        value = data.copy()
        if inf2nan==True:
            value[np.isinf(value)]=np.nan
            mean = np.mean(value.dropna())
            std = np.std(value.dropna(),ddof=1)
            value = (value-mean)/std
            return value
        else: 
            s = value[~np.isinf(value)]
            mean = np.mean(s)
            std = np.std(s,ddof=1)
            value = (value-mean)/std
            return value
    else:
        print('data不是pd.Series和pd.DataFrame类型')
        return

    