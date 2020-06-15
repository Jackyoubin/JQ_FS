# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:18:40 2019

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
clean_factor_data.to_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\aa.csv')


#%%
bb = pd.read_csv('D:\\spyder_code\\jqfactor_analyzer01\\华泰单因子测试之估值类因子\\aa.csv',\
                ).set_index(['date','asset'])









