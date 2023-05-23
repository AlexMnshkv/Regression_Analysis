#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import re
from io import BytesIO
import requests
import json
from urllib.parse import urlencode
import gspread
import pingouin as pg
from pingouin import multivariate_normality
import math as math
import scipy as scipy
import scipy.stats as stats
from df2gspread import df2gspread as d2g
from oauth2client.service_account import ServiceAccountCredentials 
# Считывем данные
df=pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-/Stat_less7/cars.csv')


# In[2]:


df


# In[3]:


df.dtypes


# In[4]:


# df.isnull().sum()


# In[5]:


df['CarName'].str.split(' ')


# In[6]:



df['CarName']=df.CarName.apply(lambda x: x.strip().split(' ')[0])


# In[7]:


df['CarName'].nunique()


# In[8]:


df['CarName']


# In[9]:


df.groupby('CarName').agg({'wheelbase':'count'})


# In[10]:


# Чтобы не перегружать модель большим количеством предикторов, оставим только часть из них
df_drop = df[['price', 'CarName', 'fueltype', 'aspiration', 'carbody' , 'drivewheel', 'wheelbase', 'carlength','carwidth', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower']]
df_drop


# In[12]:


# посчитаем корреляцию между price и другими переменными
df_drop.corr()


# In[13]:


# Применим преобразование One-Hot Encoding. Она создаёт фиктивные переменные на основе изначальных категорий, представленные в виде 0 и 1
df_11 = pd.get_dummies(data=df[['CarName','fueltype','aspiration', 'carbody', 'drivewheel']], drop_first = True)
df_11


# In[15]:


# Посчитаем корреляцию для всей таблицы
df_11.corr()


# In[104]:


df_112 = pd.get_dummies(data=df[['CarName','fueltype', 'aspiration','carbody', 'drivewheel', 'wheelbase', 'carlength','carwidth', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower','price']], drop_first = True)
df_112


# In[105]:


df1111=df_112.corr()
df1111


# In[98]:


df1111.shape


# In[20]:


df['CarName']=df.CarName.apply(lambda x: x.lower().replace('maxda', 'mazda').replace('Nissan', 'nissan').replace('porcshce', 'porsche').replace('toyouta', 'toyota').replace('vokswagen', 'volkswagen').replace('vw', 'volkswagen'))
df


# In[21]:


df_112 = pd.get_dummies(data=df[['CarName','fueltype', 'aspiration','carbody', 'drivewheel', 'wheelbase', 'carlength','carwidth', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower','price']], drop_first = True)
df_112


# In[22]:


df1111=df_112.corr()
df1111


# In[12]:


import statsmodels.formula.api as smf 
from scipy import stats 
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = smf.ols(formula = "price ~ C(horsepower)", data = df1111).fit()
anova_lm(model)


# In[24]:


# модель с одним предиктором
import statsmodels.formula.api as smf 

results = smf.ols('price ~ C(horsepower)', data = df).fit()
results.summary()


# In[26]:


# модель с несколькими предикторами
import statsmodels.formula.api as smf 

results = smf.ols('price ~ C(horsepower)+C(CarName)+C(fueltype)+C(aspiration)+C(carbody)+C(drivewheel)+C(wheelbase)+C(carlength)+C(carwidth)+C(curbweight)+C(enginetype)+C(cylindernumber)+C(enginesize)+C(boreratio)', data = df).fit()
results.summary()


# In[27]:


# модель с несколькими предикторами без марки машин
import statsmodels.formula.api as smf 

results = smf.ols('price ~ C(horsepower)+C(fueltype)+C(aspiration)+C(carbody)+C(drivewheel)+C(wheelbase)+C(carlength)+C(carwidth)+C(curbweight)+C(enginetype)+C(cylindernumber)+C(enginesize)+C(boreratio)', data = df).fit()
results.summary()


# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf 

# способ первый
# Y = одномерный массив с ЗП, X - массив со всеми нужными нам НП

X = sm.add_constant(X)  # добавить константу, чтобы был свободный член
model = sm.OLS(Y, X)  # говорим модели, что у нас ЗП, а что НП
results = model.fit()  # строим регрессионную прямую
print(results.summary())  # смотрим результат

# способ второй, потенциально более удобный

results = smf.ols('Y ~ X1 + X2 + ... + Xn', data).fit()
print(results.summary())


# In[ ]:




