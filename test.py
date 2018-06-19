# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 00:29:34 2018

@author: Anshit
"""

import sys
import pandas as pd
import numpy as np
import pickle



input_file = 'test.csv' 
gt_file = 'gt.csv'
filename = 'finalmodel1.pkl'
filename_2 = 'finalmodel2.pkl'

df=pd.read_csv('train.csv')
df=df.replace(np.nan, 0 ,regex=True)
df=df.replace(['^SoldFast$'], 3,regex=True)
df=df.replace(['^SoldSlow$'], 2,regex=True)
df=df.replace(['^NotSold$'], 1,regex=True)
Y =  df['SaleStatus']
df.drop(['SaleStatus'],axis=1,inplace=True)  

df2 = pd.read_csv(input_file)
df2=df2.replace(np.nan, 0 ,regex=True)

train_objs_num = len(df)
dataset = pd.concat(objs=[df, df2], axis=0)
dataset = pd.get_dummies(dataset)
train = dataset[:train_objs_num]
test = dataset[train_objs_num:]

train= train.apply(pd.to_numeric)
test= test.apply(pd.to_numeric)

# load the model from disk

loaded_model = pickle.load(open(filename, 'rb'))
y_pred_3 = loaded_model .predict(test)  

x=pd.DataFrame()
arr=df2.Id.values
x['ID']=arr
x['SaleStatus']= y_pred_3
x['SaleStatus'] = x['SaleStatus'].map({ 1 :'NotSold', 2:'SoldSlow' , 3:'SoldFast' })
x.to_csv("out1.csv",sep=',',encoding='utf-8',index=False)

df3= pd.read_csv(gt_file)
count=0

for i in range(len(df3['ID'])) :
    if df3['SaleStatus'][i] == x['SaleStatus'][i] :
        count+=1
        
print("Accuracy Model 1:",count*100/len(df3['ID']) ,"%")
        


loaded_model = pickle.load(open(filename_2, 'rb'))
y_pred_3 = loaded_model .predict(test)  
x=pd.DataFrame()
arr=df2.Id.values
x['ID']=arr
x['SaleStatus']= y_pred_3
x['SaleStatus'] = x['SaleStatus'].map({ 1 :'NotSold', 2:'SoldSlow' , 3:'SoldFast' })
x.to_csv("out2.csv",sep=',',encoding='utf-8',index=False)

count=0
for i in range(len(df3['ID'])) :
    if df3['SaleStatus'][i] == x['SaleStatus'][i] :
        count+=1
        
print("Accuracy Model 2:",count*100/len(df3['ID']) ,"%")
        


