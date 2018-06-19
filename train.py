# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 04:04:19 2018

@author: Anshit
"""
import sys
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score



df=pd.read_csv("train.csv", error_bad_lines=False)
df=df.replace(np.nan, 0 ,regex=True)
df=df.replace(['^SoldFast$'], 3,regex=True)
df=df.replace(['^SoldSlow$'], 2,regex=True)
df=df.replace(['^NotSold$'], 1,regex=True)
Y =  df['SaleStatus']
df.drop(['SaleStatus'],axis=1,inplace=True)  

df2 = pd.read_csv("test.csv")
df2=df2.replace(np.nan, 0 ,regex=True)

train_objs_num = len(df)
dataset = pd.concat(objs=[df, df2], axis=0)
dataset = pd.get_dummies(dataset)
train = dataset[:train_objs_num]
test = dataset[train_objs_num:]

train= train.apply(pd.to_numeric)
test= test.apply(pd.to_numeric)


model=XGBClassifier(
 learning_rate =0.05,
 n_estimators=7,
 max_depth=3,
 min_child_weight=1,
 gamma=5,
 objective= 'multi:softprob',
 nthread=8,
 seed=27)

model.fit(train,Y)

# save the model to disk
filename = 'finalmodel1.pkl'
pickle.dump(model, open(filename, 'wb'))

## grid search dont work on multiclass classification
kfold = StratifiedKFold(n_splits=5, random_state=7)
results = cross_val_score(model, train, Y, cv=kfold)
print("K fold Cross validation Accuracy model 1: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


model_2 = RandomForestClassifier(n_estimators=400, n_jobs=2, oob_score = True, random_state=0,
                                 max_features = "auto")

model_2.fit(train,Y)

# save the model to disk
filename = 'finalmodel2.pkl'
pickle.dump(model_2  , open(filename, 'wb'))

## grid search dont work on multiclass classification
kfold = StratifiedKFold(n_splits=5, random_state=7)
results = cross_val_score(model_2, train, Y, cv=kfold)
print("K fold Cross validation Accuracy model 2: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))













