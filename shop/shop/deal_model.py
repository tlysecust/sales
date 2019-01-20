#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#导入相关的数据包
import numpy as np
import pandas as pd
import seaborn as sns

root_path= 'D:/taoliyuan/ML/sales/shop/'
shopid_itemid = pd.read_csv('%s%s'%(root_path,'shopid_itemid_cnt_month.csv'))
shopid=pd.read_csv('%s%s'%(root_path,'item_id_counts.csv'))
itemid=pd.read_csv('%s%s'%(root_path,'item_category_id.csv'))


#合并所有的数据
train_test=pd.merge(shopid_itemid,shopid,on="shop_id",how='outer')
train_test=pd.merge(train_test,itemid,on="item_id",how='outer')
train_test.to_csv("train_test.csv")
#找出下个月的销量
train_test["nextmonth"]=train_test["date_block_num"]-1
left=train_test[["date_block_num","shop_id","item_id","item_cnt_month","item_id_counts","item_category_id"]]
right=train_test[["shop_id","item_id","nextmonth","item_cnt_month"]]
com_test=pd.merge(left,right,left_on=["date_block_num","shop_id","item_id"],right_on=["nextmonth","shop_id","item_id"],how='left')
com_test=com_test[com_test['nextmonth'].notnull()]
com_test.columns=["date_block_num","shop_id","item_id","item_cnt_month","item_id_counts","item_category_id","nextmonth","item_cnt_nextmonth"]
com_test=com_test[["date_block_num","shop_id","item_id","item_cnt_month","item_id_counts","item_category_id","item_cnt_nextmonth"]]
com_test.to_csv("com_test1.csv")


#训练模型
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression(normalize=True,fit_intercept=True)
x_cols = [x for x in com_test.columns if x != 'item_cnt_nextmonth']
X = com_test[x_cols]
y=com_test.item_cnt_nextmonth
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试
#trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0
lr.fit(trainX,trainY)
print(lr.fit(trainX,trainY))
pred=lr.predict(testX)
# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(testY, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(testY, pred))




# from sklearn.metrics import confusion_matrix,accuracy_score
# score=accuracy_score(y_pred=pred,y_true=testY)  #计算模型在测试集上的准确性
# print(score)
# confusion=confusion_matrix(y_pred=pred,y_true=testY)
# print(confusion)




# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# #sex_data=pd.read_csv("test.csv",sep=",",header=0,index_col=0)
# x_cols = [x for x in com_test.columns if x != 'item_cnt_nextmonth']
# X = com_test[x_cols]
# y=com_test.item_cnt_nextmonth
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试
# lr = LogisticRegression(class_weight='balanced')
# lr.fit(trainX,trainY)
# print(lr.fit(trainX,trainY))
# pred=lr.predict(testX)
# from sklearn.metrics import confusion_matrix,accuracy_score
# score=accuracy_score(y_pred=pred,y_true=testY)  #计算模型在测试集上的准确性
# print(score)
# confusion=confusion_matrix(y_pred=pred,y_true=testY)
# print(confusion)
#
#

# def get_item_cnt_nextmonth(hang):
#     month =hang["date_block_num"]
#     nextmoth=month+1
#     shop =hang["shop_id"]
#     item =hang["item_id"]
#     return (month,nextmoth,shop,item)
#
# train_tes=train_test.apply(get_item_cnt_nextmonth,axis=1)
# #train_test["item_cnt_nextmonth"]=train_test.ix[(train_test["shop_id"] > 1) & (df3["pvalue"] < 0.1), ['score', 'MEAN','description',"adduct"]]
# print(train_tes)
#df=pd.read_csv("train_test.csv")








