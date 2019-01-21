#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#导入相关的数据包
import numpy as np
import pandas as pd
import seaborn as sns

root_path= 'D:/sales/shop/shop/'
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
com_test.to_csv("com_test.csv")
#找出最后一个月的数据集作为预测
test_data=pd.read_csv('%s%s'%(root_path,'test.csv'))
shopid_itemid1=shopid_itemid.ix[shopid_itemid["date_block_num"]==33,]
test_data1=pd.merge(test_data,shopid_itemid1,left_on=["shop_id","item_id"],right_on=["shop_id","item_id"],how='outer')
test_data1.to_csv("test_data1.csv")
test_data2=pd.merge(test_data1,shopid,on="shop_id",how='outer')
test_data3=pd.merge(test_data2,itemid,on="item_id",how='outer')
test_data3.to_csv("test_data.csv")
test_data3.columns=["ID","shop_id","item_id","date_block_num","item_cnt_month","item_id_counts","item_category_id"]
test_data_X=test_data3[["ID","date_block_num","shop_id","item_id","item_cnt_month","item_id_counts","item_category_id"]]
test_data_X=test_data_X.fillna(0)

#训练模型
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression(normalize=True,fit_intercept=True,copy_X =True)
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

#测试集预测
test_pred=lr.predict(test_data_X[["date_block_num","shop_id","item_id","item_cnt_month","item_id_counts","item_category_id"]])
test_pred=pd.DataFrame(test_pred)
test_pred.to_csv("test_pred.csv")








