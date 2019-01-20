
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on Mon Dec 17 14:23:53 2018

@author: lytao
"""

#导入相关的数据包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

root_path= 'D:/taoliyuan/ML/sales/'
train = pd.read_csv('%s%s'%(root_path,'train.csv'))
#train=pd.read_csv('%s%s'%(root_path,'sales_train_v2.csv'))
test=pd.read_csv('%s%s'%(root_path,'test.csv'))
train[['item_id']] = train[['item_id']].astype(object)


#处理shopid侧的特征
train_groupby=train.groupby(by=["shop_id"])
df_item_id_counts = pd.DataFrame(columns = ["shop_id","item_id_counts"]) #创建一个空的dataframe
for name , group in train_groupby:
    item_id_counts=len(np.unique(group[["item_id"]]))
    df_item_id_counts=df_item_id_counts.append(pd.DataFrame({'shop_id':[name],'item_id_counts':[item_id_counts]}),ignore_index=True)
df_item_id_counts.to_csv("item_id_counts.csv",index_label=False,index=False)

#处理itemid侧的特征
item_id = pd.read_csv('%s%s'%(root_path,'items.csv'))
item_category_id=item_id[["item_id","item_category_id"]]
item_category_id.to_csv("item_category_id.csv",index=False)


#处理shopid_itemid共同的特征
df_shopid_itemid_cnt_month = pd.DataFrame(columns = ["date_block_num","shop_id","item_id","item_cnt_month"]) #创建一个空的dataframe
shopid_itemid_cnt_month=train.groupby(by=["date_block_num","shop_id","item_id"])
shopid_itemid_cnt_month=shopid_itemid_cnt_month["item_cnt_day"].sum()
shopid_itemid_cnt_month=pd.DataFrame(shopid_itemid_cnt_month)
shopid_itemid_cnt_month.columns=["item_cnt_month"]

# for name ,group in shopid_itemid_cnt_month:                                                           #计算每个月每个商店每种物品的卖出的个数
#     item_cnt_month=group[["item_cnt_day"]].sum
#     print(item_cnt_month)
#     data_num=name[0]
#     shopid=name[1]
#     itemid=name[2]
#     df_shopid_itemid_cnt_month = df_shopid_itemid_cnt_month.append(pd.DataFrame({'date_block_num': [data_num],
#                                                                                  'shop_id':[shopid],
#                                                                                  'item_id':[itemid],
#                                                                                  'item_cnt_month':[item_cnt_month]}),ignore_index=True)
shopid_itemid_cnt_month.to_csv("shopid_itemid_cnt_month.csv")




# shopid_itemid_cnt_month=pd.DataFrame(shopid_itemid_cnt_month)
# shopid_itemid_cnt_month.columns=["item_cnt_month"]
# shopid_itemid_cnt_month.reset_index(inplace=True)
# shopid_itemid_cnt_month.to_csv("shopid_itemid_cnt_month.csv",header=True,index=False)
# item_id_counts=train.groupby(by=["shop_id"])["item_id"].value_counts() #计算每个商店每种物品的个数
# item_id_counts=pd.DataFrame(item_id_counts)
# item_id_counts.columns=['item_id_counts']
# print(item_id_counts.index[1])
#item_id_counts.to_csv("item_id_counts.csv",header=True)


















