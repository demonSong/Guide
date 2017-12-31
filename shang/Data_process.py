# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime

action_train = pd.read_csv("trainingset/action_train.csv", index_col=None)
action_test = pd.read_csv("test/action_test.csv", index_col=None)
order_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
order_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)

# -------------------处理时间----------------------------
def date_prodcess(action):
    date = pd.to_datetime(action["actionTime"], unit='s')# format='%Y/%m/%d %H:%M:%S')
    year, month, day, hour = [], [], [], []
    for i in date:
        year.append(i.strftime('%Y'))
        month.append(i.strftime('%m'))
        day.append(i.strftime('%d'))
        hour.append(i.strftime('%H'))

    action['year'] = pd.DataFrame({'year': year})
    action['month'] = pd.DataFrame({'month': month})
    action['day'] = pd.DataFrame({'day': day})
    action['hour'] = pd.DataFrame({'hour': hour})
    return action

# -------------------处理用户id--------------------------
# 5填写 6提交 7下单 8确认 9支付
def encoder_userid(train):
    le = LabelEncoder()
    train['userid'] = le.fit_transform(train['userid'])
    return train

def get_dummies(train):
    dummy_label = ['year', 'month', 'day', 'hour']
    train = pd.get_dummies(train, prefix=dummy_label)
    return train

def drop_label(df):
    drop_label = ['actionTime','userid']
    return df.drop(drop_label, axis=1)

if __name__ =="__main__":
    action = date_prodcess(action_train)
    train = pd.merge(action, order_train)  # 合并两个表

    train = encoder_userid(train)
    train = get_dummies(train)
    train = drop_label(train)
    train.to_csv('Process/action_process.csv', index=False)