# -*- coding:utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

# action_train = pd.read_csv("trainingset/action_train.csv", index_col=None)
# action_test = pd.read_csv("test/action_test.csv", index_col=None)
action_train = pd.read_csv("feature_process/action_process_train.csv", index_col=None)
action_test = pd.read_csv("feature_process/action_process_test.csv", index_col=None)
orderFuture_train = pd.read_csv("trainingset/orderFuture_train.csv", index_col=None)
orderFuture_test = pd.read_csv("test/orderFuture_test.csv", index_col=None)
history_train = pd.read_csv('trainingset/orderHistory_train.csv')
history_test = pd.read_csv('test/orderHistory_test.csv')
feature_score = pd.read_csv('feature_score/feature_score.csv')

userid_train = orderFuture_train['userid']
userid_test = orderFuture_test['userid']
label_train = orderFuture_train['orderType']
feature_train = pd.DataFrame({
    'userid': userid_train,
    'label': label_train
}, columns = ['userid', 'label'])
feature_test = pd.DataFrame({'userid': userid_test})
print 'read_csv okay'

#-------------------type-------------------------
def userType():
    type_last1_train = []
    type_first1_train = []
    type_last1_test = []
    type_first1_test = []
    for i in userid_train:
        type = action_train[action_train['userid']==i]['actionType'].tolist()
        type_first1_train.append(type[0])
        type_last1_train.append(type[-1])
    for i in userid_test:
        type = action_test[action_test['userid']==i]['actionType'].tolist()
        type_first1_test.append(type[0])
        type_last1_test.append(type[-1])

    return type_first1_train, type_last1_train,type_first1_test, type_last1_test

type_first1_train, type_last1_train, type_first1_test, type_last1_test = userType()
feature_train['type_first1'] = type_first1_train
feature_train['type_last1'] = type_last1_train
feature_test['type_first1'] = type_first1_test
feature_test['type_last1'] = type_last1_test
print 'type okay'


#-----------------actionTime/Type------------------------
actionTimetrain = {}
for i in userid_train:
    actionTime_train = action_train[action_train['userid'] == i]['actionTime'].tolist()
    actionTimetrain[i] = actionTime_train

actionTimetest = {}
for i in userid_test:
    actionTime_test = action_test[action_test['userid'] == i]['actionTime'].tolist()
    actionTimetest[i] = actionTime_test
print 'actionTime okay'

actionTypetrain = {}
for i in userid_train:
    actionType_train = action_train[action_train['userid'] == i]['actionType'].tolist()
    actionTypetrain[i] = actionType_train

actionTypetest = {}
for i in userid_test:
    actionType_test = action_test[action_test['userid'] == i]['actionType'].tolist()
    actionTypetest[i] = actionType_test
print 'actionType okay'

actionYeartest = {}
for i in userid_test:
    actionYear_test = action_test[action_test['userid'] == i]['year'].tolist()
    actionYeartest[i] = actionYear_test

actionYeartrain = {}
for i in userid_train:
    actionYear_train = action_train[action_train['userid'] == i]['year'].tolist()
    actionYeartrain[i] = actionYear_train
print 'actionYear okay'

actionMonthtest = {}
for i in userid_test:
    actionMonth_test = action_test[action_test['userid'] == i]['month'].tolist()
    actionMonthtest[i] = actionMonth_test

actionMonthtrain = {}
for i in userid_train:
    actionMonth_train = action_train[action_train['userid'] == i]['month'].tolist()
    actionMonthtrain[i] = actionMonth_train
print 'actionMonth okay'

# ----------------time-------------------------------
def action_time(userid, actionTime):
    timeFirst = []
    timeLast = []
    timeInterval_FL = []
    for i in userid:
        timeFirst.append(actionTime[i][0])
        timeLast.append(actionTime[i][-1])
        timeInterval_FL.append(actionTime[i][-1]-actionTime[i][0])
    return timeFirst, timeLast, timeInterval_FL

def year_month(userid, actionMonth, actionYear):
    monthFirst = []
    monthLast = []
    yearFirst = []
    yearLast = []
    for i in userid:
        monthFirst.append(actionMonth[i][0])
        monthLast.append(actionMonth[i][-1])
        yearFirst.append(actionYear[i][0])
        yearLast.append(actionYear[i][-1])
    return monthFirst, monthLast, yearFirst, yearLast
monthFirsttrain, monthLasttrain, yearFirsttrain, yearLasttrain = year_month(userid_train, actionMonthtrain, actionYeartrain)
monthFirsttest, monthLasttest, yearFirsttest, yearLasttest = year_month(userid_test, actionMonthtest, actionYeartest)
feature_train['monthFirst'] = monthFirsttrain
feature_train['monthLast'] = monthLasttrain
feature_train['yearFirst'] = yearFirsttrain
feature_train['yearLast'] = yearLasttrain
feature_test['monthFirst'] = monthFirsttest
feature_test['monthLast'] = monthLasttest
feature_test['yearFirst'] = yearFirsttest
feature_test['yearLast'] = yearLasttest
print 'year_month okay~'


timeFirst_train, timeLast_train, timeInterval_FL_train = action_time(userid_train, actionTimetrain)
timeFirst_test, timeLast_test, timeInterval_FL_test = action_time(userid_test, actionTimetest)
feature_train['timeFirst'] = timeFirst_train
feature_train['timeLast'] = timeLast_train
feature_train['timeInterval_FL'] = timeInterval_FL_train
feature_test['timeFirst'] = timeFirst_test
feature_test['timeLast'] = timeLast_test
feature_test['timeInterval_FL'] = timeInterval_FL_test

print 'action_time okay'

#-----------------timeInterval------------------------
def timeInterval():
    timeIntervaltrain = {}
    timeIntervaltrain_min = []
    timeIntervaltrain_mean = []
    timeIntervaltrain_first = []
    timeIntervaltrain_last = []
    timeIntervaltrain_var = []
    for i in userid_train:
        temp = 0
        timeIntervaltemp = []
        for index,j in enumerate(actionTimetrain[i]):
            timeIntervaltemp.append(j-temp)
            temp = j
        if len(timeIntervaltemp)==0:
            timeIntervaltemp.append(0)
        timeIntervaltrain_min.append(min(timeIntervaltemp))
        timeIntervaltrain_mean.append(sum(timeIntervaltemp)/len(timeIntervaltemp))
        timeIntervaltrain_first.append(timeIntervaltemp[0])
        timeIntervaltrain_last.append(timeIntervaltemp[-1])
        timeIntervaltrain_var.append(np.var(timeIntervaltemp))
        timeIntervaltrain[i] = timeIntervaltemp
    #print timeIntervaltrain[100000000013]

    timeIntervaltest = {}
    timeIntervaltest_min = []
    timeIntervaltest_mean = []
    timeIntervaltest_first = []
    timeIntervaltest_last = []
    timeIntervaltest_var = []
    for i in userid_test:
        temp = 0
        timeIntervaltemp = []
        for index,j in enumerate(actionTimetest[i]):
            timeIntervaltemp.append(j - temp)
            temp = j
        if len(timeIntervaltemp)==0:
            timeIntervaltemp.append(0)
        timeIntervaltest_min.append(min(timeIntervaltemp))
        timeIntervaltest_mean.append(sum(timeIntervaltemp) / len(timeIntervaltemp))
        timeIntervaltest_first.append(timeIntervaltemp[0])
        timeIntervaltest_last.append(timeIntervaltemp[-1])
        timeIntervaltest_var.append(np.var(timeIntervaltemp))
        timeIntervaltest[i] = timeIntervaltemp
    return timeIntervaltrain, timeIntervaltest, timeIntervaltrain_min, timeIntervaltrain_mean, timeIntervaltrain_first, \
           timeIntervaltrain_last, timeIntervaltrain_var, timeIntervaltest_min, timeIntervaltest_mean, timeIntervaltest_first, \
           timeIntervaltest_last, timeIntervaltest_var
timeIntervaltrain, timeIntervaltest, timeIntervaltrain_min, timeIntervaltrain_mean, timeIntervaltrain_first, \
timeIntervaltrain_last, timeIntervaltrain_var, timeIntervaltest_min, timeIntervaltest_mean, timeIntervaltest_first, \
timeIntervaltest_last, timeIntervaltest_var = timeInterval()

feature_train['timeInterval_min'] = timeIntervaltrain_min
feature_train['timeInterval_mean'] = timeIntervaltrain_mean
feature_train['timeInterval_first'] = timeIntervaltrain_first
feature_train['timeInterval_last'] = timeIntervaltrain_last
feature_train['timeInterval_var'] = timeIntervaltrain_var
feature_test['timeInterval_min'] = timeIntervaltest_min
feature_test['timeInterval_mean'] = timeIntervaltest_mean
feature_test['timeInterval_first'] = timeIntervaltest_first
feature_test['timeInterval_last'] = timeIntervaltest_last
feature_test['timeInterval_var'] = timeIntervaltest_var
print 'timeInterval okay'


# ----------------timeInterval------------------------------------
# timeIntervaltrain, timeIntervaltest, actionTimetrain = {}
def buyNum(timeInterval, train_test, userid):
    buyTime = {}
    buyCount = []
    buyInterval = 21600
    for i in userid:
        buyCount_temp = 0
        buyTime_temp = []
        for index, j in enumerate(timeInterval[i]):
            if j > buyInterval:
                buyTime_temp.append(index)
                buyCount_temp += 1
        buyCount.append(buyCount_temp)
        buyTime[i] = buyTime_temp
    return buyTime, buyCount

buytime_train, buyCount_train = buyNum(timeIntervaltrain, feature_train, userid_train)
print 'train buyNum okay!'
buytime_test, buyCount_test = buyNum(timeIntervaltest, feature_test, userid_test)
print 'train buyNum okay!'

def finalActionrate(buytime, buyCount, userid, feature, actionTime, actionType):
    finalAction_len = []
    finalAction_time = []
    finalAction_1 = []
    finalAction_2 = []
    finalAction_3 = []
    finalAction_4 = []
    finalAction_5 = []
    finalAction_6 = []
    finalAction_7 = []
    finalAction_8 = []
    finalAction_9 = []
    finalAction_rate_1 = []
    finalAction_rate_2 = []
    finalAction_rate_3 = []
    finalAction_rate_4 = []
    finalAction_rate_5 = []
    finalAction_rate_6 = []
    finalAction_rate_7 = []
    finalAction_rate_8 = []
    finalAction_rate_9 = []

    for i in userid:
        finalAction = actionType[i][buytime[i][-1]:]
        finalAction_len_temp = len(finalAction)
        finalAction_time_temp = actionTime[i][len(actionType[i])-1] - actionTime[i][buytime[i][-1]]
        finalAction_1.append(finalAction.count(1))
        finalAction_2.append(finalAction.count(2))
        finalAction_3.append(finalAction.count(3))
        finalAction_4.append(finalAction.count(4))
        finalAction_5.append(finalAction.count(5))
        finalAction_6.append(finalAction.count(6))
        finalAction_7.append(finalAction.count(7))
        finalAction_8.append(finalAction.count(8))
        finalAction_9.append(finalAction.count(9))
        finalAction_rate_1.append(finalAction.count(1) / finalAction_len_temp)
        finalAction_rate_2.append(finalAction.count(2) / finalAction_len_temp)
        finalAction_rate_3.append(finalAction.count(3) / finalAction_len_temp)
        finalAction_rate_4.append(finalAction.count(4) / finalAction_len_temp)
        finalAction_rate_5.append(finalAction.count(5) / finalAction_len_temp)
        finalAction_rate_6.append(finalAction.count(6) / finalAction_len_temp)
        finalAction_rate_7.append(finalAction.count(7) / finalAction_len_temp)
        finalAction_rate_8.append(finalAction.count(8) / finalAction_len_temp)
        finalAction_rate_9.append(finalAction.count(9) / finalAction_len_temp)
        finalAction_len.append(finalAction_len_temp)
        finalAction_time.append(finalAction_time_temp)

    feature['buyCount'] = buyCount
    feature['finalAction_len'] = finalAction_len
    feature['finalAction_time'] = finalAction_time
    feature['finalAction_1'] = finalAction_1
    feature['finalAction_2'] = finalAction_2
    feature['finalAction_3'] = finalAction_3
    feature['finalAction_4'] = finalAction_4
    feature['finalAction_5'] = finalAction_5
    feature['finalAction_6'] = finalAction_6
    feature['finalAction_7'] = finalAction_7
    feature['finalAction_8'] = finalAction_8
    feature['finalAction_9'] = finalAction_9
    feature['finalAction_rate_1'] = finalAction_rate_1
    feature['finalAction_rate_2'] = finalAction_rate_2
    feature['finalAction_rate_3'] = finalAction_rate_3
    feature['finalAction_rate_4'] = finalAction_rate_4
    feature['finalAction_rate_5'] = finalAction_rate_5
    feature['finalAction_rate_6'] = finalAction_rate_6
    feature['finalAction_rate_7'] = finalAction_rate_7
    feature['finalAction_rate_8'] = finalAction_rate_8
    feature['finalAction_rate_9'] = finalAction_rate_9
    return feature
print 'finalAction okay~'

feature_train = finalActionrate(buytime_train, buyCount_train, userid_train, feature_train, actionTimetrain, actionTypetrain)
feature_test = finalActionrate(buytime_test, buyCount_test, userid_test, feature_test, actionTimetest, actionTypetest)


#--------------------------actionRate--------------------
def action_rate(userid, action):
    action_Type = []
    for i in userid:
        action_Type.append(action[(action.userid == i)]['actionType'].tolist())
    action1_rate = []
    action2_rate = []
    action3_rate = []
    action4_rate = []
    action5_rate = []
    action6_rate = []
    action7_rate = []
    action8_rate = []
    action9_rate = []
    for i in action_Type:
        len_action = len(i)
        action1_rate.append(i.count(1) / len_action)
        action2_rate.append(i.count(2) / len_action)
        action3_rate.append(i.count(3) / len_action)
        action4_rate.append(i.count(4) / len_action)
        action5_rate.append(i.count(5) / len_action)
        action6_rate.append(i.count(6) / len_action)
        action7_rate.append(i.count(7) / len_action)
        action8_rate.append(i.count(8) / len_action)
        action9_rate.append(i.count(9) / len_action)
    return action1_rate, action2_rate, action3_rate, action4_rate, action5_rate, action6_rate, action7_rate, action8_rate, action9_rate

action1_rate_train, action2_rate_train, action3_rate_train, action4_rate_train, action5_rate_train, action6_rate_train, action7_rate_train, action8_rate_train, action9_rate_train \
    =action_rate(userid_train, action_train)
action1_rate_test, action2_rate_test, action3_rate_test, action4_rate_test, action5_rate_test, action6_rate_test, action7_rate_test, action8_rate_test, action9_rate_test \
    =action_rate(userid_test, action_test)
feature_train['action1_rate'] = action1_rate_train
feature_train['action2_rate'] = action2_rate_train
feature_train['action3_rate'] = action3_rate_train
feature_train['action4_rate'] = action4_rate_train
feature_train['action5_rate'] = action5_rate_train
feature_train['action6_rate'] = action6_rate_train
feature_train['action7_rate'] = action7_rate_train
feature_train['action8_rate'] = action8_rate_train
feature_train['action9_rate'] = action9_rate_train
feature_test['action1_rate'] = action1_rate_test
feature_test['action2_rate'] = action2_rate_test
feature_test['action3_rate'] = action3_rate_test
feature_test['action4_rate'] = action4_rate_test
feature_test['action5_rate'] = action5_rate_test
feature_test['action6_rate'] = action6_rate_test
feature_test['action7_rate'] = action7_rate_test
feature_test['action8_rate'] = action8_rate_test
feature_test['action9_rate'] = action9_rate_test
print 'action rate okay'

#-------------------------history----------------------
def isHistory(train_test, history):
    isHistory = []
    isOrder = []
    for i in train_test:
        if i in history['userid'].tolist():
            isHistory.append(1)
            if history[history['userid']==i][history['orderType'].isin(['1'])].empty == True:
                isOrder.append(1)
            else:
                isOrder.append(0)
        else:
            isHistory.append(0)
            isOrder.append(2)
    return isHistory, isOrder
isHistory_train, isOrder_train = isHistory(userid_train, history_train)
isHistory_test, isOrder_test = isHistory(userid_test, history_test)
feature_train['isHistory'] = isHistory_train
feature_train['isOrder'] = isOrder_train
feature_test['isHistory'] = isHistory_test
feature_test['isOrder'] = isOrder_test
print 'history okay'

def getDummies(train_test):
    label = ['isHistory', 'isOrder']
    temp1 = pd.get_dummies(train_test['isHistory'], prefix='isHistory')
    temp2 = pd.get_dummies(train_test['isOrder'], prefix='isOrder')
    train_test = pd.DataFrame(pd.concat([train_test, temp1], axis=1))
    train_test = pd.DataFrame(pd.concat([train_test, temp2], axis=1))
    train_test = pd.DataFrame(train_test.drop(label, axis=1))
    return train_test

feature_train = getDummies(feature_train)
feature_test = getDummies(feature_test)
print 'dummies okay'



# ------------nearest time interval------------------
def findLastindex(list_s, label):  # find last index
    reverse_index = list(reversed(list_s)).index(label)
    lenth = len(list_s)
    return lenth-1-reverse_index

def nearestInterval(userid, actionType,timeInterval,feature):
    interval_label = [2, 3, 4, 5, 6, 7, 8, 9]
    for label in interval_label:
        nearestIntervalvar = []
        nearestIntervalmean = []
        nearestIntervalmin = []
        nearestIntervalmax = []
        for i in userid:
            actionType_temp = actionType[i]
            if label in actionType_temp:
                action_index = findLastindex(actionType_temp, label)
                temp = timeInterval[i][action_index:]
                nearestIntervalvar.append(np.var(temp))
                nearestIntervalmean.append(np.mean(temp))
                nearestIntervalmin.append(np.min(temp))
                nearestIntervalmax.append(np.max(temp))
            else:
                nearestIntervalvar.append(0)
                nearestIntervalmean.append(0)
                nearestIntervalmin.append(0)
                nearestIntervalmax.append(0)
        name_var = 'nearestInterval_var'+str(label)
        name_mean = 'nearestInterval_mean' + str(label)
        name_min = 'nearestInterval_min' + str(label)
        name_max = 'nearestInterval_max' + str(label)
        feature[name_var] = nearestIntervalvar
        feature[name_mean] = nearestIntervalmean
        feature[name_min] = nearestIntervalmin
        feature[name_max] = nearestIntervalmax
    return feature
feature_train = nearestInterval(userid_train, actionTypetrain,timeIntervaltrain,feature_train)
feature_test = nearestInterval(userid_test, actionTypetest,timeIntervaltest,feature_test)
print 'nearestimeInterval okay'


# ----------------nearestTime----------------------------
def nearestTime(userid, actionType, actionTime,feature):
    time_label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for label in time_label:
        nearestTime = []
        for i in userid:
            actionType_temp = actionType[i]
            actionTime_temp = actionTime[i]
            if label in actionType_temp:
                action_index = findLastindex(actionType_temp, label)
                temp = actionTime_temp[-1]-actionTime_temp[action_index]
                nearestTime.append(temp)
            else:
                nearestTime.append(0)
        name_time = 'nearestTime'+str(label)
        feature[name_time] = nearestTime
    return feature
feature_train = nearestTime(userid_train, actionTypetrain, actionTimetrain, feature_train)
feature_test = nearestTime(userid_test, actionTypetest, actionTimetest, feature_test)
print 'nearesTime okay'

# ---------------nearestDistance-------------------------
def nearestDistance(userid, actionType, feature):
    distance_label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for label in distance_label:
        nearestDistance = []
        for i in userid:
            actionType_temp = actionType[i]
            if label in actionType_temp:
                action_index = findLastindex(actionType_temp, label)
                nearestDistance.append(len(actionType_temp) - action_index - 1)
            else:
                nearestDistance.append(0)
        name_distance = 'nearestDistance'+str(label)
        feature[name_distance] = nearestDistance
    return feature
feature_train = nearestDistance(userid_train, actionTypetrain,  feature_train)
feature_test = nearestDistance(userid_test, actionTypetest,  feature_test)
print 'nearestDistance okay'

# ---------------poly--------------------------------------
# feature_list = feature_score[(feature_score.score > 0.5)].feature.tolist()
# train_combine = feature_train[feature_list]
# test_combine = feature_test[feature_list]
#
# poly = PolynomialFeatures(interaction_only=True)
# train_combine = poly.fit_transform(train_combine)
# df = pd.DataFrame(train_combine.reshape(40307, -1))
# df.interpolate(axis=1).values.reshape(train_combine.shape)
# #df[df.columns] = MinMaxScaler().fit_transform(df[df.columns])
# train_combine = pd.concat([feature_train, df], axis=1)
# feature_train = pd.DataFrame(train_combine)
#
# poly = PolynomialFeatures(interaction_only=True)
# test_combine = poly.fit_transform(test_combine)
# df = pd.DataFrame(test_combine.reshape(10076, -1))
# df.interpolate(axis=1).values.reshape(test_combine.shape)
# #df[df.columns] = MinMaxScaler().fit_transform(df[df.columns])
# test_combine = pd.concat([feature_test, df], axis=1)
# feature_test = pd.DataFrame(test_combine)



feature_train.to_csv('feature/feature_train.csv', index=None)
feature_test.to_csv('feature/feature_test.csv', index=None)
# feature_train.to_csv('feature/feature_train_poly.csv', index=None)
# feature_test.to_csv('feature/feature_test_poly.csv', index=None)