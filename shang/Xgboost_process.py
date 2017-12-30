#Import libraries:
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6

train = pd.read_csv('Process/action_process.csv')
#test = pd.read_csv('process_test.csv')
target = 'orderType'
predictors = [x for x in train.columns if x not in [target]]

# target = 'price_sqft'
# predictors = ['lon', 'lat']


X_train, X_test, y_train, y_test = train_test_split(train[predictors], train[target], test_size=0.1, random_state=0)

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds = 10, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                           early_stopping_rounds=early_stopping_rounds, metrics='rmse')#metrics='mse',
        alg.set_params(n_estimators=cvresult.shape[0])
        print cvresult

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])#, eval_metric='rmse')#, eval_metric='mse')
    #alg.fit(X_train, y_train, eval_metric='rmse')

    # Predict training set:
    #dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predictions = alg.predict(X_test)

    # Print model report:
    print "\nModel Report"
    #print "Accuracy : %.4g" % metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)
    print "Accuracy : %.4g" % metrics.mean_squared_error(y_test, dtrain_predictions)

    plt.figure()
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


xgb1 = XGBRegressor(
 learning_rate=0.1,
 n_estimators=3993,
 max_depth=5,
 min_child_weight=3,
 gamma=0.2,
 subsample=0.9,
 colsample_bytree=0.7,
 nthread=4,
 reg_alpha=0.01,
 seed=0)

modelfit(xgb1, train, predictors)
#
# xgb1.fit(train[predictors], train[target])
# x_predict = xgb1.predict(X_test)
# print "Accuracy : %.4g" % metrics.mean_squared_error(np.expm1(x_predict), np.expm1(y_test))

#---------------cv---------------
# def log_transform(feature):
#     train[feature] = np.log1p(train[feature].values)
#     return

# compare = []
# xgb1.fit(X_train, y_train)
# x_predict = xgb1.predict(X_test)
# #x_predict = xgb1.predict(X_train)
# #print x_predict
# compare.append(np.expm1(x_predict.tolist()))
# compare.append(np.expm1(y_test.values.tolist()))
# #compare.append(np.expm1(y_train.values.tolist()))
# compare = map(list, zip(*compare))  # list traverse
# print compare
# name = ['predict', 'actual']
# test = pd.DataFrame(columns=name, data=compare)
# test.to_csv('compare.csv', index=False)


#---------------final-----------
# price_mean = pickle.load(open("tmp_mean.txt", 'r'))
# price_std = pickle.load(open("tmp_std.txt", 'r'))
# print price_mean, price_std
#
# xgb1.fit(train[predictors], train[target])
# x_predict = xgb1.predict(test)
# print x_predict
# name = ['price']
# test = pd.DataFrame(columns=name, data=np.expm1(x_predict))
# test.to_csv('result.csv', index=False)

'''
param_test1 = {
 'max_depth': range(3, 10, 2),
 'min_child_weight': range(1, 6, 2)
}
gsearch1 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=211, gamma=0, subsample=0.8, colsample_bytree=0.8, seed=0),
                        param_grid=param_test1, scoring='neg_mean_squared_error',
                        cv=10, iid=False, verbose=10)

#gsearch1 = XGBRegressor(n_estimators=1000,seed=0)
gsearch1.fit(train[predictors], train[target])
print gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
'''

'''
param_test2 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}
gsearch2 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=273, max_depth=5, min_child_weight=3, gamma=0.0, subsample=0.8, colsample_bytree=0.8, seed=0),
                        param_grid=param_test2, scoring='neg_mean_squared_error',
                        cv=10, iid=False, verbose=10)

#gsearch1 = XGBRegressor(n_estimators=1000,seed=0)
gsearch2.fit(train[predictors], train[target])
print gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
'''

'''
param_test3 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
gsearch3 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=273, max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=0.8, seed=0),
                        param_grid=param_test3, scoring='neg_mean_squared_error',
                        cv=10, iid=False, verbose=10)

#gsearch1 = XGBRegressor(n_estimators=1000,seed=0)
gsearch3.fit(train[predictors], train[target])
print gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
'''

'''
param_test4 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator=XGBRegressor(learning_rate=0.1, n_estimators=273, max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.9, colsample_bytree=0.7, seed=0),
                        param_grid=param_test4, scoring='neg_mean_squared_error',
                        cv=10, iid=False, verbose=10)

#gsearch1 = XGBRegressor(n_estimators=1000,seed=0)
gsearch4.fit(train[predictors], train[target])
print gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_

'''