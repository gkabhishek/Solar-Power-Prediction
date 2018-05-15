# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:56:39 2018

@author: gkabh
"""


# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# Set Working Directory
os.chdir('G:\Engineering Management\CourseWork\Spring 2018\Machine Learning and Pattern Recognition\Project')

# Importing Dataset
weather = pd.read_csv('weather.csv')
power = pd.read_csv('power_formatted.csv')

power['Date'] =  pd.to_datetime(power['Date'], infer_datetime_format = True)
weather['Date'] = pd.to_datetime(weather['Date'], infer_datetime_format = True)

# Combining Dataset based on Date time Column
final = pd.merge(weather, power, left_on='Date', right_on = 'Date')
final['Date'] =  pd.to_datetime(final['Date'], infer_datetime_format = True)

# Removing Date
del final['Date']
# Correlation
final.corr()
#Separating Independent and Dependent Variables
abc = np.split(final, [9], axis=1)
x = abc[0]
y = abc[1]

# Splitting the Dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.75,
                                     test_size = 0.25,random_state = 438)

x_train = x_train.reset_index()
y_train = y_train.reset_index()
x_test = x_test.reset_index()
y_test = y_test.reset_index()

del x_train['index'] 
del y_train['index'] 
del x_test['index']
del y_test['index']

# Linear Model 
LM = LinearRegression(fit_intercept=True)
LM = LM.fit(x_train,y_train)
LM_pred = pd.DataFrame(LM.predict(x_test))
LM_trainpred = LM.predict(x_train)
# Model Evaluation
print(np.sqrt(metrics.mean_squared_error(y_train,LM_trainpred)))
print(np.sqrt(metrics.mean_squared_error(y_test,LM_pred)))
print('Coefficients: \n', model1.coef_)
print('Variance score: %.2f' % metrics.r2_score(y_test,LM_pred))
from sklearn.model_selection import cross_val_score

print(cross_val_score(LM, y_test, LM_pred, cv=10).mean())
# Plot the model
plt.plot(y_test.iloc[1:100,:].values, label = "Actual")
plt.plot(LM_pred.iloc[1:100,:].values, label = "Predicted")
plt.legend()
plt.show()

plt.plot(y_train.iloc[1:100,:].values,label = "Actual")
plt.plot(LM_trainpred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()

# SVR
from sklearn.svm import SVR
svr = SVR(kernel = 'poly', degree = 1,C=1.5)
svr = svr.fit(x_train,y_train)
svr_pred = pd.DataFrame(svr.predict(x_test))
svr_trainpred = pd.DataFrame(svr.predict(x_train))
# Model Evaluation
print(np.sqrt(metrics.mean_squared_error(y_train,svr_trainpred)))
print(np.sqrt(metrics.mean_squared_error(y_test,svr_pred)))
print('Variance score: %.2f' % metrics.r2_score(y_test,svr_pred))
print(cross_val_score(svr, y_test, svr_pred, cv=10).mean())

plt.plot(y_test.iloc[1:100,:].values,label = "Actual")
plt.plot(svr_pred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()


# Plot the model
plt.plot(y_test.iloc[1:100,:].values, label = "Actual")
plt.plot(svr_pred.iloc[1:100,:].values, label = "Predicted")
plt.legend()
plt.show()

plt.plot(y_train.iloc[1:100,:].values,label = "Actual")
plt.plot(svr_trainpred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()
# Random Forest
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators = 10, max_features=2, 
                                    max_depth=1000, min_samples_leaf=1, 
                                    min_samples_split=2, n_jobs=-1)
RF.fit(x_train, y_train)
RF_pred = pd.DataFrame(RF.predict(x_test))
RF_trainpred = pd.DataFrame(RF.predict(x_train))
# Model Evaluation
print(np.sqrt(metrics.mean_squared_error(y_train,RF_trainpred)))
print(np.sqrt(metrics.mean_squared_error(y_test,RF_pred)))
print('Variance score: %.2f' % metrics.r2_score(y_test,RF_pred))
print(cross_val_score(RF, y_test, RF_pred, cv=10).mean())

#Plot the Model
plt.plot(y_test.iloc[1:100,:].values,label = "Actual")
plt.plot(RF_pred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()

plt.plot(y_train.iloc[1:100,:].values,label = "Actual")
plt.plot(RF_trainpred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()

# Multi Layer Perceptron
from sklearn.neural_network import MLPRegressor 

mlp =  MLPRegressor(hidden_layer_sizes=(13,),activation='relu',solver='adam', 
                    alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, power_t=0.5, max_iter=1000, 
                    shuffle=True, random_state=None, tol=0.0001, warm_start=False,
                    early_stopping=False,validation_fraction=0.1,
                    beta_1=0.9,beta_2=0.999,epsilon=1e-08)
                    

mlp = mlp.fit(x_train, y_train)
MLP_pred = pd.DataFrame(mlp.predict(x_test))
MLP_trainpred = pd.DataFrame(mlp.predict(x_train))
#Model Evaluation
print(np.sqrt(metrics.mean_squared_error(y_test,MLP_pred)))
print(np.sqrt(metrics.mean_squared_error(y_train,MLP_trainpred)))
print('Variance score: %.2f' % metrics.r2_score(y_test,MLP_pred))
print(cross_val_score(mlp, y_test, MLP_pred, cv=10).mean())

# Plot the model
plt.plot(y_test.iloc[1:100,:].values,label = "Actual")
plt.plot(MLP_pred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()

plt.plot(y_test.iloc[1:100,:].values,label = "Actual")
plt.plot(MLP_trainpred.iloc[1:100,:].values,label = "Predicted")
plt.legend()
plt.show()

# Polynomial Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly =pd.DataFrame(poly_reg.fit_transform(x_train))
poly_reg.fit(x_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y_train)

lin_reg.predict(x_test)

y_predPoly = pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(x_test)))
y_trainpredPoly = pd.DataFrame(lin_reg_2.predict(poly_reg.fit_transform(x_train)))
print(np.sqrt(metrics.mean_squared_error(y_test,y_predPoly)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredPoly)))

plt.plot(y_test.iloc[1:100,:].values)
plt.plot(y_predPoly.iloc[1:100,:].values)
plt.show()


plt.plot(y_test.iloc[1:100,:].values)
plt.plot(y_predPoly.iloc[1:100,:].values)
plt.show()

# Elastic Net 
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
enet = linear_model.ElasticNet(alpha=10,
                 fit_intercept=True, l1_ratio=1, max_iter=1000)
enet1 = enet.fit(x_train, y_train)
y_predenet = pd.DataFrame( enet1.predict(x_test) )
y_trainpredenet = pd.DataFrame( enet1.predict(x_train) )
print(np.sqrt(metrics.mean_squared_error(y_test,y_predenet)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredenet)))

print('Variance score: %.2f' % metrics.r2_score(y_test,y_predenet))

plt.plot(y_test.iloc[1:100,:].values)
plt.plot(y_predenet.iloc[1:100,:].values)
plt.show()

# Linear Regression Lasso
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#x_trainLasso = sc_x.fit_transform(x_train)
#y_trainLasso = sc_y.fit_transform(y_train)
#x_testLasso = sc_x.fit_transform(x_test)
#y_testLasso = sc_y.fit_transform(y_test)

lasso = linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None)
lasso1 = lasso.fit(x_train,y_train)
y_predLasso = pd.DataFrame(lasso1.predict(x_test))
y_trainpredLasso = pd.DataFrame(lasso1.predict(x_train))
print(np.sqrt(metrics.mean_squared_error(y_test,y_predLasso)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredLasso)))

print('Variance score: %.2f' % metrics.r2_score(y_test,y_predLasso))

plt.plot(y_test.iloc[1:100,:].values)
plt.plot(y_predenet.iloc[1:100,:].values)
plt.show()

# Linear Regression Ridge

ridge = linear_model.RidgeCV(alphas=(1.0,0.1,0.01,.005,.0025,.001,.00025), fit_intercept=True, 
                                normalize=True) 
ridge1 = linear_model.Ridge(alpha=0.001, fit_intercept=True, 
                                normalize=True)                    
ridge1 = ridge.fit(x_train,y_train)
ridge1.alpha_
y_predRidge = pd.DataFrame(ridge1.predict(x_test))
y_trainpredRidge = pd.DataFrame(ridge1.predict(x_train))
print(np.sqrt(metrics.mean_squared_error(y_test,y_predRidge)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredRidge)))

print('Variance score: %.2f' % metrics.r2_score(y_test,y_predRidge))
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

regressorDT = DecisionTreeRegressor(random_state = 0)
regressorDT.fit(x_train, y_train)
y_predDT = regressorDT.predict(x_test)
y_trainpredDT = regressorDT.predict(x_train)
print(np.sqrt(metrics.mean_squared_error(y_test,y_predDT)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredDT)))

print('Variance score: %.2f' % metrics.r2_score(y_test,y_predDT))

# AdaBoost
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(base_estimator = regressorDT ,learning_rate=1.0, loss='linear',
                        n_estimators=50, random_state=None)
ada.fit(x_train,y_train)
y_predada = ada.predict(x_test)
y_trainpredada = ada.predict(x_train)
print(np.sqrt(metrics.mean_squared_error(y_test,y_predada)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredada))) 

print('Variance score: %.2f' % metrics.r2_score(y_test,y_predada))
# ExtraTree Classifier

from sklearn.ensemble import ExtraTreesRegressor

extra = ExtraTreesRegressor(n_estimators=10, criterion='mse', 
                            max_depth=None, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_features='auto', max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            bootstrap=False, 
                            random_state=None, warm_start=False)


extra.fit(x_train, y_train)
y_predET = extra.predict(x_test)
y_trainpredET = extra.predict(x_train)
print(np.sqrt(metrics.mean_squared_error(y_test,y_predET)))
print(np.sqrt(metrics.mean_squared_error(y_train,y_trainpredET)))
print('Variance score: %.2f' % metrics.r2_score(y_test,y_predET))


