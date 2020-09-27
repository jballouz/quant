# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:28:47 2018

@author: User-PC
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import util
from p05b_lwr import LocallyWeightedLinearRegression as LocallyWeightedLinearRegression
from linear_model import LinearModel

 # Load training set
cwd = os.getcwd()
x_train, y_train = util.load_dataset(cwd +'\\IBM_train_2009.csv', add_intercept=True)
x_eval, y_eval = util.load_dataset(cwd +'\\IBM_dev.csv', add_intercept=True)
x_test, y_test = util.load_dataset(cwd +'\\IBM_test.csv', add_intercept=True)
# Search tau_values for the best tau (lowest MSE on the validation set)
best_tau = 0
min_MSE = np.inf
tau_values=[6e-4,8e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
for i in range(len(tau_values)):
    print(i)
    LWR = LocallyWeightedLinearRegression(tau = tau_values[i], x_eval = x_eval)
    LWR.fit(x_train, y_train)
    MSE = np.mean((y_eval - LWR.predict())**2)
    print('For tau = ' + str(tau_values[i]) + ', the MSE value is ' + str(MSE))
    if MSE < min_MSE:
        best_tau = tau_values[i]
        min_MSE = MSE
    plt.plot(x_eval[:,1], y_eval, color='red', label='development')
    index=np.argsort(x_eval[:,-1])
    plt.plot(x_eval[index,-1], np.reshape(LWR.predict(),(x_eval.shape[0],1))[index], color="black", label='fit')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.title('Locally Weighted Regression on development data, tau = ' + str(tau_values[i]))
    plt.show()
# Fit a LWR model with the best tau value
print('For the validation set, the tau value which achieves the lowest MSE = ' + str(min_MSE) +  ', is tau = ' + str(best_tau))
# Run on the test set to get the MSE value
best_tau=0.001;
LWR = LocallyWeightedLinearRegression(tau = best_tau, x_eval = x_test)
LWR.fit(x_train, y_train)
MSE = np.mean((y_test - LWR.predict())**2)
print('For the test set, using the best tau = ' + str(best_tau) + ', the MSE value is ' + str(MSE))
# Plot data
plt.plot(x_test[:,1], y_test, color='green', label='test')
index=np.argsort(x_test[:,-1])
plt.plot(x_test[index,-1], np.reshape(LWR.predict(),(x_test.shape[0],1))[index], color="black", label='fit')
plt.xlabel('day of the year')
plt.ylabel('closing value')
plt.title('Locally Weighted Regression on IBM year 2017, tau = ' + str(best_tau))
plt.legend()
plt.savefig('IBM0001.png')
plt.show()

best_tau=0.01;
LWR = LocallyWeightedLinearRegression(tau = best_tau, x_eval = x_test)
LWR.fit(x_train, y_train)
MSE = np.mean((y_test - LWR.predict())**2)
print('For the test set, using the best tau = ' + str(best_tau) + ', the MSE value is ' + str(MSE))
# Plot data
plt.plot(x_test[:,1], y_test, color='green', label='test')
index=np.argsort(x_test[:,-1])
plt.plot(x_test[index,-1], np.reshape(LWR.predict(),(x_test.shape[0],1))[index], color="black", label='fit')
plt.xlabel('day of the year')
plt.ylabel('closing value')
plt.title('Locally Weighted Regression on IBM year 2017, tau = ' + str(best_tau))
plt.legend()
plt.savefig('IBM001.png')
plt.show()


 # Load training set
cwd = os.getcwd()
x_train, y_train = util.load_dataset(cwd +'\\dow_train_2009.csv', add_intercept=True)
x_eval, y_eval = util.load_dataset(cwd +'\\dow_dev.csv', add_intercept=True)
x_test, y_test = util.load_dataset(cwd +'\\dow_test.csv', add_intercept=True)
# Search tau_values for the best tau (lowest MSE on the validation set)
best_tau = 0
min_MSE = np.inf
tau_values=[6e-4,8e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
for i in range(len(tau_values)):
    print(i)
    LWR = LocallyWeightedLinearRegression(tau = tau_values[i], x_eval = x_eval)
    LWR.fit(x_train, y_train)
    MSE = np.mean((y_eval - LWR.predict())**2)
    print('For tau = ' + str(tau_values[i]) + ', the MSE value is ' + str(MSE))
    if MSE < min_MSE:
        best_tau = tau_values[i]
        min_MSE = MSE
    plt.plot(x_eval[:,1], y_eval, color='red', label='development')
    index=np.argsort(x_eval[:,-1])
    plt.plot(x_eval[index,-1], np.reshape(LWR.predict(),(x_eval.shape[0],1))[index], color="black", label='fit')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.title('Locally Weighted Regression on development data, tau = ' + str(tau_values[i]))
    plt.show()
# Fit a LWR model with the best tau value
print('For the validation set, the tau value which achieves the lowest MSE = ' + str(min_MSE) +  ', is tau = ' + str(best_tau))
# Run on the test set to get the MSE value
best_tau=0.01
LWR = LocallyWeightedLinearRegression(tau = best_tau, x_eval = x_test)
LWR.fit(x_train, y_train)
MSE = np.mean((y_test - LWR.predict())**2)
print('For the test set, using the best tau = ' + str(best_tau) + ', the MSE value is ' + str(MSE))
# Plot data
plt.plot(x_test[:,1], y_test, color='green', label='test')
index=np.argsort(x_test[:,-1])
plt.plot(x_test[index,-1], np.reshape(LWR.predict(),(x_test.shape[0],1))[index], color="black", label='fit')
plt.xlabel('day of the year')
plt.ylabel('closing value')
plt.title('Locally Weighted Regression on DOW JONES year 2017, tau = ' + str(best_tau))
plt.legend()
plt.savefig('dow001.png')
plt.show()


# Load training set
cwd = os.getcwd()
x_train, y_train = util.load_dataset(cwd +'\\sandp_train_2009.csv', add_intercept=True)
x_eval, y_eval = util.load_dataset(cwd +'\\sandp_dev.csv', add_intercept=True)
x_test, y_test = util.load_dataset(cwd +'\\sandp_test.csv', add_intercept=True)
# Search tau_values for the best tau (lowest MSE on the validation set)
best_tau = 0
min_MSE = np.inf
tau_values=tau_values=[6e-4,8e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
for i in range(len(tau_values)):
    print(i)
    LWR = LocallyWeightedLinearRegression(tau = tau_values[i], x_eval = x_eval)
    LWR.fit(x_train, y_train)
    MSE = np.mean((y_eval - LWR.predict())**2)
    print('For tau = ' + str(tau_values[i]) + ', the MSE value is ' + str(MSE))
    if MSE < min_MSE:
        best_tau = tau_values[i]
        min_MSE = MSE
    plt.plot(x_eval[:,1], y_eval, color='red', label='development')
    index=np.argsort(x_eval[:,-1])
    plt.plot(x_eval[index,-1], np.reshape(LWR.predict(),(x_eval.shape[0],1))[index], color="black", label='fit')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.title('Locally Weighted Regression on development data, tau = ' + str(tau_values[i]))
    plt.show()
# Fit a LWR model with the best tau value
print('For the validation set, the tau value which achieves the lowest MSE = ' + str(min_MSE) +  ', is tau = ' + str(best_tau))
# Run on the test set to get the MSE value
best_tau=0.1
LWR = LocallyWeightedLinearRegression(tau = best_tau, x_eval = x_test)
LWR.fit(x_train, y_train)
MSE = np.mean((y_test - LWR.predict())**2)
print('For the test set, using the best tau = ' + str(best_tau) + ', the MSE value is ' + str(MSE))
# Plot data
plt.plot(x_test[:,1], y_test, color='green', label='test')
index=np.argsort(x_test[:,-1])
plt.plot(x_test[index,-1], np.reshape(LWR.predict(),(x_test.shape[0],1))[index], color="black", label='fit')
plt.xlabel('day of the year')
plt.ylabel('closing value')
plt.title('Locally Weighted Regression on S&P 500 year 2017, tau = ' + str(best_tau))
plt.legend()
plt.savefig('sandp01.png')
plt.show()

best_tau=0.01
LWR = LocallyWeightedLinearRegression(tau = best_tau, x_eval = x_test)
LWR.fit(x_train, y_train)
MSE = np.mean((y_test - LWR.predict())**2)
print('For the test set, using the best tau = ' + str(best_tau) + ', the MSE value is ' + str(MSE))
# Plot data
plt.plot(x_test[:,1], y_test, color='green', label='test')
index=np.argsort(x_test[:,-1])
plt.plot(x_test[index,-1], np.reshape(LWR.predict(),(x_test.shape[0],1))[index], color="black", label='fit')
plt.xlabel('day of the year')
plt.ylabel('closing value')
plt.title('Locally Weighted Regression on S&P 500 year 2017, tau = ' + str(best_tau))
plt.legend()
plt.savefig('sandp001.png')
plt.show()


