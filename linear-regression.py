#%%
# Install & import packages
import sys
!{sys.executable} -m pip install -r requirements.txt

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
# Import & setup data
data = pd.read_csv('./data/housing-data.csv')
x = data[['rm']]
y = data[['medv']]

features = x.shape[1]

x = (x - x.mean()) / x.std()
xB = np.c_[np.ones((len(x),1)),x]

iterations = 100
learningRate = 0.1
theta = np.random.randn(features+1, 1)

#%%
# Implementation

# error between predicted values and expected values, only used to show a nice graph
def calCost(theta, x, y):
    m = y.size # number of training examples
    predictions = x.dot(theta)
    return (1 / (2 * m)) * np.sum(np.square(predictions - y))

def gradientDescent(x, y, theta, learningRate=0.01, iterations=100):
    m = len(y)
    costHistory = np.zeros(iterations)
    thetaHistory = np.zeros((iterations,features+1))

    for it in range(iterations):

        prediction = np.dot(x,theta)

        theta = theta - (learningRate * (1/m)) * x.T.dot((prediction - y))
        thetaHistory[it,:] = theta.T
        costHistory[it]  = calCost(theta, x, y)

    return theta, costHistory, thetaHistory

def normalEquation(x, y):
    return np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x).dot(y) )

#%%
# Plot data

plt.figure(figsize=(10,6))

plt1 = plt.subplot(2, 1, 1)
plt1.set_title("Learning rate:{}".format(learningRate))

plt2 = plt.subplot(2, 1, 2)
plt2.set_title("Iterations:{}".format(iterations))
plt2.plot(x, y, 'b.')

theta = np.random.randn(features+1,1)

costHistory = np.zeros(iterations)

for i in range(iterations):
    theta,h,_ = gradientDescent(xB, y, theta, learningRate, 1)
    costHistory[i] = h[0]
    # plot data
    plt2.plot(x, xB.dot(theta), 'r-')

# plot cost history
plt1.plot(range(iterations),costHistory,'b.')

# plot the normal equation
theta = normalEquation(xB, y)
plt2.plot(x, xB.dot(theta), 'g-')

plt.show()