#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:36:40 2019

@author: sandilya
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def dropColumns(xData,dataset):
	# plot the data
	sns.pairplot(dataset,x_vars=xData.columns.values,y_vars=['medv'])
	plt.show()
	# drop the columns that has no relation with the dependent variable
	xData=xData.drop(columns=['chas','rad','zn','age','tax','b','indus','nox','crim','dis','ptratio'])
	sns.pairplot(dataset,x_vars=xData.columns.values,y_vars=['medv'])
	plt.show()
	return xData
def dataAnalysis(xData,dataset):
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	sns.distplot(dataset['medv'], bins=30)
	plt.show()
	#find the correalted variables
	corelationMatrix=dataset.corr().round(2)
	sns.heatmap(data=corelationMatrix,annot=True)
	plt.show()
	xData=dropColumns(xData,dataset)
	return xData
def main():
	#read the dataset
	dataset=pd.read_csv("boston.csv")
	# find dependent and independent variables
	x=dataset.iloc[:,:-1]
	y=dataset.iloc[:,-1]
	x=dataAnalysis(x,dataset)
	x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45,train_size=0.7)
	lr=LinearRegression()
	#fit the train data
	lr.fit(x_train,y_train)
	#predict for the test data
	y_pred=lr.predict(x_test)
	error=mean_squared_error(y_test,y_pred)
	print("The Mean squared error is {:.4f}".format(error))
if __name__=='__main__':
	main()