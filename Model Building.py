#run the file ALY 6110_Swapnil_Lokhande_EDA.py before executing this file 
#model building
#install matplotlib using python -mpip install matplotlib

import pandas as pd
import numpy as np
import matplotlib.pylab
from matplotlib.pylab import rcParams

#build base line model using Average sales by product

#import csv files
train_data=pd.read_csv("D:/Github/Big Mart Sales/train_modified.csv",na_values='.')
test_data=pd.read_csv("D:/Github/Big Mart Sales/test_modified.csv")


#mean_sales = train_data['Item_Outlet_Sales'].mean()
mean_sales=train_data.groupby('Item_Identifier').Item_Outlet_Sales.transform('mean')

#Define a dataframe with IDs for submission:
base1 = test_data[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("Baseline_algo.csv",index=False)

#generic function to be used by the algorithm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the train data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Remember the target had been normalized
    Sq_train = (dtrain[target])**2

    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target],Sq_train, cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    


    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    return(dtrain_predictions)
    

predictors = train_data.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])

#Linear Regression Model
#import the following libraries
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

LR = LinearRegression(normalize=True)
#predictors = train_data.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
train_data_predictions=modelfit(LR, train_data, test_data, predictors, target, IDcol, 'LR.csv')
#plot scatter plot to visualize how the model got fir with test data
#plot a scatter plot to see how good the model is fit
plt.scatter(train_data[target].values,train_data_predictions)
plt.title("Relation between Original Sales and Predicted Sales")
plt.xlabel('Original Sales')
plt.ylabel('Predicted Sales')
axes = plt.gca()
m, b = np.polyfit(train_data[target].values, train_data_predictions, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-')
plt.show()

coef1 = pd.Series(LR.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
plt.show()


#Decision Tree model
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
#predictors = train_data.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
train_data_predictions=modelfit(DT, train_data, test_data, predictors, target, IDcol, 'DT.csv')

#plot a scatter plot to see how good the model is fit
plt.scatter(train_data[target].values,train_data_predictions)
plt.title("Relation between Original Sales and Predicted Sales")
plt.xlabel('Original Sales')
plt.ylabel('Predicted Sales')
axes = plt.gca()
m, b = np.polyfit(train_data[target].values, train_data_predictions, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-')
plt.show()
coef3 = pd.Series(DT.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')
plt.show()

#Random Forest model
RF = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
train_data_predictions=modelfit(RF, train_data, test_data, predictors, target, IDcol, 'RF.csv')

#plot a scatter plot to see how good the model is fit
plt.scatter(train_data[target].values,train_data_predictions)
plt.title("Relation between Original Sales and Predicted Sales")
plt.xlabel('Original Sales')
plt.ylabel('Predicted Sales')
axes = plt.gca()
m, b = np.polyfit(train_data[target].values, train_data_predictions, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-')
plt.show()
coef5 = pd.Series(RF.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')
plt.show()
