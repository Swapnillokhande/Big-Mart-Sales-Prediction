#install python 3.7.0
#install pandas using pip install pandas in windows
#import required libraries
#ensure that the below libraries are installed before importing them

import pandas as pd
import numpy as np

#read data available in csv file

train_data=pd.read_csv("D:/Github/Big Mart Sales/Train_Data.csv",na_values='.')
test_data=pd.read_csv("D:/Github/Big Mart Sales/Test_Data.csv")

#Data exploration

#combine the data sets (train and test) to check the properties

train_data['source']='train_data'
test_data['source']='test_data'
comb_data = pd.concat([train_data, test_data],ignore_index=True)
print ("number of rows and columns in Train dataset ",train_data.shape)
print("number of rows and columns in Test dataset ",test_data.shape)
print("number of rows and columns in combined dataset ",comb_data.shape)

#find the unique entries
print("\nSummary for unique values\n",comb_data.apply(lambda x: len(x.unique())))

#check whether the combined dataset contains any missing values in any column

print("Summary for missing values\n",comb_data.isnull().sum())

#Check the percentage of null values per variable
print(comb_data.isnull().sum()/comb_data.shape[0]*100) #show values in percentage

#get the summary of the combined dataset
print("\nStatistical Summary of Item Visibility\n",comb_data['Item_Visibility'].describe())
print("\n",comb_data.describe())

#Filter categorical variables
category_column = [x for x in comb_data.dtypes.index if comb_data.dtypes[x]=='object']

#Exclude ID cols and source:
category_column = [x for x in category_column if x not in ['Item_Identifier','Outlet_Identifier','source']]

#Print frequency of object type categories
for column in category_column:
    print ('\nFrequency of Categories for varible %s'%column)
    print (comb_data[column].value_counts())


#Data cleansing
#impute missing data
#Determine the average weight per item:
item_avg_weight = comb_data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight)



#Impute data and check missing values before and after imputation for Item_Weight
#Get a boolean variable specifying missing Item_Weight values
miss_cells = comb_data['Item_Weight'].isnull() 
print ('missing values before imputation: %d'% sum(miss_cells))

#fill missing Item_weight based on the mean weight of each product (use Item_Identifier)
comb_data.loc[comb_data.Item_Weight.isnull(), 'Item_Weight'] = comb_data.groupby('Item_Identifier').Item_Weight.transform('mean')
print ('Final count of missing values: %d'% sum(comb_data['Item_Weight'].isnull()))


#impute the missing values in column Outlet_Type
#Determing the mode for each Outlet_Size categorizing with the Outlet_Type
outlet_size_mode = comb_data.pivot_table(values='Outlet_Size',
                                   columns='Outlet_Type',
                                   aggfunc=lambda x: x.mode().iat[0])
print ('Mode for each Outlet_Type:')
print (outlet_size_mode)

#Impute data and check #missing values before and after imputation to confirm
miss_cells = comb_data['Outlet_Size'].isnull()
print ('\nOrignal missing values: %d'% sum(miss_cells))

#Impute data and check #missing values before and after imputation to confirm
comb_data.loc[comb_data.Outlet_Size.isnull(),'Outlet_Size']=comb_data.groupby('Outlet_Type')['Outlet_Size'].apply(lambda x:x.fillna(x.value_counts().index.tolist()[0]))
print ('\nFinal count of missing values: %d'% sum(comb_data['Outlet_Size'].isnull()))


#perform Data Engineering
#visibility of some Item_Identifier was found to be 0, replace 0 with the mean visibility of that product
#Determine average visibility of a product

visibility_avg = comb_data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#find cells with 0 visibility
#Impute 0 values with mean visibility of that product:
miss_cell = (comb_data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(miss_cell))

comb_data.loc[comb_data['Item_Visibility'] == 0,'Item_Visibility'] = comb_data.groupby('Item_Identifier').Item_Visibility.transform('mean')

print ('Number of 0 values after modification: %d'%sum(comb_data['Item_Visibility'] == 0))

#Determine the year of operartion of store
#Remember the data is from 2013
comb_data['Outlet_Years'] = 2013 - comb_data['Outlet_Establishment_Year']
print(comb_data['Outlet_Years'].describe())

#Determine another variable with means ratio
#data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
#print (data['Item_Visibility_MeanRatio'].describe())

#create a broad category Item_Type_Combined of items using Item_Identifier

#Get the first two characters of ID:
comb_data['Item_Type_Combined'] = comb_data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
comb_data['Item_Type_Combined'] = comb_data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
print(comb_data['Item_Type_Combined'].value_counts())

#modify the categories in Item_Fat :
print ('Original Categories:')
print (comb_data['Item_Fat_Content'].value_counts())


comb_data['Item_Fat_Content'] = comb_data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})


print (comb_data['Item_Fat_Content'].value_counts)

#Mark Non-consumable as Non_Edible in Item_Fat_Content
comb_data.loc[comb_data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"

print ('\nModified Categories:')
print(comb_data['Item_Fat_Content'].value_counts())

#Feature transformation
#Create variable Item_Visibility_Mean_Ratio
func = lambda x: x['Item_Visibility']/visibility_avg['Item_Visibility'][visibility_avg.index == x['Item_Identifier']][0]
comb_data['Item_Visibility_MeanRatio'] = comb_data.apply(func,axis=1).astype(float)
print(comb_data['Item_Visibility_MeanRatio'].describe())



#Import library:
#to import below library, first install scikit using pip install -U scikit-learn
#pre-requisite for installing scikit
#Scikit-learn requires:

#Python (>= 2.7 or >= 3.4),
#NumPy (>= 1.8.2),
#SciPy (>= 0.13.3).
#also upgrade the pip version using python -m pip install --upgrade (recommended)

#scikit-learn accepts only numerical variables, convert all categories of nominal variables into numeric types.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
comb_data['Outlet'] = le.fit_transform(comb_data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    comb_data[i] = le.fit_transform(comb_data[i])

#One-Hot-Coding refers to creating dummy variables, one for each category of a categorical variable.
#For example, the Item_Fat_Content has 3 categories – ‘Low Fat’, ‘Regular’ and ‘Non-Edible’.
#One hot coding will remove this variable and generate 3 new variables.
#Each will have binary numbers – 0 (if the category is not present) and 1(if category is present).

#One Hot Coding:
comb_data = pd.get_dummies(comb_data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

#Export data set
#break combined data frame into test and train

#Drop the columns which have been converted to different types:
comb_data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = comb_data.loc[comb_data['source']=="train_data"]
test = comb_data.loc[comb_data['source']=="test_data"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("D:/Github/Big Mart Sales/train_modified.csv",index=False)
test.to_csv("D:/Github/Big Mart Sales/test_modified.csv",index=False)
