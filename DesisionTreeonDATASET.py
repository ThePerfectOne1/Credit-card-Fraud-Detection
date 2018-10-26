

import numpy as np
import pandas as pd
import pydotplus
import  graphviz
from sklearn.tree import export_graphviz
# from sklearn.metrics import classification_report
import sklearn.metrics
# from pandas import Series, DataFrame
from sklearn import tree
# from os import *
# import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

#retval = os.getcwd()
#print ("Current working directory %s" % retval)
#os.chdir("C:\Users\kungfu_panda\Downloads\ML Coursera")
# reding data
data_clean =pd.read_csv("DATASET.csv")
#var_list= ah_data.columns.tolist
#print(var_list)

print(data_clean.shape, "\n", data_clean.columns.tolist)

#data_clean.dtypes
#data_clean.describe()
print(data_clean.describe())
data_clean.loc[data_clean.MARRIAGE == 0, 'MARRIAGE'] = 3
#data_clean.MARRIAGE.value_counts()

fil = (data_clean.EDUCATION == 5) | (data_clean.EDUCATION == 6) | (data_clean.EDUCATION == 0)
data_clean.loc[fil, 'EDUCATION'] = 4

#print(data_clean.EDUCATION.value_counts())
data_clean = data_clean.sample(frac=0.05, random_state=1 ) 

#Modeling
var_list=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
#predictor/ explanatory variables
predictor_var = data_clean[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

count_classes = pd.value_counts(data_clean['Class'])#, sort = True)#.sort_index()
print("Total of Class 0 and Class 1:\n", count_classes)

#response variable/ target variable
targets=data_clean.Class
#splitting data for training and testing
pred_train, pred_test, tar_train, tar_test= train_test_split(predictor_var, targets)#, test_size= 0.3)
# BY Default 25% is the test data size
pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

# fit method: Training the model on the data, storing the information learned from the data
classify=DecisionTreeClassifier()
classify= classify.fit(pred_train,tar_train)

predictions=classify.predict(pred_test)

x=sklearn.metrics.confusion_matrix(tar_test,predictions)
y=sklearn.metrics.accuracy_score(tar_test,predictions)
print("Confusion Matrix\n",x)
print("\nTrue Positives : ",x[0][0],"\nFalse Positives:",x[0][1],"\nTrue Negatives :",x[1][0],"\nFalse Negatives:",x[1][1],"\n")
print("\nAccuracy Score/ Efficiency:",y,"\n")
#Precision tells how close we are to the true value=TP/(TP+FP) 
print("Precision :",x[0][1]/(x[0][0]+x[0][1]), "\n")

#display relative importance of variables
list1=classify.feature_importances_ #list of features
max_in_list=max(classify.feature_importances_)
#print(list1)
print(classify.feature_importances_)
print("\n")
print('Most significant explanatory variable:',max_in_list, "\n")
a= np.amax(list1)
print("List Length: ",len(list1), "\n")#,"Index of the most significant variable :", list1.index(n) )
index = np.argmax(list1)    #max ka index argument of max
print("Max Index:",index,"\n")
print("Most significant variable:",var_list[index]) #var_list is the list of all variables



#Display Decision Tree


#import collections
#from sklearn.tree import export_graphviz
#from IPython.display import Image


dot_data=tree.export_graphviz(classify,out_file=None, filled=True, rounded=True)
graph=pydotplus.graph_from_dot_data(dot_data)
#colors=('red', 'black')
graph.write_png('treeDATASET.png')

    #user input 
for _ in range (0, 20):
    try:
        
        print("\nUser Input:")
        z1=(float(input('LIMIT_BAL(10,000 - 1,00,000): '))) 
        z2=float(input('SEX(1- male, 2-female): ')) 
        z3=float(input('EDUCATION(1-4): ')) 
        z4=float(input('MARRIAGE(1-3): ' )) 
        z5=float(input('AGE(21-80): ' )) 
        
        z6=float(input( 'PAY_1(-2 - 8): ')) 
        z7=float(input('PAY_2(-2 - 8): ' ))
        z8=float(input('PAY_3(-2 - 8): '))
        z9=float(input( 'PAY_4(-2 - 8): '  )) 
        z10=float(input('PAY_5(-2 - 8): '))
        z11=float(input( 'PAY_6(-2 - 8): ' ))
        
        z12=float(input('BILL_AMT1(-5,00,000 - 5,00,000): '))
        z13=float(input('BILL_AMT2(-5,00,000 - 5,00,000): '))
        z14=float(input('BILL_AMT3(-5,00,000 - 5,00,000): '))
        z15=float(input('BILL_AMT4(-5,00,000 - 5,00,000): '))
        z16=float(input('BILL_AMT5(-5,00,000 - 5,00,000): '))
        z17=float(input( 'BILL_AMT6(-5,00,000 - 5,00,000): ' ))
        
        z18=float(input('PAY_AMT1(0-2,00,000): '))
        z19=float(input('PAY_AMT2(0-2,00,000): '))
        z20=float(input('PAY_AMT3(0-2,00,000): '))
        z21=float(input('PAY_AMT4(0-2,00,000): ' ))
        z22=float(input('PAY_AMT5(0-2,00,000): ' )) 
        z23=float(input('PAY_AMT6(0-2,00,000): ' ))
    except ValueError:
        print("Input ERROR!! \nRe-enter Inputs")
    else:    
        if z1 not in range(10000, 100001) or z2 not in range(1,3) or z3 not in range(1,5) or z4 not in range(1,4) or z5 not in range(21, 81) or z6 not in range(-2,9) or z7 not in range(-2,9) or z8 not in range(-2,9) or z9 not in range(-2,9) or z10 not in range(-2,9)or z11 not in range(-2,9) or z12 not in range(-500000,500001) or z13 not in range(-500000,500001) or z14 not in range(-500000,500001) or z15 not in range(-500000,500001) or z16 not in range(-500000,500001) or z17 not in range(-500000,500001) or z18 not in range(0,200001) or z19 not in range(0,200001) or z20 not in range(0,200001) or z21 not in range(0,200001) or z22 not in range(0,200001) or z23 not in range(0,200001):
            print("Input ERROR!! \nRe-enter Inputs")
            continue                
                        
        T=[0,1]
        list_prediction=[[z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23]]
        
        predictions=classify.predict(list_prediction,T)
        print(predictions)
        if predictions==1:
            print("Fradulent")
        else:
            print("Not Fraudulent")
            
        morePred=int(input("For More Predictions(0- false, 1-true): "))
        if morePred == 0:
            break       
