import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR
# Scikit-learn library: For SVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab




df = pd.read_csv('DATASET.csv') # Reading the file .csv
df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
df.loc[fil, 'EDUCATION'] = 4


df = pd.DataFrame(df)
df = df.sample(frac=0.05, random_state = 1)
df.drop(['PAY_4', 'PAY_5', 'PAY_6','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5','PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'] , 1 , inplace = True)

var_list=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


predictor_var = df[['EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2']]


targets=df['Class']
pred_train, pred_test, tar_train, tar_test= train_test_split(predictor_var, targets)#, test_size= 0.3)
'''
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(pred_train, tar_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(pred_train)
print(X_new.shape)
print(X_new[0])
'''

Fraud = df[df['Class'] == 1]
Valid = df[df['Class'] == 0]
print(len(Fraud))
print(len(Valid))
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)



classify = svm.SVC(kernel='linear', class_weight={0:0.60, 1:0.40})
'''
classify= classify.fit(pred_train,tar_train)
'''
filename = 'finalized_model.sav'
'''
pickle.dump(classify , open(filename1, 'wb'))
'''
classify = pickle.load(open(filename, 'rb'))


predictions=classify.predict(pred_test)
x=sklearn.metrics.confusion_matrix(tar_test,predictions)
y=sklearn.metrics.accuracy_score(tar_test,predictions)


print("Confusion Matrix\n",x)
print("\nTrue Positives : ",x[0][0],"\nFalse Positives:",x[0][1],"\nTrue Negatives :",x[1][0],"\nFalse Negatives:",x[1][1],"\n")
print("\nAccuracy Score/ Efficiency:",y,"\n")
#Precision tells how close we are to the true value=TP/(TP+FP)
print("Precision :",x[0][1]/(x[0][0]+x[0][1]), "\n")


'''
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
'''

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
        list_prediction=[[z3,z4,z5,z6,z7,z8,z12,z13,z17,z18,z19]]

        predictions=classify.predict(list_prediction)
        print(predictions)
        if predictions==1:
            print("Fradulent")
        else:
            print("Not Fraudulent")

        morePred=int(input("For More Predictions(0- false, 1-true): "))
        if morePred == 0:
            break
