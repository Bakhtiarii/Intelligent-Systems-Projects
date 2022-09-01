import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import math
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.utils import shuffle
from sklearn import preprocessing
from random import sample
from sklearn.ensemble import RandomForestClassifier
from func import Entropy,Information_Gain_2,Information_Gain_3,Additional_Entropy,Check_List_3_1,Gain_4,Check_List_2_1,Check_List_3_2,Main_Function,Check_List_2_2,Random_Sampling_2,Check_List_3_3,Evaluate_Data_Accuracy,Evaluate_Data_Confusion_Matrix,Check_List_2_3,Random_Sampling,predict,Indicate_Separate_Items,Random_Forest,Indicate_Number_Of_Separate_Items,Evaluate_Data_Accuracy_1,Evaluate_Data_Confusion_Matrix_1,Find_Maximum_Gain_1,Find_Maximum_Gain_2,Find_Zero_Label,Find_Maximum_Gain_3

myfile = pd.read_csv('prison_dataset.csv')
My_File = pd.read_csv('prison_dataset.csv',names=['Fiscal Year Released','Recidivism Reporting Year','Race - Ethnicity','Age At Release','Convicting Offense Classification','Convicting Offense Type','Convicting Offense Subtype','Main Supervising District','Release Type','Part of Target Population','Recidivism - Return to Prison numeric'],header = 1)
My_File = shuffle(My_File)
Attribute = ['Recidivism - Return to Prison numeric', 'Fiscal Year Released', 'Recidivism Reporting Year', 'Race - Ethnicity', 'Age At Release', 'Convicting Offense Classification', 'Convicting Offense Type', 'Convicting Offense Subtype', 'Main Supervising District', 'Release Type', 'Part of Target Population']
Range = int(0.8 * len(My_File))
Prediction = My_File.iloc[Range:,-1].values
Attribute = My_File.iloc[Range:,:-1].values
y = My_File.iloc[:Range,-1].values
X = My_File.iloc[:Range,:-1].values
[Train_Data,Test_Data] = Random_Sampling(myfile ,80)
Range = int(0.8 * len(My_File))
Prediction = My_File.iloc[Range:,-1].values
Attribute = My_File.iloc[Range:,:-1].values
y = My_File.iloc[:Range,-1].values
X = My_File.iloc[:Range,:-1].values
for Counter in range(10):
    le = preprocessing.LabelEncoder()
    le.fit(X[:,Counter])
    X[:,Counter] = le.transform(X[:,Counter])
    Attribute[:,Counter] = le.transform(Attribute[:,Counter])
Random_Forest_Out = RandomForestClassifier(criterion='entropy',max_depth=3, random_state=0)
Random_Forest_Out.fit(X, y)
Y_Prediction = Prediction
Y_Attribute = Random_Forest_Out.predict(Attribute)
                                            #Accuracy
print('Accuracy = ',accuracy_score(Y_Prediction, Y_Attribute)*100)
                                            #Confusion Matrix
print('Confusion Matrix = ')
print(confusion_matrix(Y_Prediction,Y_Attribute)[0])
print(confusion_matrix(Y_Prediction,Y_Attribute)[1])

