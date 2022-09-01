import pandas as pd
import numpy as np
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import math
from math import sqrt
#from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.utils import shuffle
#from sklearn import preprocessing
from random import sample
#from sklearn.ensemble import RandomForestClassifier
from func import Entropy,Information_Gain_2,Information_Gain_3,Additional_Entropy,Check_List_3_1,Gain_4,Check_List_2_1,Check_List_3_2,Main_Function,Check_List_2_2,Random_Sampling_2,Check_List_3_3,Evaluate_Data_Accuracy,Evaluate_Data_Confusion_Matrix,Check_List_2_3,Random_Sampling,predict,Indicate_Separate_Items,Indicate_Number_Of_Separate_Items,Find_Maximum_Gain_1,Find_Maximum_Gain_2,Find_Zero_Label,Find_Maximum_Gain_3

Depth = 3
myfile = pd.read_csv('prison_dataset.csv')
Attribute = ['Recidivism - Return to Prison numeric', 'Fiscal Year Released', 'Recidivism Reporting Year', 'Race - Ethnicity', 'Age At Release', 'Convicting Offense Classification', 'Convicting Offense Type', 'Convicting Offense Subtype', 'Main Supervising District', 'Release Type', 'Part of Target Population']
dataset = pd.read_csv('prison_dataset.csv',names=['Fiscal Year Released','Recidivism Reporting Year','Race - Ethnicity','Age At Release','Convicting Offense Classification','Convicting Offense Type','Convicting Offense Subtype','Main Supervising District','Release Type','Part of Target Population','Recidivism - Return to Prison numeric'],header = 1)
[Train_Data,Test_Data] = Random_Sampling(myfile ,80)
dataset = shuffle(dataset)
First_Branch = Find_Maximum_Gain_1(Train_Data, Attribute[0], Attribute[1], Attribute[2], Attribute[3], Attribute[4], Attribute[5], Attribute[6], Attribute[7], Attribute[8], Attribute[9], Attribute[10])
Train = Random_Sampling_2(dataset)[0]
[First_Branch, Zero_item, Result] = Find_Zero_Label(First_Branch)
Test = Random_Sampling_2(dataset)[1] 
Attribute.remove(First_Branch[0])
tree = Main_Function(Train,Train,Train.columns[:-1],0,Depth)
if Zero_item!=[]:
    Ways = [[First_Branch[0],Zero_item,Result]]
Second_Branch_1 = []
Second_Branch_2 = []
Second_Branch_1 = Find_Maximum_Gain_2(Train_Data, Attribute[0], Attribute[1], Attribute[2], Attribute[3], Attribute[4], Attribute[5], Attribute[6], Attribute[7], Attribute[8], Attribute[9], First_Branch[0], First_Branch[1][0])
Second_Branch_2 = Find_Maximum_Gain_2(Train_Data, Attribute[0], Attribute[1], Attribute[2], Attribute[3], Attribute[4], Attribute[5], Attribute[6], Attribute[7], Attribute[8], Attribute[9], First_Branch[0], First_Branch[1][1])
[Second_Branch_1, Zero_item_1, Result_1] = Find_Zero_Label(Second_Branch_1)
[Second_Branch_2, Zero_item_2, Result_2] = Find_Zero_Label(Second_Branch_2)
Attribute.remove(Second_Branch_1[0])
Attribute.remove(Second_Branch_2[0])
Third_Branch = Find_Maximum_Gain_3(Train_Data, Attribute[0], Attribute[1], Attribute[2], Attribute[3], Attribute[4], Attribute[5], Attribute[6], Attribute[7], First_Branch[0], First_Branch[1][1], Second_Branch_1[0], Second_Branch_1[1][0] )
                                    # Write in txt file
textfile = open('tree.txt','w+')
with open('tree.txt', 'w') as f:
    f.write(str(tree))
                                    # Confusion Matrix
Evaluate_Data_Confusion_Matrix(Test,tree)
                                    # Accuracy
Evaluate_Data_Accuracy(Test,tree)


