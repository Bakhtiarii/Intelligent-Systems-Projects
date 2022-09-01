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
                                    #Entropy
def Entropy(input_1, input_2):
    if (input_1==0)or(input_2==0):
        Entropy = 0
    else:
        Entropy = (-input_1/(input_1 + input_2))*math.log((input_1/(input_1 + input_2)),2) + (-input_2/(input_1 + input_2))*math.log(input_2/(input_1 + input_2),2)
    return Entropy

                                    #Gain_2
def Information_Gain_2(input_1, input_2, input_3, input_4):
    if (input_1+input_2+input_3+input_4) != 0:
        Gain_2 = Entropy(input_1+input_3, input_2+input_4) - ((input_1+input_2)/(input_1+input_2+input_3+input_4))*Entropy(input_1, input_2) - ((input_3+input_4)/(input_1+input_2+input_3+input_4))*Entropy(input_3, input_4)
    else:
        Gain_2 = 0
    return Gain_2

                                    #Gain_3
def Information_Gain_3(input_1, input_2, input_3, input_4, input_5, input_6):
    if (input_1+input_2+input_3+input_4+input_5+input_6) != 0:
        Gain_3 = Entropy(input_1+input_3+input_5, input_2+input_4+input_6) - ((input_1+input_2)/(input_1+input_2+input_3+input_4+input_5+input_6))*Entropy(input_1, input_2) - ((input_3+input_4)/(input_1+input_2+input_3+input_4+input_5+input_6))*Entropy(input_3, input_4) - ((input_5+input_6)/(input_1+input_2+input_3+input_4+input_5+input_6))*Entropy(input_5, input_6)
    else:
        Gain_3 = 0
    return Gain_3
                                    #Gain_4
def Gain_4(data,name1,name2='Recidivism - Return to Prison numeric'):
    E2 = Additional_Entropy(data[name2])
    input,number= np.unique(data[name1],return_counts=True)
    E1 = np.sum([(number[i]/np.sum(number))*Additional_Entropy(data.where(data[name1]==input[i]).dropna()[name2]) for i in range(len(input))])
    Gain4 = E2 - E1
    return Gain4

                                    #Check_List_3_1
def Check_List_3_1(dictionaryObject, name_0, name_1, var_1, var_2, var_3):
    column_1 = dictionaryObject[name_1]
    column_2 = dictionaryObject[name_0]
    column_1_1 = column_1_2 = column_1_3 = column_1_4 = column_1_5 = column_1_6 = 0
    for Counter in column_1:
        if (column_1[Counter]==var_1)and(column_2[Counter]==1):
            column_1_1 = column_1_1 + 1
        elif (column_1[Counter]==var_1)and(column_2[Counter]==0):
            column_1_2 = column_1_2 + 1
        elif (column_1[Counter]==var_2)and(column_2[Counter]==1):
            column_1_3 = column_1_3 + 1
        elif (column_1[Counter]==var_2)and(column_2[Counter]==0):
            column_1_4 = column_1_4 + 1
        elif (column_1[Counter]==var_3)and(column_2[Counter]==1):
            column_1_5 = column_1_5 + 1
        elif (column_1[Counter]==var_3)and(column_2[Counter]==0):
            column_1_6 = column_1_6 + 1
    return column_1_1,column_1_2,column_1_3,column_1_4,column_1_5,column_1_6

                                    #Additional_Entropy
def Additional_Entropy(My_Column):
    array,Number = np.unique(My_Column,return_counts = True)
    Additional_Entropy = np.sum([(-Number[i]/np.sum(Number))*np.log2(Number[i]/np.sum(Number)) for i in range(len(array))])
    return Additional_Entropy

                                    #predict
def predict(enquiry,tree,default = 1):
    for key in list(enquiry.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][enquiry[key]] 
            except:
                return default
            result = tree[key][enquiry[key]]
            if isinstance(result,dict):
                return predict(enquiry,result)
            else:
                return result

                                    #Check_List_2_1
def Check_List_2_1(dictionaryObject, name_0, name_1, var_1, var_2):
    column_1 = dictionaryObject[name_1]
    column_2 = dictionaryObject[name_0]
    column_1_1 = column_1_2 = column_1_3 = column_1_4 = 0
    for Counter in column_1:
        if (column_1[Counter]==var_1)and(column_2[Counter]==1):
            column_1_1 = column_1_1 + 1
        elif (column_1[Counter]==var_1)and(column_2[Counter]==0):
            column_1_2 = column_1_2 + 1
        elif (column_1[Counter]==var_2)and(column_2[Counter]==1):
            column_1_3 = column_1_3 + 1
        elif (column_1[Counter]==var_2)and(column_2[Counter]==0):
            column_1_4 = column_1_4 + 1
    return column_1_1,column_1_2,column_1_3,column_1_4

                                     #Main_Function
def Main_Function(My_Data,originaldata,Attribute,D1,D2 = 3,Name_2="Recidivism - Return to Prison numeric",parent_node_class = None):
    dataset = pd.read_csv('prison_dataset.csv',names=['Fiscal Year Released','Recidivism Reporting Year','Race - Ethnicity','Age At Release','Convicting Offense Classification','Convicting Offense Type','Convicting Offense Subtype','Main Supervising District','Release Type','Part of Target Population','Recidivism - Return to Prison numeric'],header = 1)
    dataset = shuffle(dataset)
    if(D1 == D2):
        return parent_node_class 
    if(D1 != D2):
        D1 += 1  
        parent_node_class = np.unique(My_Data[Name_2])[np.argmax(np.unique(My_Data[Name_2],return_counts=True)[1])]
        Index = [Gain_4(My_Data,feature,Name_2) for feature in Attribute] #Return the information gain values for the Attribute in the dataset
        Main_Index = np.argmax(Index)
        Main_Attribute = Attribute[Main_Index]
        My_Decision_Tree = {Main_Attribute:{}}
        Attribute = [i for i in Attribute if i != Main_Attribute]
        for amount in np.unique(My_Data[Main_Attribute]):
            amount = amount
            D3 = My_Data.where(My_Data[Main_Attribute] == amount).dropna()
            D4 = Main_Function(D3,dataset,Attribute,D1,D2,Name_2,parent_node_class)
            My_Decision_Tree[Main_Attribute][amount] = D4  
        return(My_Decision_Tree)

                                    #Check_List_3_2
def Check_List_3_2(dictionaryObject, name_0, name_1, var_1, var_2, var_3, name_2, var_4):
    column_1 = dictionaryObject[name_1]
    column_2 = dictionaryObject[name_0]
    column_3 = dictionaryObject[name_2]
    column_1_1 = column_1_2 = column_1_3 = column_1_4 = column_1_5 = column_1_6 = 0
    for Counter in column_1:
        if column_3[Counter]==var_4:
            if (column_1[Counter]==var_1)and(column_2[Counter]==1):
                column_1_1 = column_1_1 + 1
            elif (column_1[Counter]==var_1)and(column_2[Counter]==0):
                column_1_2 = column_1_2 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==1):
                column_1_3 = column_1_3 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==0):
                column_1_4 = column_1_4 + 1
            elif (column_1[Counter]==var_3)and(column_2[Counter]==1):
                column_1_5 = column_1_5 + 1
            elif (column_1[Counter]==var_3)and(column_2[Counter]==0):
                column_1_6 = column_1_6 + 1
    return column_1_1,column_1_2,column_1_3,column_1_4,column_1_5,column_1_6

                                    #Evaluate_Data_Accuracy
def Evaluate_Data_Accuracy(data,tree):
    name1 = 'Recidivism - Return to Prison numeric'
    name2 = "forestall"
    enquiry = data.iloc[:,:-1].to_dict(orient = "records")
    forestall = pd.DataFrame(columns=[name2]) 
    for i in range(len(data)):
        forestall.loc[i,name2] = predict(enquiry[i],tree,1.0) 
    print('Accuracy =  ',(np.sum(forestall[name2] == data[name1])/len(data))*100)

                                    #Evaluate_Data_Confusion_Matrix
def Evaluate_Data_Confusion_Matrix(data,tree):
    name1 = 'Recidivism - Return to Prison numeric'
    name2 = "forestall"
    enquiry = data.iloc[:,:-1].to_dict(orient = "records")
    forestall = pd.DataFrame(columns=[name2]) 
    for i in range(len(data)):
        forestall.loc[i,name2] = predict(enquiry[i],tree,1.0) 
    tp = np.sum((forestall[name2] == data[name1]) & (data[name1] == 1))
    tn = np.sum((forestall[name2] == data[name1]) & (data[name1] == 0))
    fp = np.sum((forestall[name2] != data[name1]) & (data[name1] == 1))
    fn = np.sum((forestall[name2] != data[name1]) & (data[name1] == 0))
    print('Confusion Matrix = ')
    print([tp,fp])
    print([fn,tn])

                                    #Check_List_2_2
def Check_List_2_2(dictionaryObject, name_0, name_1, var_1, var_2, name_2, var_4):
    column_1 = dictionaryObject[name_1]
    column_2 = dictionaryObject[name_0]
    column_3 = dictionaryObject[name_2]
    column_1_1 = column_1_2 = column_1_3 = column_1_4 = 0
    for Counter in column_1:
        if column_3[Counter]==var_4:
            if (column_1[Counter]==var_1)and(column_2[Counter]==1):
                column_1_1 = column_1_1 + 1
            elif (column_1[Counter]==var_1)and(column_2[Counter]==0):
                column_1_2 = column_1_2 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==1):
                column_1_3 = column_1_3 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==0):
                column_1_4 = column_1_4 + 1
    return column_1_1,column_1_2,column_1_3,column_1_4

                                    #Check_List_3_3
def Check_List_3_3(dictionaryObject, name_0, name_1, var_1, var_2, var_3, name_2, var_4, name_3, var_5):
    column_1 = dictionaryObject[name_1]
    column_2 = dictionaryObject[name_0]
    column_3 = dictionaryObject[name_2]
    column_4 = dictionaryObject[name_3]
    column_1_1 = column_1_2 = column_1_3 = column_1_4 = column_1_5 = column_1_6 = 0
    for Counter in column_1:
        if (column_3[Counter]==var_4)and(column_4[Counter]==var_5):
            if (column_1[Counter]==var_1)and(column_2[Counter]==1):
                column_1_1 = column_1_1 + 1
            elif (column_1[Counter]==var_1)and(column_2[Counter]==0):
                column_1_2 = column_1_2 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==1):
                column_1_3 = column_1_3 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==0):
                column_1_4 = column_1_4 + 1
            elif (column_1[Counter]==var_3)and(column_2[Counter]==1):
                column_1_5 = column_1_5 + 1
            elif (column_1[Counter]==var_3)and(column_2[Counter]==0):
                column_1_6 = column_1_6 + 1
    return column_1_1,column_1_2,column_1_3,column_1_4,column_1_5,column_1_6

                                    #Random_Sampling_2
def Random_Sampling_2(dataset):
    percent = 0.8
    Range = int(percent * dataset.shape[0])
    Train = dataset.iloc[:Range].reset_index(drop=True)
    Test = dataset.iloc[Range:].reset_index(drop=True)
    return Train,Test

                                    #Check_List_2_3
def Check_List_2_3(dictionaryObject, name_0, name_1, var_1, var_2, name_2, var_4, name_3, var_5):
    column_1 = dictionaryObject[name_1]
    column_2 = dictionaryObject[name_0]
    column_3 = dictionaryObject[name_2]
    column_4 = dictionaryObject[name_3]
    column_1_1 = column_1_2 = column_1_3 = column_1_4 = 0
    for Counter in column_1:
        if (column_3[Counter]==var_4)and(column_4[Counter]==var_5):
            if (column_1[Counter]==var_1)and(column_2[Counter]==1):
                column_1_1 = column_1_1 + 1
            elif (column_1[Counter]==var_1)and(column_2[Counter]==0):
                column_1_2 = column_1_2 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==1):
                column_1_3 = column_1_3 + 1
            elif (column_1[Counter]==var_2)and(column_2[Counter]==0):
                column_1_4 = column_1_4 + 1
    return column_1_1,column_1_2,column_1_3,column_1_4

                                    #Random_Sampling
def Random_Sampling(file ,Train_Percent):
    Test_Dict = {'Fiscal Year Released': {}, 'Recidivism Reporting Year': {}, 'Race - Ethnicity': {}, 'Age At Release': {}, 'Convicting Offense Classification': {}, 'Convicting Offense Type': {}, 'Convicting Offense Subtype': {}, 'Main Supervising District': {}, 'Release Type': {}, 'Part of Target Population': {}, 'Recidivism - Return to Prison numeric': {}}
    Train_Dict = {'Fiscal Year Released': {}, 'Recidivism Reporting Year': {}, 'Race - Ethnicity': {}, 'Age At Release': {}, 'Convicting Offense Classification': {}, 'Convicting Offense Type': {}, 'Convicting Offense Subtype': {}, 'Main Supervising District': {}, 'Release Type': {}, 'Part of Target Population': {}, 'Recidivism - Return to Prison numeric': {}}
    Test_Data = pd.DataFrame(file).values.tolist()                                          
    size = int((Train_Percent/100)*len(Test_Data))  
    Train_Data = [Test_Data.pop(random.randrange(len(Test_Data))) for _ in range(size)]    
    Counter = 0
    Number = 0
    for item in Test_Data:
        Counter = 0
        for name in item:
            if Counter == 0 :
                Test_Dict["Fiscal Year Released"][Number] = name 
            if Counter == 1 :
                Test_Dict["Recidivism Reporting Year"][Number] = name 
            if Counter == 2 :
                Test_Dict["Race - Ethnicity"][Number] = name 
            if Counter == 3 :
                Test_Dict["Age At Release"][Number] = name 
            if Counter == 4 :
                Test_Dict["Convicting Offense Classification"][Number] = name 
            if Counter == 5 :
                Test_Dict["Convicting Offense Type"][Number] = name 
            if Counter == 6 :
                Test_Dict["Convicting Offense Subtype"][Number] = name 
            if Counter == 7 :
                Test_Dict["Main Supervising District"][Number] = name 
            if Counter == 8 :
                Test_Dict["Release Type"][Number] = name 
            if Counter == 9 :
                Test_Dict["Part of Target Population"][Number] = name 
            if Counter == 10 :
                Test_Dict["Recidivism - Return to Prison numeric"][Number] = name 
            Counter = Counter + 1
            if Counter == 11 :
                Counter = 0
        Number = Number + 1
    Counter = 0
    Number = 0
    for item in Train_Data:
        Counter = 0
        for name in item:
            if Counter == 0 :
                Train_Dict["Fiscal Year Released"][Number] = name 
            if Counter == 1 :
                Train_Dict["Recidivism Reporting Year"][Number] = name 
            if Counter == 2 :
                Train_Dict["Race - Ethnicity"][Number] = name 
            if Counter == 3 :
                Train_Dict["Age At Release"][Number] = name 
            if Counter == 4 :
                Train_Dict["Convicting Offense Classification"][Number] = name 
            if Counter == 5 :
                Train_Dict["Convicting Offense Type"][Number] = name 
            if Counter == 6 :
                Train_Dict["Convicting Offense Subtype"][Number] = name 
            if Counter == 7 :
                Train_Dict["Main Supervising District"][Number] = name 
            if Counter == 8 :
                Train_Dict["Release Type"][Number] = name 
            if Counter == 9 :
                Train_Dict["Part of Target Population"][Number] = name 
            if Counter == 10 :
                Train_Dict["Recidivism - Return to Prison numeric"][Number] = name 
            Counter = Counter + 1
            if Counter == 11 :
                Counter = 0
        Number = Number + 1
    return Train_Dict,Test_Dict

                                    #Random_Forest
def Random_Forest(Train,Depth,Random_Attribute,trees):
    RandomForest = []
    for i in range(trees):
        RandomAttributeCounter = list(np.random.choice(np.arange(0,11), Random_Attribute, replace=False))
        Attribute = []
        for Counter2 in range(Random_Attribute):
            Attribute.append(Train.columns[RandomAttributeCounter[Counter2]])
        tree = Main_Function(Train,Train,Attribute,0,Depth)
        RandomForest.append(tree)
    return RandomForest

                                    #Indicate_Separate_Items
def Indicate_Separate_Items(dictionaryObject, name):
    column = dictionaryObject[name]
    List_Of_Items = []
    Separate_Items = []
    for item in column:
        List_Of_Items.append(column[item])
    for i in List_Of_Items:
        if i not in Separate_Items:
            Separate_Items.append(i)
    return Separate_Items

                                    #Indicate_Number_Of_Separate_Items
def Indicate_Number_Of_Separate_Items(dictionaryObject, name):
    column = dictionaryObject[name]
    List_Of_Items = []
    Separate_Items = []
    for item in column:
        List_Of_Items.append(column[item])
    for i in List_Of_Items:
        if i not in Separate_Items:
            Separate_Items.append(i)
    return len(Separate_Items)

                                     #Find_Maximum_Gain_1
def Find_Maximum_Gain_1(Main_Dict, name_0, name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_8, name_9, name_10):
    Attribute = [name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_8, name_9, name_10, name_0]
    Maximum_Gain_Name = []
    Maximum_Gain_Items = []
    Maximum_Gain_Labels = []
    G1 = G2 = G3 = G4 = G5 = G6 = G7 = G8 = G9 = G10 = 0
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_1) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_1)
        Label = Check_List_2_1(Main_Dict, name_0, name_1, var[0], var[1])
        G1 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_1) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_1)
        Label = Check_List_3_1(Main_Dict, name_0, name_1, var[0], var[1], var[2])
        G1 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_2) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_2)
        Label = Check_List_2_1(Main_Dict, name_0, name_2, var[0], var[1])
        G2 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_2) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_2)
        Label = Check_List_3_1(Main_Dict, name_0, name_2, var[0], var[1], var[2])
        G2 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])    
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_3) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_3)
        Label = Check_List_2_1(Main_Dict, name_0, name_3, var[0], var[1])
        G3 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_3) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_3)
        Label = Check_List_3_1(Main_Dict, name_0, name_3, var[0], var[1], var[2])
        G3 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_4) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_4)
        Label = Check_List_2_1(Main_Dict, name_0, name_4, var[0], var[1])
        G4 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_4) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_4)
        Label = Check_List_3_1(Main_Dict, name_0, name_4, var[0], var[1], var[2])
        G4 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_5) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_5)
        Label = Check_List_2_1(Main_Dict, name_0, name_5, var[0], var[1])
        G5 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_5) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_5)
        Label = Check_List_3_1(Main_Dict, name_0, name_5, var[0], var[1], var[2])
        G5 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_6) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_6)
        Label = Check_List_2_1(Main_Dict, name_0, name_6, var[0], var[1])
        G6 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_6) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_6)
        Label = Check_List_3_1(Main_Dict, name_0, name_6, var[0], var[1], var[2])
        G6 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_7) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_7)
        Label = Check_List_2_1(Main_Dict, name_0, name_7, var[0], var[1])
        G7 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_7) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_7)
        Label = Check_List_3_1(Main_Dict, name_0, name_7, var[0], var[1], var[2])
        G7 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_8) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_8)
        Label = Check_List_2_1(Main_Dict, name_0, name_8, var[0], var[1])
        G8 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_8) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_8)
        Label = Check_List_3_1(Main_Dict, name_0, name_8, var[0], var[1], var[2])
        G8 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_9) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_9)
        Label = Check_List_2_1(Main_Dict, name_0, name_9, var[0], var[1])
        G9 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_9) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_9)
        Label = Check_List_3_1(Main_Dict, name_0, name_9, var[0], var[1], var[2])
        G9 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_10) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_10)
        Label = Check_List_2_1(Main_Dict, name_0, name_10, var[0], var[1])
        G10 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_10) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_10)
        Label = Check_List_3_1(Main_Dict, name_0, name_10, var[0], var[1], var[2])
        G10 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    Max_Gain = max(G1,G2,G3,G4,G5,G6,G7,G8,G9,G10)
    if Max_Gain == G1 :
        Maximum_Gain_Name = Attribute[0]
    if Max_Gain == G2 :
        Maximum_Gain_Name = Attribute[1]
    if Max_Gain == G3 :
        Maximum_Gain_Name = Attribute[2]
    if Max_Gain == G4 :
        Maximum_Gain_Name = Attribute[3]
    if Max_Gain == G5 :
        Maximum_Gain_Name = Attribute[4]
    if Max_Gain == G6 :
        Maximum_Gain_Name = Attribute[5]
    if Max_Gain == G7 :
        Maximum_Gain_Name = Attribute[6]
    if Max_Gain == G8 :
        Maximum_Gain_Name = Attribute[7]
    if Max_Gain == G9 :
        Maximum_Gain_Name = Attribute[8]
    if Max_Gain == G10 :
        Maximum_Gain_Name = Attribute[9]
    Maximum_Gain_Items = Indicate_Separate_Items(Main_Dict, Maximum_Gain_Name)
    if Indicate_Number_Of_Separate_Items(Main_Dict, Maximum_Gain_Name) == 2:
        Maximum_Gain_Labels = Check_List_2_1(Main_Dict, name_0, Maximum_Gain_Name, Maximum_Gain_Items[0], Maximum_Gain_Items[1])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, Maximum_Gain_Name) == 3:
        Maximum_Gain_Labels = Check_List_3_1(Main_Dict, name_0, Maximum_Gain_Name, Maximum_Gain_Items[0], Maximum_Gain_Items[1], Maximum_Gain_Items[2])
    return Maximum_Gain_Name,Maximum_Gain_Items,Maximum_Gain_Labels

                                     #Evaluate_Data_Accuracy_1
def Evaluate_Data_Accuracy_1(data,My_Random_Forest):
    name1 = "predicted"
    name2 = 'Recidivism - Return to Prison numeric'
    enquiries = data.iloc[:,:-1].to_dict(orient = "records")
    forestall = pd.DataFrame(columns=[name1]) 
    for i in range(len(data)):
        outcome = np.zeros(len(My_Random_Forest))
        for Counter2 in range(len(My_Random_Forest)):
            outcome[Counter2] = predict(enquiries[i],My_Random_Forest[Counter2],1.0)
        if (np.sum(outcome == 1) > np.sum(outcome == 0)):
            ultimate = 1
        else:
            ultimate = 0
        forestall.loc[i,name1] = ultimate
    print('Accuracy = ',(np.sum(forestall[name1] == data[name2])/len(data))*100)


                                     #Evaluate_Data_Confusion_Matrix_1
def Evaluate_Data_Confusion_Matrix_1(data,My_Random_Forest):
    name1 = "predicted"
    name2 = 'Recidivism - Return to Prison numeric'
    enquiries = data.iloc[:,:-1].to_dict(orient = "records")
    forestall = pd.DataFrame(columns=[name1]) 
    for i in range(len(data)):
        outcome = np.zeros(len(My_Random_Forest))
        for Counter2 in range(len(My_Random_Forest)):
            outcome[Counter2] = predict(enquiries[i],My_Random_Forest[Counter2],1.0)
        if (np.sum(outcome == 1) > np.sum(outcome == 0)):
            ultimate = 1
        else:
            ultimate = 0
        forestall.loc[i,name1] = ultimate
    tp = np.sum((forestall[name1] == data[name2]) & (data[name2] == 1))
    tn = np.sum((forestall[name1] == data[name2]) & (data[name2] == 0))
    fp = np.sum((forestall[name1] != data[name2]) & (data[name2] == 1))
    fn = np.sum((forestall[name1] != data[name2]) & (data[name2] == 0))
    print('Confusion Matrix = ')
    print([tp,fp])
    print([fn,tn])

                                     #Find_Maximum_Gain_2
def Find_Maximum_Gain_2(Main_Dict, name_0, name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_8, name_9, Conditional_Name_1, Conditional_Value_1):
    Attribute = [name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_8, name_9, name_0]
    Maximum_Gain_Name = []
    Maximum_Gain_Items = []
    Maximum_Gain_Labels = []
    G1 = G2 = G3 = G4 = G5 = G6 = G7 = G8 = G9 = 0
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_1) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_1)
        Label = Check_List_2_2(Main_Dict, name_0, name_1, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G1 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_1) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_1)
        Label = Check_List_3_2(Main_Dict, name_0, name_1, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G1 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_2) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_2)
        Label = Check_List_2_2(Main_Dict, name_0, name_2, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G2 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_2) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_2)
        Label = Check_List_3_2(Main_Dict, name_0, name_2, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G2 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])    
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_3) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_3)
        Label = Check_List_2_2(Main_Dict, name_0, name_3, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G3 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_3) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_3)
        Label = Check_List_3_2(Main_Dict, name_0, name_3, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G3 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_4) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_4)
        Label = Check_List_2_2(Main_Dict, name_0, name_4, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G4 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_4) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_4)
        Label = Check_List_3_2(Main_Dict, name_0, name_4, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G4 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_5) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_5)
        Label = Check_List_2_2(Main_Dict, name_0, name_5, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G5 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_5) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_5)
        Label = Check_List_3_2(Main_Dict, name_0, name_5, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G5 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_6) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_6)
        Label = Check_List_2_2(Main_Dict, name_0, name_6, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G6 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_6) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_6)
        Label = Check_List_3_2(Main_Dict, name_0, name_6, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G6 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_7) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_7)
        Label = Check_List_2_2(Main_Dict, name_0, name_7, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G7 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_7) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_7)
        Label = Check_List_3_2(Main_Dict, name_0, name_7, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G7 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_8) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_8)
        Label = Check_List_2_2(Main_Dict, name_0, name_8, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G8 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_8) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_8)
        Label = Check_List_3_2(Main_Dict, name_0, name_8, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G8 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_9) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_9)
        Label = Check_List_2_2(Main_Dict, name_0, name_9, var[0], var[1], Conditional_Name_1, Conditional_Value_1)
        G9 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_9) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_9)
        Label = Check_List_3_2(Main_Dict, name_0, name_9, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1)
        G9 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    Max_Gain = max(G1,G2,G3,G4,G5,G6,G7,G8,G9)
    if Max_Gain == G1 :
        Maximum_Gain_Name = Attribute[0]
    if Max_Gain == G2 :
        Maximum_Gain_Name = Attribute[1]
    if Max_Gain == G3 :
        Maximum_Gain_Name = Attribute[2]
    if Max_Gain == G4 :
        Maximum_Gain_Name = Attribute[3]
    if Max_Gain == G5 :
        Maximum_Gain_Name = Attribute[4]
    if Max_Gain == G6 :
        Maximum_Gain_Name = Attribute[5]
    if Max_Gain == G7 :
        Maximum_Gain_Name = Attribute[6]
    if Max_Gain == G8 :
        Maximum_Gain_Name = Attribute[7]
    if Max_Gain == G9 :
        Maximum_Gain_Name = Attribute[8]
    Maximum_Gain_Items = Indicate_Separate_Items(Main_Dict, Maximum_Gain_Name)
    if Indicate_Number_Of_Separate_Items(Main_Dict, Maximum_Gain_Name) == 2:
        Maximum_Gain_Labels = Check_List_2_1(Main_Dict, name_0, Maximum_Gain_Name, Maximum_Gain_Items[0], Maximum_Gain_Items[1])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, Maximum_Gain_Name) == 3:
        Maximum_Gain_Labels = Check_List_3_1(Main_Dict, name_0, Maximum_Gain_Name, Maximum_Gain_Items[0], Maximum_Gain_Items[1], Maximum_Gain_Items[2])
    return Maximum_Gain_Name,Maximum_Gain_Items,Maximum_Gain_Labels

                                    #Find_Zero_Label
def Find_Zero_Label(First_Branch):
    Index = []
    Counter = Result = 0
    Zero_Label = []
    Zero_item = []
    First_Branch = list(First_Branch)
    First_Branch[2] = list(First_Branch[2])
    First_Branch[1] = list(First_Branch[1])
    for item in First_Branch[2]:
        if item == 0:
            Index.append(Counter)
        Counter = Counter + 1
    for index in Index:
        if index % 2 == 0:
            Zero_Label.append(First_Branch[2][index])
            Zero_Label.append(First_Branch[2][index+1])
            First_Branch[2].pop(index+1)
            First_Branch[2].pop(index)
            Zero_item.append(First_Branch[1][int(index/2)])
            First_Branch[1].pop(int(index/2))
            Result = 1
        elif index % 2 == 1:
            Zero_Label.append(First_Branch[2][index-1])
            Zero_Label.append(First_Branch[2][index])
            First_Branch[2].pop(index)
            First_Branch[2].pop(index-1)
            Zero_item.append(First_Branch[1][int((index-1)/2)])
            First_Branch[1].pop(int((index-1)/2))
            Result = 0
    return First_Branch,Zero_item,Result

                                     #Find_Maximum_Gain_3
def Find_Maximum_Gain_3(Main_Dict, name_0, name_1, name_2, name_3, name_4, name_5, name_6, name_7, Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2):
    Attribute = [name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_0]
    Maximum_Gain_Name = []
    Maximum_Gain_Items = []
    Maximum_Gain_Labels = []
    G1 = G2 = G3 = G4 = G5 = G6 = G7 = 0
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_1) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_1)
        Label = Check_List_2_3(Main_Dict, name_0, name_1, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G1 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_1) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_1)
        Label = Check_List_3_3(Main_Dict, name_0, name_1, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G1 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_2) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_2)
        Label = Check_List_2_3(Main_Dict, name_0, name_2, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G2 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_2) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_2)
        Label = Check_List_3_3(Main_Dict, name_0, name_2, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G2 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])    
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_3) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_3)
        Label = Check_List_2_3(Main_Dict, name_0, name_3, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G3 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_3) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_3)
        Label = Check_List_3_3(Main_Dict, name_0, name_3, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G3 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_4) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_4)
        Label = Check_List_2_3(Main_Dict, name_0, name_4, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G4 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_4) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_4)
        Label = Check_List_3_3(Main_Dict, name_0, name_4, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G4 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_5) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_5)
        Label = Check_List_2_3(Main_Dict, name_0, name_5, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G5 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_5) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_5)
        Label = Check_List_3_3(Main_Dict, name_0, name_5, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G5 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_6) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_6)
        Label = Check_List_2_3(Main_Dict, name_0, name_6, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G6 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_6) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_6)
        Label = Check_List_3_3(Main_Dict, name_0, name_6, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G6 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    if Indicate_Number_Of_Separate_Items(Main_Dict, name_7) == 2:
        var = Indicate_Separate_Items(Main_Dict, name_7)
        Label = Check_List_2_3(Main_Dict, name_0, name_7, var[0], var[1], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G7 = Information_Gain_2(Label[0], Label[1], Label[2], Label[3])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, name_7) == 3:
        var = Indicate_Separate_Items(Main_Dict, name_7)
        Label = Check_List_3_3(Main_Dict, name_0, name_7, var[0], var[1], var[2], Conditional_Name_1, Conditional_Value_1, Conditional_Name_2, Conditional_Value_2)
        G7 = Information_Gain_3(Label[0], Label[1], Label[2], Label[3], Label[4], Label[5])
    Max_Gain = max(G1,G2,G3,G4,G5,G6,G7)
    if Max_Gain == G1 :
        Maximum_Gain_Name = Attribute[0]
    if Max_Gain == G2 :
        Maximum_Gain_Name = Attribute[1]
    if Max_Gain == G3 :
        Maximum_Gain_Name = Attribute[2]
    if Max_Gain == G4 :
        Maximum_Gain_Name = Attribute[3]
    if Max_Gain == G5 :
        Maximum_Gain_Name = Attribute[4]
    if Max_Gain == G6 :
        Maximum_Gain_Name = Attribute[5]
    if Max_Gain == G7 :
        Maximum_Gain_Name = Attribute[6]
    Maximum_Gain_Items = Indicate_Separate_Items(Main_Dict, Maximum_Gain_Name)
    if Indicate_Number_Of_Separate_Items(Main_Dict, Maximum_Gain_Name) == 2:
        Maximum_Gain_Labels = Check_List_2_1(Main_Dict, name_0, Maximum_Gain_Name, Maximum_Gain_Items[0], Maximum_Gain_Items[1])
    elif Indicate_Number_Of_Separate_Items(Main_Dict, Maximum_Gain_Name) == 3:
        Maximum_Gain_Labels = Check_List_3_1(Main_Dict, name_0, Maximum_Gain_Name, Maximum_Gain_Items[0], Maximum_Gain_Items[1], Maximum_Gain_Items[2])
    return Maximum_Gain_Name,Maximum_Gain_Items,Maximum_Gain_Labels
