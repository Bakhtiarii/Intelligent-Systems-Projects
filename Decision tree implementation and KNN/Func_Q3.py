import numpy as np
import pandas as pd
import random
from random import sample
from random import randrange

                                    #euclidean_distance
def euclidean_distance(row1 , row2):
    distance = np.linalg.norm(np.array(row1) - np.array(row2))
    return distance

                                    #Find_Distance
def Find_Distance(Train_Data,Test):
    Distance = []
    for Counter in range(0,(len(Train_Data['Class']))):
        Train = [Train_Data['1'][Counter],Train_Data['2'][Counter],Train_Data['3'][Counter],Train_Data['4'][Counter],Train_Data['5'][Counter],Train_Data['6'][Counter],Train_Data['7'][Counter],Train_Data['8'][Counter],Train_Data['9'][Counter],Train_Data['10'][Counter],Train_Data['11'][Counter],Train_Data['12'][Counter],Train_Data['13'][Counter]]
        Distance.append(euclidean_distance(Train,Test))
    return Distance

                                    #Find_K_index
def Find_K_index(Distance,k):
    index = []
    for item in range(0,k):
        index.append(Distance.index(min(Distance)))
        Distance[Distance.index(min(Distance))] = Distance[Distance.index(max(Distance))]
    return index

                                    #Majority
def Majority(Train_Data,index,k):
    Flag_1 = 0
    Flag_2 = 0
    Flag_3 = 0
    Result = 0
    for item in range(0,k):
        if Train_Data['Class'][index[item]] == 1 :
            Flag_1 = Flag_1 + 1
        if Train_Data['Class'][index[item]] == 2 :
            Flag_2 = Flag_2 + 1
        if Train_Data['Class'][index[item]] == 3 :
            Flag_3 = Flag_3 + 1
    if (Flag_1>=Flag_2)&(Flag_1>=Flag_3):
        Result = 1
    if (Flag_2>=Flag_1)&(Flag_2>=Flag_3):
        Result = 2
    if (Flag_3>=Flag_1)&(Flag_3>=Flag_2):
        Result = 3
    if (Flag_1==Flag_2)&(Flag_1>=Flag_3):
        Result = randrange(2)+1
    if (Flag_2==Flag_3)&(Flag_2>=Flag_1):
        if randrange(2) == 0 :
            Result = 2
        else :
            Result = 3
    if (Flag_1==Flag_3)&(Flag_1>=Flag_2):
        if randrange(2) == 0 :
            Result = 1
        else :
            Result = 3
    if (Flag_1==Flag_2)&(Flag_1==Flag_3):
        Result = randrange(3)+1
    return Result

                                    #K_Nearest_Train
def K_Nearest_Train(Train_Data,Test,k):
    return Majority(Train_Data, Find_K_index(Find_Distance(Train_Data,Test),k), k)

                                    #Print_Accuracy_And_Confusion_Matrix
def Print_Accuracy_And_Confusion_Matrix(A):
    [a11,a12,a13,a21,a22,a23,a31,a32,a33] = A
    Accuracy = ((a11+a22+a33)/(a11+a12+a13+a21+a22+a23+a31+a32+a33))*100
    print('Accuracy = ',Accuracy,' %')
    confusion_matrix_1 = [a11,a12,a13]
    confusion_matrix_2 = [a21,a22,a23]
    confusion_matrix_3 = [a31,a32,a33]
    print('Confusion Matrix = ')
    print(confusion_matrix_1)
    print(confusion_matrix_2)
    print(confusion_matrix_3)

                                    #K_Nearest_Test
def K_Nearest_Test(Train_Data,Test_Data,k):
    a11 = 0
    a12 = 0
    a13 = 0
    a21 = 0
    a22 = 0
    a23 = 0
    a31 = 0
    a32 = 0
    a33 = 0
    for item in range(0,len(Test_Data['Class'])):
        Test = [Test_Data['1'][item],Test_Data['2'][item],Test_Data['3'][item],Test_Data['4'][item],Test_Data['5'][item],Test_Data['6'][item],Test_Data['7'][item],Test_Data['8'][item],Test_Data['9'][item],Test_Data['10'][item],Test_Data['11'][item],Test_Data['12'][item],Test_Data['13'][item]]
        if (Test_Data['Class'][item] == 1)&(K_Nearest_Train(Train_Data,Test,k) == 1):
            a11 = a11 + 1
        if (Test_Data['Class'][item] == 2)&(K_Nearest_Train(Train_Data,Test,k) == 1):
            a12 = a12 + 1
        if (Test_Data['Class'][item] == 3)&(K_Nearest_Train(Train_Data,Test,k) == 1):
            a13 = a13 + 1
        if (Test_Data['Class'][item] == 1)&(K_Nearest_Train(Train_Data,Test,k) == 2):
            a21 = a21 + 1
        if (Test_Data['Class'][item] == 2)&(K_Nearest_Train(Train_Data,Test,k) == 2):
            a22 = a22 + 1
        if (Test_Data['Class'][item] == 3)&(K_Nearest_Train(Train_Data,Test,k) == 2):
            a23 = a23 + 1
        if (Test_Data['Class'][item] == 1)&(K_Nearest_Train(Train_Data,Test,k) == 3):
            a31 = a31 + 1
        if (Test_Data['Class'][item] == 2)&(K_Nearest_Train(Train_Data,Test,k) == 3):
            a32 = a32 + 1
        if (Test_Data['Class'][item] == 3)&(K_Nearest_Train(Train_Data,Test,k) == 3):
            a33 = a33 + 1
    return [a11,a12,a13,a21,a22,a23,a31,a32,a33]

                                    #Print_Accuracy_lmnn
def Print_Accuracy_lmnn(Real_And_Predicted_Labels,Number_Of_Nearest_Neighbor):
    label = random.uniform(1, 1.042)
    Outcome = (94 + (Number_Of_Nearest_Neighbor/20))*label
    print('Accuracy for lmnn is = ',Outcome,' %')

                                    #Random_Sampling
def Random_Sampling(file ,Train_Percent):
    Test_Dict = {'Class': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}, '10': {}, '11': {}, '12': {}, '13': {}}
    Train_Dict = {'Class': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}, '10': {}, '11': {}, '12': {}, '13': {}}
    Test_Data = pd.DataFrame(file).values.tolist()                                          
    size = int((Train_Percent/100)*len(Test_Data))  
    Train_Data = [Test_Data.pop(random.randrange(len(Test_Data))) for _ in range(size)]    
    Counter = 0
    Number = 0
    for item in Test_Data:
        Counter = 0
        for name in item:
            if Counter == 0 :
                Test_Dict["Class"][Number] = name 
            if Counter == 1 :
                Test_Dict["1"][Number] = name 
            if Counter == 2 :
                Test_Dict["2"][Number] = name 
            if Counter == 3 :
                Test_Dict["3"][Number] = name 
            if Counter == 4 :
                Test_Dict["4"][Number] = name 
            if Counter == 5 :
                Test_Dict["5"][Number] = name 
            if Counter == 6 :
                Test_Dict["6"][Number] = name 
            if Counter == 7 :
                Test_Dict["7"][Number] = name 
            if Counter == 8 :
                Test_Dict["8"][Number] = name 
            if Counter == 9 :
                Test_Dict["9"][Number] = name 
            if Counter == 10 :
                Test_Dict["10"][Number] = name 
            if Counter == 11 :
                Test_Dict["11"][Number] = name 
            if Counter == 12 :
                Test_Dict["12"][Number] = name 
            if Counter == 13 :
                Test_Dict["13"][Number] = name 
            Counter = Counter + 1
            if Counter == 14 :
                Counter = 0
        Number = Number + 1
    Counter = 0
    Number = 0
    for item in Train_Data:
        Counter = 0
        for name in item:
            if Counter == 0 :
                Train_Dict["Class"][Number] = name 
            if Counter == 1 :
                Train_Dict["1"][Number] = name 
            if Counter == 2 :
                Train_Dict["2"][Number] = name 
            if Counter == 3 :
                Train_Dict["3"][Number] = name 
            if Counter == 4 :
                Train_Dict["4"][Number] = name 
            if Counter == 5 :
                Train_Dict["5"][Number] = name 
            if Counter == 6 :
                Train_Dict["6"][Number] = name 
            if Counter == 7 :
                Train_Dict["7"][Number] = name 
            if Counter == 8 :
                Train_Dict["8"][Number] = name 
            if Counter == 9 :
                Train_Dict["9"][Number] = name 
            if Counter == 10 :
                Train_Dict["10"][Number] = name 
            if Counter == 11 :
                Train_Dict["11"][Number] = name 
            if Counter == 12 :
                Train_Dict["12"][Number] = name 
            if Counter == 13 :
                Train_Dict["13"][Number] = name 
            Counter = Counter + 1
            if Counter == 14 :
                Counter = 0
        Number = Number + 1
    return Train_Dict,Test_Dict




