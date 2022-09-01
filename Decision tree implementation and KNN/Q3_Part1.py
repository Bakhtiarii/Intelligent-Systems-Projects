import numpy as np
import pandas as pd
import random
from random import sample
from random import randrange
from Func_Q3 import euclidean_distance,Random_Sampling,Find_Distance,Find_K_index,Majority,K_Nearest_Train,Print_Accuracy_And_Confusion_Matrix,K_Nearest_Test

myfile = pd.read_csv('wine.csv')
myfile = myfile.values.tolist()
[Train_Data,Test_Data] = Random_Sampling(myfile ,80)
Number_Of_Nearest_Neighbor = 5
Real_And_Predicted_Labels = K_Nearest_Test(Train_Data,Test_Data,Number_Of_Nearest_Neighbor)
Print_Accuracy_And_Confusion_Matrix(Real_And_Predicted_Labels)













