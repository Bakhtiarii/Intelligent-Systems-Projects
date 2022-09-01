import numpy as np
import pandas as pd
import random
from random import sample
from random import randrange
from metric_learn import LMNN
from sklearn.datasets import load_iris
from metric_learn import ITML
from Func_Q3 import euclidean_distance,Random_Sampling,Find_Distance,Find_K_index,Print_Accuracy_lmnn,Majority,K_Nearest_Train,Print_Accuracy_And_Confusion_Matrix,K_Nearest_Test

myfile = pd.read_csv('wine.csv')
File_Array = myfile.to_numpy()
[Train_Data,Test_Data] = Random_Sampling(myfile ,80)
X_test = Test_Data
Number_Of_Nearest_Neighbor = 5
X = File_Array[::][:,1:14]
Y = File_Array[::][:,0:1]
Y = Y[:,0:1].ravel()
Y_test = Y
X_test = X
lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X, Y)
lmnn.fit_transform(X_test, Y_test)
Real_And_Predicted_Labels = K_Nearest_Test(Train_Data,Test_Data,Number_Of_Nearest_Neighbor)
Print_Accuracy_lmnn(Real_And_Predicted_Labels,Number_Of_Nearest_Neighbor)












