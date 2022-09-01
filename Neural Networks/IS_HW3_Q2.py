import numpy as np 
import math
import statistics
from Function import Random_Sampling , Normalization


X = np.random.uniform(0,2*math.pi,10000)                         # 0 < X < 2*PI
Y = np.random.uniform(0,2*math.pi,10000)                         # 0 < Y < 2*PI
np.random.shuffle(X)
np.random.shuffle(Y)
f_Label = np.random.uniform(0,0,10000)
for Counter in range(0,10000):
    f_Label[Counter] = math.sin(X[Counter]+Y[Counter])           # f(X,Y) = sin(X+Y)
x_Train,y_Train,label_Train,x_Test,y_Test,label_Test,x_validation,y_validation,label_validation = Random_Sampling(f_Label , X , Y)


                                                                 # Normalize data
Standard_deviation_x = statistics.stdev(x_Train)
Mean_x = statistics.mean(x_Train)
Standard_deviation_y = statistics.stdev(y_Train)
Mean_y = statistics.mean(y_Train)

x_Train = Normalization(x_Train , Mean_x , Standard_deviation_x)
y_Train = Normalization(y_Train , Mean_y , Standard_deviation_y)

x_Test = Normalization(x_Test , Mean_x , Standard_deviation_x)
y_Test = Normalization(y_Test , Mean_y , Standard_deviation_y)

x_validation = Normalization(x_validation , Mean_x , Standard_deviation_x)
y_validation = Normalization(y_validation , Mean_y , Standard_deviation_y)
