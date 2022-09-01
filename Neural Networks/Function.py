import statistics
                                    #Random_Sampling
def Random_Sampling(Label , x , y):
    x_Train = x[1600:8000]
    y_Train = y[1600:8000]
    x_Test = x[8000:]
    y_Test = y[8000:]
    x_validation = x[:1600]
    y_validation = y[:1600]
    label_Train = Label[1600:8000]
    label_Test = Label[8000:]
    label_validation = Label[:1600]
    return x_Train,y_Train,label_Train,x_Test,y_Test,label_Test,x_validation,y_validation,label_validation

                                    #Normalization
def Normalization(data , Mean , Standard_deviation):
    for Counter in range(0,len(data)):
        data[Counter] = (data[Counter] - Mean) / Standard_deviation
    return data
