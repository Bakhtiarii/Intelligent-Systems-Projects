import csv
                                    # Conditional_Probability
def Conditional_Probability(rows, Number):
    List = []
    for Object in rows:
        List.append(Object[Number])
    List = list(dict.fromkeys(List))
    Count_e = [0] * len(List)
    Count_p = [0] * len(List)
    Number_e = 0
    Number_p = 0
    Index = 0
    for Value in rows:
        Index = List.index(Value[Number])
        if Value[0] == 'e':
            Number_e = Number_e + 1
            Count_e[Index] = Count_e[Index] + 1
        if Value[0] == 'p':
            Number_p = Number_p + 1
            Count_p[Index] = Count_p[Index] + 1
    Count_e = [((x+1)/(Number_e+len(Count_e))) for x in Count_e]
    Count_p = [((y+1)/(Number_p+len(Count_p))) for y in Count_p]
    List.pop(0)
    Count_e.pop(0)
    Count_p.pop(0)
    return Count_e,Count_p,List

File_Train = open('Mushroom_Train.csv')
File_Test = open('Mushroom_Test.csv')

Csvreader_Train = csv.reader(File_Train)
Csvreader_Test = csv.reader(File_Test)

Header_Test = []
Header_Test = next(Csvreader_Test)

Rows_Train = []
for Row_Train in Csvreader_Train:
        Rows_Train.append(Row_Train)
Rows_Test = []
for Row_Test in Csvreader_Test:
        Rows_Test.append(Row_Test)

prob_e = 0                          
prob_p = 0

for item in Rows_Train:
    if item[0] == 'e':
        prob_e = prob_e + 1
    if item[0] == 'p':
        prob_p = prob_p + 1  
                                     # Priors probability
prob_e = prob_e/len(Rows_Train)     
prob_p = prob_p/len(Rows_Train)

Index = 0
Conditional_Prob_E = 1
Conditional_Prob_P = 1
Data_Class = 'Test'
T_e = 0
T_p = 0
F_e = 0
F_p = 0

for Test_Data in Rows_Test :
    for Counter in range(len(Header_Test)):
        if Counter != 0 :
            [Count_e,Count_p,List] = Conditional_Probability(Rows_Test, Counter)
            if Test_Data[Counter] in List :
                Index = List.index(Test_Data[Counter])
                Conditional_Prob_E = Conditional_Prob_E * Count_e[Index]
                Conditional_Prob_P = Conditional_Prob_P * Count_p[Index]

    if prob_e*Conditional_Prob_E > prob_p*Conditional_Prob_P:
        Data_Class = 'e'
    if prob_e*Conditional_Prob_E < prob_p*Conditional_Prob_P:
        Data_Class = 'p'
                                     # Confusion Matrix
    if (Data_Class == 'e')&(Test_Data[0]=='e'):
        T_e = T_e + 1
    if (Data_Class == 'e')&(Test_Data[0]=='p'):
        F_e = F_e + 1
    if (Data_Class == 'p')&(Test_Data[0]=='p'):
        T_p = T_p + 1
    if (Data_Class == 'p')&(Test_Data[0]=='e'):
        F_p = F_p + 1
    
    Conditional_Prob_E = 1
    Conditional_Prob_P = 1

Accuracy = (( T_e + T_p ) / ( T_e + T_p + F_e + F_p )) * 100
print('Accuracy is :')
print(Accuracy,'%')
print('\n')
print('Confusion Matrix is :')
print([T_e,F_e])
print([F_p,T_p])
