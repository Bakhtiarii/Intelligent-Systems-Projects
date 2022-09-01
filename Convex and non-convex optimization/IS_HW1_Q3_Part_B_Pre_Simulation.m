 Iris_setosa_Length = VarName1(1:50) ;
 Iris_setosa_width =  VarName2(1:50) ;
 Iris_versicolor_Length = VarName1(51:100) ;
 Iris_versicolor_width =  VarName2(51:100) ;
 Iris_virginica_Length =  VarName1(101:150) ;
 Iris_virginica_width =   VarName2(101:150) ;
 hold on                                            %Plot
 plot(Iris_setosa_Length,Iris_setosa_width,'r*');
 plot(Iris_versicolor_Length,Iris_versicolor_width,'b*');
 plot(Iris_virginica_Length,Iris_virginica_width,'g*');
 hold off
 xlabel('Sepal Length') ;
 ylabel('Sepal Width') ;