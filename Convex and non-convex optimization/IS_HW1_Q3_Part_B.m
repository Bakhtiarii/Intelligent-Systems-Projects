 Iris_setosa_Length = VarName1(1:50) ;         %Setosa Length
 Iris_setosa_width =  VarName2(1:50) ;         %Setosa Width
 Iris_versicolor_Length = VarName1(51:100) ;   %Versicolor Length
 Iris_versicolor_width =  VarName2(51:100) ;   %Versicolor Width
 Iris_virginica_Length =  VarName1(101:150) ;  %Virginica Length
 Iris_virginica_width =   VarName2(101:150) ;  %Virginica Width
 S = [Iris_setosa_Length(1) , Iris_setosa_width(1) , 1] ;
 Temp = [ , ] ;      Summ = [ ] ;        Temp_Sum = [ ];
 Counter_1 = 2 ;     Counter_2 = 1 ;     Counter_3 = 1 ;   Counter_4 = 1 ; 
                                               %Create S Vectors
 while Counter_1 < 151
     Temp = [VarName1(Counter_1) , VarName2(Counter_1) , 1] ;
     S = [S ; Temp] ;
     Counter_1 = Counter_1 + 1 ;
 end
                                               %Create matrix Summ
 while Counter_2 < 151
     Counter_3 = 1 ;
     while Counter_3 < 151
         Temp_Sum(end+1) = sum(S(Counter_2,:).*S(Counter_3,:)) ;
         Counter_3 = Counter_3 + 1 ;
     end
     Summ = [Summ ; Temp_Sum] ;
     Temp_Sum = [ ];
     Counter_2 = Counter_2 + 1 ;
 end
 
Y_positive = 1.+zeros(1,50) ;                   % (+1) Label
Y_negative = -1.+zeros(1,50) ;                  % (-1) Label
Y_1 = [Y_negative Y_positive Y_positive] ;
Y_2 = [Y_positive Y_negative Y_positive] ;
Y_3 = [Y_positive Y_positive Y_negative] ;
A_1 = Y_1*inv(Summ) ;
A_2 = Y_2*inv(Summ) ;
A_3 = Y_3*inv(Summ) ;
W = [0,0,0] ;
                                                %Create matrix W
while Counter_4 < 151
    W = W + (A_1(Counter_4).*S(Counter_4,:)) ;
    Counter_4 = Counter_4 + 1 ;
end
                                                % Y = WX + b
f = @(x) ((W(1)/W(2))*x)+(-W(3)/W(2)-3);
hold on
 plot(Iris_setosa_Length,Iris_setosa_width,'r*');
 plot(Iris_versicolor_Length,Iris_versicolor_width,'b*');
 plot(Iris_virginica_Length,Iris_virginica_width,'g*');
 fplot( f ) ; 
 xline(0) ;
 yline(0) ;
 xlabel('Sepal Length') ;
 ylabel('Sepal Width') ;
 title('Iris setosa VS Rest') ;
 hold off
