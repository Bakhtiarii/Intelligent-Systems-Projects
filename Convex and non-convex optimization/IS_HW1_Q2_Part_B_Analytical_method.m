syms f(x1,x2) Df(x1,x2) X_k(a) X1_k(a) X2_k(a) g(a) Dg(a)
f(x1,x2) = (x1*x1) - (10*x2*cos(0.2*pi*x1)) + (x2*x2) - (15*x1*cos(0.4*pi*x2)) ;     %Define F
Df(x1,x2) = gradient(f(x1,x2), [x1, x2]) ;                                           %Define F gradient
Df(x1,x2) = Df(x1,x2).' ;
X_new = [0 , 0] ;          X_old = [2 , 2] ;     Start_Point = X_new ;               %Start point
Error = 0.2 ;              Threshold = 0.1 ;                                         %Error function constraint
Temp = [ , ] ;             Temp_X = [ , ] ;
alpha = 0 ;                Number_Of_Iteration = 0 ;                                 %Step size

while Error > Threshold
    Number_Of_Iteration = Number_Of_Iteration + 1 ;
    P = -Df(X_new(1),X_new(2)) ; 
    X_k(a) = X_new + a.*P ; 
    Temp = X_k(a);
    X1_k(a) = Temp(1) ; 
    X2_k(a) = Temp(2) ; 
    g(a) = f(X1_k(a),X2_k(a)) ;                                                       %Define g(a)
    Dg(a) = gradient(g(a), [a]) ;                                                     %Define dg(a)/da
    eqn = Dg(a) == 0 ;                                                                %Find Optimal a
    alpha = vpasolve(eqn,a) ;
    Temp_X = X_new ;
    X_new = X_k(alpha) ;
    X_old = Temp_X ; 
    Error = abs( norm(X_new) - norm(X_old) ) ;                                        %Update Loop constraint 
end

disp(['alpha is : ( ' num2str(double(alpha)) ' )   and number of iteration is : ( ' num2str(double(Number_Of_Iteration))  ' )   and the Start point is : ( x_1 = ' num2str(double(Start_Point(1))) '   x_2 = ' num2str(double(Start_Point(2))) ' )'  ]) ;
