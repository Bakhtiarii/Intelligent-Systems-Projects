syms f(x1,x2) Df(x1,x2) X_k(a) X1_k(a) X2_k(a) g(a) Dg(a)
f(x1,x2) = 3*(x1*x1) + 12*x1 + 8*(x2*x2) + 8*x2 + 6*x1*x2 ;                %Define F
Df(x1,x2) = gradient(f(x1,x2), [x1, x2]) ;                                 %Define F gradient
Df(x1,x2) = Df(x1,x2).' ;
X_new = [1 , 1] ;          X_old = [2 , 2] ;                               %Start point
Error = 0.2 ;              Threshold = 0.01 ;                               %Error function constraint
Temp = [ , ] ;             Temp_X = [ , ] ;
alpha = 0 ;                                                                %Step size

while Error > Threshold
    P = -Df(X_new(1),X_new(2)) ; 
    X_k(a) = X_new + a.*P ; 
    Temp = X_k(a);
    X1_k(a) = Temp(1) ; 
    X2_k(a) = Temp(2) ; 
    g(a) = f(X1_k(a),X2_k(a)) ;                                            %Define g(a)
    Dg(a) = gradient(g(a), [a]) ;                                          %Define dg(a)/da
    eqn = Dg(a) == 0 ;                                                     %Find Optimal a
    alpha = solve(eqn) ;
    Temp_X = X_new ;
    X_new = X_k(alpha) ;
    X_old = Temp_X ; 
    Error = abs( norm(X_new) - norm(X_old) ) ;                             %Update Loop constraint 
end

disp(['Optimal X :   X_1 = ' num2str(double(X_new(1))) '   X_2 = ' num2str(double(X_new(2))) '   and alpha is : ' num2str(double(alpha))]) ;






