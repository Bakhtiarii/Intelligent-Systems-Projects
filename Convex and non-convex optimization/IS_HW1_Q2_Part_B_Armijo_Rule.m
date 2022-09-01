syms f(x1,x2) Df(x1,x2) 
f(x1,x2) = (x1*x1) - (10*x2*cos(0.2*pi*x1)) + (x2*x2) - (15*x1*cos(0.4*pi*x2)) ;     %Define F
Df(x1,x2) = gradient(f(x1,x2), [x1, x2]) ;                                           %Define F gradient
Df(x1,x2) = Df(x1,x2).' ;
alpha = 10 ;              beta = 0.5 ;     c = 0.001 ;                               %Default Value 
X_old = [0 , 0] ;         F_new = 0 ;      Status = 1 ;                              %Start Point
while Status          
    X_new = X_old - (alpha.*Df(X_old(1),X_old(2))) ;
    F_new = f(X_new(1),X_new(2)) ;
    if F_new<=f(X_old(1),X_old(2)) - ((c*alpha).*(Df(X_old(1),X_old(2))*transpose(Df(X_old(1),X_old(2)))))%Check armijo condition
        Status = 0 ;
    end
    alpha = alpha * beta ;
end
disp(['alpha is :  ' num2str(double(alpha)) '  and Optimal f is : ' num2str(double(f(X_new(1),X_new(2))))  ' and the Start point is : ( x_1 = ' num2str(double(X_old(1))) '   x_2 = ' num2str(double(X_old(2))) ' )' ]) ;







