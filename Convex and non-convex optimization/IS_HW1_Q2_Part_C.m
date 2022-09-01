syms f(x1,x2) 
f(x1,x2) = (x1*x1) - (10*x2*cos(0.2*pi*x1)) + (x2*x2) - (15*x1*cos(0.4*pi*x2)) ;     %Define F
X_min = [0,0] ;       F_min = f(X_min(1),X_min(2)) ;
X_new = [ , ] ;       Counter = 0 ;
X_old = [1,1] ;       X_Random = [ , ] ;       Start_Point = X_old ;                 %Set start point
Temperature = 1000 ;  Temperature_rate = 1 ;   T = [Temperature , Temperature] ;     %Set the temperature
 
while Counter < 1000
    X_Random  = normrnd(X_old , T)/100;
    X_new = X_old + X_Random ;
    if rand() <= min(1 , exp((f(X_old(1),X_old(2)) - f(X_new(1),X_new(2)))/Temperature))
        X_old = X_new ;
    end
    if rand() > min(1 , exp((f(X_old(1),X_old(2)) - f(X_new(1),X_new(2)))/Temperature))
        X_old = X_old ;                                                              %do nothing
    end
    if f(X_old(1),X_old(2)) < F_min
        X_min = X_old ;
        F_min = f(X_min(1),X_min(2)) ;
    end
    Temperature = Temperature - Temperature_rate ;                                   %update temperature
    T = [Temperature , Temperature] ;
    Counter = Counter + 1 ;
end

disp(['Optimal F is : ( ' num2str(double(F_min)) ' ) and Optimal X is : ( ' num2str(double(X_min)) ' ) and Start point is : ( ' num2str(double(Start_Point)) ' )']) ;


