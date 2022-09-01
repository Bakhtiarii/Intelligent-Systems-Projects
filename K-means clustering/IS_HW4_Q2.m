load iris.dat
Number_Of_Clusters = 5 ;       % # Number Of Clusters
iteration = 150 ;               % # Number Of iteration
Index_Of_Center_Of_Clusters = randi([1 150],1,Number_Of_Clusters) ;     % Randomly choose center
Center_Of_Clusters = [] ; 
Index_Of_Data = [] ;

for Index = 1:Number_Of_Clusters
    Center_Of_Clusters(Index , 1) = iris(Index_Of_Center_Of_Clusters(Index) , 1) ;    % Index 1 Clusters
    Center_Of_Clusters(Index , 2) = iris(Index_Of_Center_Of_Clusters(Index) , 2) ;    % Index 2 Clusters
    Center_Of_Clusters(Index , 3) = iris(Index_Of_Center_Of_Clusters(Index) , 3) ;    % Index 3 Clusters
    Center_Of_Clusters(Index , 4) = iris(Index_Of_Center_Of_Clusters(Index) , 4) ;    % Index 4 Clusters
    Center_Of_Clusters(Index , 5) = iris(Index_Of_Center_Of_Clusters(Index) , 5) ;    % Index 5 Clusters
end


for Number_Of_iteration = 1:iteration

    Index_Of_Data = [] ;    
    Min_Distance = [] ;
    Min_Distance_Index = 0 ;
    Min_Value = 0 ;

    Data = 1 ;
    for Data = 1:150

        Min_Distance = [] ;
        Min_Distance_Index = 0 ;
        Min_Value = 0 ;

        Counter = 1 ;
        for Counter = 1:Number_Of_Clusters
            Min_Distance(end+1) = Distance(iris(Data , :) , Center_Of_Clusters(Counter , :)) ;
        end

        [Min_Value,Min_Distance_Index] = min(Min_Distance) ;
        Index_Of_Data(Min_Distance_Index , end+1) = Data ;

    end

    List = [] ;     s = [] ;     Len = 0 ;
    y1 = 0 ;    y2 = 0 ;    y3 = 0 ;    y4 = 0 ;    y5 = 0 ;
    item = 1 ;      s1 = [] ;       Len1 = 0 ;   
    
    for item = 1:Number_Of_Clusters
        
        s1 = [] ;       Len1 = 0 ; 
        s1 = size(Index_Of_Data) ; 
        Len1 = s1(1) ;
        
        if item <= Len1
            
            List = [] ;     s = [] ;     Len = 0 ;
            y1 = 0 ;    y2 = 0 ;    y3 = 0 ;    y4 = 0 ;    y5 = 0 ;
            List = Index_Of_Data(item , :) ;
            List(List==0) = [];
            s = size(List) ; 
            Len = s(2) ;
            i = 1 ;

            for i = 1:Len           
                y1 = y1 + iris(i,1) ;
                y2 = y2 + iris(i,2) ;
                y3 = y3 + iris(i,3) ;
                y4 = y4 + iris(i,4) ;
                y5 = y5 + iris(i,5) ;            
            end
                                                                % Update Center of Clusters                              
            if Len ~= 0             
                Center_Of_Clusters(item , 1) = y1 / Len ;
                Center_Of_Clusters(item , 2) = y2 / Len ;
                Center_Of_Clusters(item , 3) = y3 / Len ;
                Center_Of_Clusters(item , 4) = y4 / Len ;
                Center_Of_Clusters(item , 5) = y5 / Len ;
            end
        
        end

    end

end



% Internal similarity
function Distance = Internal_similarity(Data , Center)
    for j = 1:size(Data)
        Distance = Distance + norm(Data(j) - Center) ;    
    end         
end

% External similarity
function Distance = External similarity(Data , Centers)
    for k = 1:size(Centers)
        Distance = Distance + norm(Data - Centers(k)) ;    
    end         
end

% Euclidean norm
function D = Distance(x1 , x2)
    D = norm(x1-x2) ;                   
end




