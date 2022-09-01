%% Part A
people = 100 ;
Birthday = [] ;
Same_Birthday = 0 ;
Len = 0 ;
Prob = [] ;
for n = 1:people
    Same_Birthday = 0 ;
    for Number_Of_Tests = 1:10000
        Birthday = randi([1 365],1,n) ;
        [au,ia,ic] = unique(Birthday,'stable') ;
        Len = size(au) ;
        if Len(2) ~= size(Birthday)
            Same_Birthday = Same_Birthday + 1 ;
        end
    end
    Prob(end+1) = Same_Birthday / 10000 ;
end

number = 1:100 ;
plot(number , Prob) ;

