%check if node "from" can be reached from node "to": meaning there is a
%cycle
function [ err ] = check_cycle( T, to, from)

%do a BFS starting from node to, if we encounter node from, then there is a
%cycle
err=false;

[ idx ] = search_tree( to, T );

%BFS
list=T{idx}.children;

while(~isempty(list))
    %take out the first element
    n=list(1);
    if(n==from)
        err=true;
        return;
    end
    list=list(2:end);
    
    i=search_tree(n,T);
    
    list=[list T{i}.children];
    idx=idx+1;
end



end

