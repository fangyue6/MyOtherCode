function [ T2 ] = organize_tree_BFS( T )

%get all roots
idx_roots=[];
for i=1:length(T)
    if(isempty(T{i}.parent))
        idx_roots=[idx_roots i];
    end
end

idx=1;
for q=1:length(idx_roots)
    T2{idx}=T{idx_roots(q)};
    list=T2{idx}.children;
    
    idx=idx+1;
    
    while(~isempty(list))
        %take out the first element
        n=list(1);
        list=list(2:end);
        i=search_tree(n,T);
        T2{idx}=T{i};
        list=[list T2{idx}.children];
        idx=idx+1;

    end
end
