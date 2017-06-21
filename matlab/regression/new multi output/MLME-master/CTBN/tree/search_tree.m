%search the tree for node n, return -1 if not found
function [ idx ] = search_tree( n, T )

%search the tree for the node
idx=-1;
for i=1:length(T)
    if(T{i}.node==n)
        idx=i;
        break;
    end
end


