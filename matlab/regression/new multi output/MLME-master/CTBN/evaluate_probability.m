%T is the tree (orgnaized in breadth-first fashion), Y is the assignment to compute it probability
function [ log_prob ] = evaluate_probability( T,  Y )

log_prob=0;
for i=1:length(T)
    y_val=Y(T{i}.node);
    if(isempty(T{i}.parent))
        log_prob=log_prob+T{i}.log_potential(y_val+1);
    else
        y_parent_val=Y(T{i}.parent);
        log_prob=log_prob+T{i}.log_potential( y_parent_val+1, y_val+1);
    end
end
