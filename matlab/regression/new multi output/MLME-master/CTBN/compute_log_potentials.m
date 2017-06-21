function [ T ] = compute_log_potentials( T, x, is_switching )

if ~exist('is_switching','var')
    is_switching = true;
end

for i = 1:length(T)
    if isempty(T{i}.parent)
        % independent node
        p = LR_predict(T{i}.weights, x);
        if(length(p)==1)
            T{i}.log_potential(1) = log(1-p);
            T{i}.log_potential(2) = log(p);
        else
            T{i}.log_potential(1) = log(p(1));
            T{i}.log_potential(2) = log(p(2));
            T{i}.log_potential(3) = log(p(3));
        end
        
    else
        % dependent node
        if ~is_switching
            % monolithic model (non-switching)
            for j = 1:2
                p = LR_predict(T{i}.weights, [x (j-1)]);
                
                if(length(p)==1)
                    T{i}.log_potential(j,1) = log(1-p);
                    T{i}.log_potential(j,2) = log(p);
                else
                    T{i}.log_potential(j,1) = log(p(1));
                    T{i}.log_potential(j,2) = log(p(2));
                    T{i}.log_potential(j,3) = log(p(3));
                end
            end
        else
            % switching model
            for j = 1:T{i}.card
                p = LR_predict(T{i}.weights{j}, x);
                if(length(p)==1)
                    T{i}.log_potential(j,1) = log(1-p);
                    T{i}.log_potential(j,2) = log(p); 
                else
                    T{i}.log_potential(j,1) = log(p(1));
                    T{i}.log_potential(j,2) = log(p(2));
                    T{i}.log_potential(j,3) = log(p(3));
                end
            end
            
        end%end-of-if(~is_switching)
    end%end-of-if(isempty(T{i}.parent))
end%end-of-for i=1:length(T)

