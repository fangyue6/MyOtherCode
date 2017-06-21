function [ LL, avg_prob, LLi ] = compute_log_prob( Y_true, Y_prob, membership )

[ n, k ] = size(Y_true);

if nargin > 2    %if exist(membership)
    %convert Y_prob into cell
    if ~iscell(Y_prob)
        temp = cell(1,k);
        for i = 1:k
            temp{i} = Y_prob(:,membership==i);
        end
    end
    Y_prob = temp;
end

LLi = zeros(1,k);
LL = 0;
sum_prob = 0;

for i = 1:n
    ll = 0;
    for j = 1:k
        if ~iscell(Y_prob)  % binary case
            if Y_true(i, j)
                ll = ll + log(Y_prob(i, j));
                LLi(j) = LLi(j) + log(Y_prob(i, j));
            else
                ll = ll + log(1 - Y_prob(i, j));
                LLi(j) = LLi(j) + log(1 - Y_prob(i, j));
            end
        else
            if size(Y_prob{j},2) == 1   % binary case (after multi-class mod')
                if Y_true(i, j)
                    ll = ll + log(Y_prob{j}(i));
                    LLi(j) = LLi(j) + log(Y_prob{j}(i));
                else
                    ll = ll + log(1 - Y_prob{j}(i));
                    LLi(j) = LLi(j) + log(1 - Y_prob{j}(i));
                end
            else                        % multi-class case
                ll = ll + log(Y_prob{j}(i,Y_true(i,j)+1));
                LLi(j) = LLi(j) + log(Y_prob{j}(i,Y_true(i,j)+1));
            end
        end
        
        if(ll < -100)
            ll=-100;
        end
    end
    LL = LL + ll;
    sum_prob=sum_prob + exp(ll);
end
avg_prob=sum_prob/n;

end