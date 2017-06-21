% function [outA, outB] = LeastSquareLDA(inX, inY, inR)

function [outA,outB] = LeastSquareLDA(inX, inY, inR)
% Solve min_{A,B}  {||Y-X'*A*B||_F^2
% input: inX: d by n data matrix
%        inY: n by k label matrix
%        inR: the low rank parameters
% output: outA: d by inR
%         outB: inR by k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xiao Cai, Chris Ding, Feiping Nie, Heng Huang. 
% On The Equivalent of Low-Rank Linear Regressions and Linear Discriminant Analysis Based Regressions. 
% The 19th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2013.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inX=inX';
% check 
[d, n1] = size(inX);
% if( d > n1 )
%     error('the feature dim is larger than the data dim !');
% end
[n2, k] = size(inY);
if((n1 ~= n2) || (k == 1))
    error('The size of the input is not compatible !');
else
    n = n1;
    clear n1 n2;
end
% normalize the label indicator matrix first
num_per_class = zeros(k, 1);
Y_n = inY;
for i = 1:k
    idx{i} = find(inY(:,i) == 1);
    num_per_class(i) = length(idx{i});
    Y_n(idx{i}, i) = 1/sqrt(num_per_class(i));
end
data_rank = rank(inX*inX');
fprintf('The rank of XX_tran is %d\n', data_rank);
% top eigenvectors of inv(St)*Sb
St = inX*inX';
Sb = inX*Y_n*Y_n'*inX';
[V, S] = eig(pinv(St)*Sb);
[s_sorted, idx_sorted] = sort(diag(S), 'descend');
outA = V(:,idx_sorted(1:inR));
% B = inv(A'*X*X'*A)*A'*inX*Y_n
outB = (outA'*inX*inX'*outA)\outA'*inX*Y_n;


end









