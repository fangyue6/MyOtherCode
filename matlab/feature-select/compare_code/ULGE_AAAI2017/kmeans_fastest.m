
function [label, iter_num, sumd, center, obj] = kmeans_fastest(X, k, label)
% X: dim*n matrix, each column is a data point

n = size(X,2);
last = 0;
if nargin < 3
label = ceil(k*rand(n,1));  % random initialization
end;

iter_num = 0;
while any(label ~= last)
    [dumb,dumb1,label] = unique(label);   % remove empty clusters
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    center = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute center of each cluster
    last = label;
    [dumb,label] = max(bsxfun(@minus,center'*X,0.5*sum(center.^2,1)'),[],1); % assign samples to the nearest centers
    label = label';
    iter_num = iter_num+1;
end


if nargout > 2
    for ii=1:k
        idxi = find(label==ii);
        Xi = X(:,idxi);
        ceni = mean(Xi,2);
        center(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
    end
    obj = sum(sumd);
end;

