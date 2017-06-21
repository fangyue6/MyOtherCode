function [A]=cosine_similarity(y)
[m n]=size(y);    % m=n, n=c
for j=1:n    % lie
    for i=1:n
       y1=y(:,j);
       y2=y(:,i);
        aa=sum(y1.*y2);
        bb=sqrt(sum(y1.*y1))*sqrt(sum(y2.*y2));
        A(i,j)=aa/bb;
        A(j,i)=aa/bb;
        
    end
end