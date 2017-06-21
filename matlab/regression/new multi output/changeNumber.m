
function [a]=changeNumber(a)
% a=[0.1 0.2 0.1;
%     0.2 0.3 0.3]
[~,na] = size(a);
for i=1:na
    a(:,i) =changeNumber1(a(:,i));
end
end

function [a]=changeNumber1(a)
% a=[0.1;0.1;0.7;0.2;0.1;0.3;0.5;0.2;0.1];
b=length(a);
c= zeros(b,1);
temp=1;
for i=1:b
    if(c(i)==0)
        d=(a==a(i));
        a( d)=temp;
        c=c+double(d);
        temp=temp+1;
    end
end
end

