function [pres,labels]=MIML_test(test_data,W,V,num_sub)
firstk=5;
n=size(test_data,1);
pres=-inf(n,size(W,2)/num_sub);
WV=V'*W;
for i=1:n
    xbag=test_data{i};
    for j=1:num_sub
        WVone=WV(:,j:num_sub:end);
        fs=max(xbag'*WVone,[],1);        
        pres(i,:)=max(pres(i,:),fs);
    end
end

[~,ord]=sort(pres(:,1:end-1),2,'descend');
top_idx=ord(:,1:firstk);
labels=-ones(n,size(W,2)/num_sub-1);
for i=1:n
    labels(i,top_idx(i,1))=1;
    labels(i,pres(i,1:end-1)>(pres(i,end)+1))=1;
end
