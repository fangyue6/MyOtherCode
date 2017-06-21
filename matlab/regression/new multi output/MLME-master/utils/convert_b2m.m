function [ Ym ] = convert_b2m( Yb, membership )
k = length(unique(membership));
n = size(Yb,1);

Ym = [];
for i = 1:k
    Yi = Yb( :, membership==i );
    [~,c] = size(Yi);
    if c == 1   % binary class
        Ym(:,i) = Yi;
    else        % multi-class
        temp = zeros(n,1);
        for j = 1:c
            temp(Yi(:,j)==1) = j-1;
        end
        Ym(:,i) = temp;
    end
end

end %end-of-function convert_b2m()