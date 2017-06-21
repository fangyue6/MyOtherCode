function [ Yb, membership ] = convert_m2b( Ym )
[~,k] = size(Ym);
membership = [];

j = 0;
for i = 1:k
    if length(unique(Ym(:,i))) > 2  % multi-class
        for l = 0:length(unique(Ym(:,i)))-1
            j = j+1;
            Yb(:,j) = (Ym(:,i) == l);
            membership = [membership i];
        end
        
    else                            % binary class
        j = j+1;
        Yb(:,j) = Ym(:,i);
        membership = [membership i];
    end
end

end %end-of-function convert_m2b()