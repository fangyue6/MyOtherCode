function [W,obj]= FRFS0(X,lambda,mu)

%X is the data matrix  of n*d , n is the number of samples and d is the
% dimension of data.
%lambad is the parameter, balance the loss function and regularizer.
%mu is largrange parameter 
[n,d] = size(X);
iter = 20;%50
rho = 1.01;

%initialize 
XX = X'*X;
W =  rand(d,d);
E = rand(n,d);
pet1 = rand(n,d);
pet2 = rand(d,d);
V = W;
inmu = 1/mu;
obj = zeros(1,iter);

    for i = 1:iter
%         fprintf('Update ==========(%d)\n',i);
        % Update E
%             Y = X*W-X+(1/mu)*pet1;
%             Y_2 = sqrt(diag(Y*Y'));
%             E =  repmat(max(Y_2-inmu,0).*Y_2,1,d).*Y;
        
            Y = X*W-X+inmu*pet1;
            for j = 1:n
                y = Y(j,:);
                la = sqrt(y*y')+1e-6;
                E(j,:) = max(la-inmu,0)/la*y;
            end
        % Update W
            tem = X'*X;
            W = inv(tem+lambda*eye(d))*(tem+X'*E-inmu*X'*pet1+lambda*V-lambda/mu*pet2);
            
        % Update V
            Z = W + inmu*pet2;
            temp = sum(Z.*Z,2);
            idx = (temp<2*inmu);
            V = Z;
            V(idx,:)=0;

        %Update pet
        pet1 = pet1 + mu*(X*W-X-E);
        pet2 = pet2 + mu*(W-V);   
        mu = min(10^10,rho*mu);
        
        %Loss function
        A = X*W-X-E+inmu*pet1;
        B = W-V+inmu*pet2;
        C = sum(sqrt(sum(E.*E,2)));
        D = sum(abs(sign(sum(V.*V,2))));
        
        l1(i) = mu/2 * (norm(A,'fro')^2);
        l2(i) = mu/2*lambda * (norm(B,'fro')^2);
        l3(i) = C;
        l4(i) = lambda*D;
        obj(1,i) =  l1(i) + l2(i) + l3(i) + l4(i);
        
        if iter > 1
            if obj(iter-1) > obj(iter) && obj(iter-1) - obj(iter) < 1e-7
                break
            end
        end
    end

    [~, indx] = sort(sum(W.*W,2),'descend');
    
end