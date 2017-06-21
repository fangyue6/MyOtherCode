function W = LS21(X, Y, r, W0)
    [n, m] = size(X);
    if nargin < 4
        d = ones(m,1);
    else
        Wi = sqrt(sum(W0.^2,2)+eps);
        d = 0.5./(Wi);
    end;

maxiter = 10;
    if n < d % n^3
        XY = X' * Y;
        for iter= 1:maxiter
            rd = 1 ./ (r * d);
            Xrd = bsxfun(@times, X, rd');
            XrdX = Xrd * X';
            A = diag(rd) - Xrd' / (eye(n) + XrdX) * Xrd;
            W = A * XY;
            Wi = sqrt(sum(W.^2,2)+eps);
            d = 0.5./(Wi);
        end
    else % d^3
        XX = X' * X;
        XY = X' * Y;
        for iter= 1:maxiter
            W = (XX + r * diag(d)) \ XY;
            Wi = sqrt(sum(W.^2,2)+eps);
            d = 0.5./(Wi);
        end
    end
end