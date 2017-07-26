function PX = lagrange_vec(X, Y)
% LAGRANGE_VEC vectorized version of LAGRANGE3.

PX = X(:,2) .* X(:,3) ./ ((X(:,1) - X(:,2)).*(X(:,1) - X(:,3))).*Y(:,1) + ...
     X(:,1) .* X(:,3) ./ ((X(:,2) - X(:,1)).*(X(:,2) - X(:,3))).*Y(:,2) + ...
     X(:,1) .* X(:,2) ./ ((X(:,3) - X(:,1)).*(X(:,3) - X(:,2))).*Y(:,3);
end