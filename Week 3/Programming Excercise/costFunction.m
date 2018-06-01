function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
constant = 1/m;
n = size(theta);
 
for i = 1:m
	t(i)=0;
	for j = 1:n
		t(i)= t(i) + theta(j)*X(i,j);
	end
	t1(i) =sigmoid(t(i));
end

t4=0;

for i=1:m
	t2 = -1 * (y(i,1)*log(t1(i)));
	t3 = (1-y(i,1))*(log(1-t1(i)));
	t4 = t4 + (t2-t3);
end

J = constant*t4;

for j=1:n
	grad(j) = 0;
	for k=1:m
		grad(j) = grad(j) + ((t1(k)-y(k,1))*X(k,j));
	end
	grad(j)= constant*grad(j);
end


% =============================================================

end

