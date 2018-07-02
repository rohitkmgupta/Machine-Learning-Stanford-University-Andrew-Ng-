function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

constant = (1/m);
constant1 = (2*m);
constant2 = (1/ (2*m));
constant3 = (lambda/ (2*m));
constant4= lambda/m;

tTheta = theta;
tTheta(1) = 0;

J = (constant2) * sum(((X * theta)-y).^2) + (constant3)*sum(tTheta.^2);
temp = X * theta;
error = temp - y;
grad = (constant) * (X' * error) + (constant4)*tTheta;





% =========================================================================

grad = grad(:);

end
