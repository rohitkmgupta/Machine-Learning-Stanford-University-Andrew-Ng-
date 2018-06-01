function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%J =1/2m summation from i=1 to m ((theta(0) + theta(1)x1)-y1)^2
constant = 1/(2*m);
T=0;

for i=1:m 
    term1 = (theta(1,1)*X(i,1));
	term2 = (theta(2,1)*X(i,2));
	term3 = term1 + term2;
	term4 = term3-y(i,1);
	term5 = term4^2;
	T = T + term5;
end
 J = constant* T;

% =========================================================================

end
