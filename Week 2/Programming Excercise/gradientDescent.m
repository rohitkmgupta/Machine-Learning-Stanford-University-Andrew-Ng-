function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

constant = 1/m;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
T1 = 0;
T2 = 0;
	for j = 1:m
		trm1 = theta(1,1)*X(j,1);
		trm2 = theta(2,1)*X(j,2);
		trm3 = trm1 + trm2;
		trm4 = trm3-y(j,1);
		trm5 = trm4*X(j,1);
		trm6 = trm5*X(j,2);
		T1 = T1 + trm5;
		T2 = T2 + trm6;
	end
	theta(1,1) = theta(1,1) - (alpha*constant*T1);
	theta(2,1) = theta(2,1) - (alpha*constant*T2);
	disp(computeCost(X,y,theta));
	
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
