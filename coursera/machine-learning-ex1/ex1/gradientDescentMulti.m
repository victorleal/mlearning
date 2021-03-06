function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
theta_len = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



    M = theta' .* X;
    M = sum(M,2);
    S = (M .- y);

    S = S .* X;

    %theta(1) = theta(1) - alpha*(sum(S(:,1),1)/m)
    %theta(2) = theta(2) - alpha*(sum(S(:,2),1)/m)
    %theta(3) = theta(3) - alpha*(sum(S(:,3),1)/m)

    theta = (theta' .- alpha/m * (sum(S,1)))';




    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
