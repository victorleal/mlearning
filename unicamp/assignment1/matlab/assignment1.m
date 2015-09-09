%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('C:\Users\Victor Leal\Desktop\mlearning\assignment1\YearPredictionMSD.txt');
%data = load('/home/victor/YearPredictionMSD.txt');
a = data(1:10000, 2:31);
b = data(1:10000, 32:61);
c = data(1:10000, 62:91);

y = data(1:10000, 1);
m = length(y);

X = [ones(m,1), sum(a,2) sum(b,2) sum(c,2)];
%X = data(1:324600, 2:91);


% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,1:3) y(1:10,1)]');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

%[X mu sigma] = featureNormalize(X);

% Add intercept term to X
%X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

%fprintf('Running gradient descent ...\n');

% Choose some alpha value
%alpha = 0.1;
%num_iters = 3000;

% Init Theta and Run Gradient Descent
%theta = zeros(90, 1);
%[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

% Display gradient descent's result
%fprintf('Theta computed from gradient descent: \n');
%fprintf(' %f \n', theta);
%fprintf('\n');

theta = normalEqn(X,y);

a_data = data(1:10000, 2:31);
b_data = data(1:10000, 32:61);
c_data = data(1:10000, 62:91);
[ones(m,1) sum(a_data,2) sum(b_data,2) sum(c_data,2)]*theta
