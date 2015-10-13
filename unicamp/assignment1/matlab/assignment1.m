%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
%data = load('C:\Users\Victor Leal\Desktop\mlearning\assignment1\YearPredictionMSD.txt');
data = load('/home/victor/YearPredictionMSD.txt');
training_examples = 324600;

a = data(1:training_examples, 2:13);
b = data(1:training_examples, 2:2:91);
c = data(1:training_examples, 3:2:91);

y = data(1:training_examples, 1);
m = length(y);

X = data(1:training_examples, 2:13);
X = [mean(a,2), X];

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,1:3) y(1:10,1)]');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


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
alpha = 0.04;
num_iters = 1000;

% Init Theta and Run Gradient Descent
theta = zeros(14, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
%fprintf('Theta computed from gradient descent: \n');
%fprintf(' %f \n', theta);
fprintf('\n');

%theta = normalEqn(X,y);

%{
a_data = data(1:10000, 2:3:91);
b_data = data(1:10000, 3:3:91);
c_data = data(1:10000, 4:3:91);
%}

% VALIDATION
valid_a = data(324600:463715, 2:13);
valid_b = data(324600:463715, 2:2:91);
valid_c = data(324600:463715, 3:2:91);

y_validation = data(324600:463715, 1);
m = length(y_validation);

X_validation = data(324600:463715, 2:13);
X_validation = [mean(valid_a,2) X_validation];
X_validation = X_validation .- mu;
X_validation = X_validation ./ sigma;

X_validation = [ones(m,1) X_validation];

predictions = X_validation * theta;

MAE = mean(abs(predictions-y_validation))
