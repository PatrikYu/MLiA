%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear all; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);  %����ֵ��������������ֵ
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
%matlab���Դ���һ���ĳ��򣬿����������������ʦ�������������Լ�д��
%%��Z����ִ�й�һ�������ڶ����Խ��й�һ���������壨�б�ʾ����������������3*4�ľ��󣬱�ʾ��4��ѵ�����ݣ�ÿ��������3������,
%Z_norm=mapminmax(Z');
%Z=Z_norm';             %�õ�����ֵ����Z

fprintf('Normalizing Features ...\n');

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1),X]; %����x0


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
%               you the best result.�����ĸ�ѧϰ�����
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.ͬһ��ͼ��������
%
% Hint: At prediction, make sure you do the same feature normalization.
%    ע�⣺Ԥ���x������֮ǰ���������ŷ�ʽҪһ��

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 100;  %��������100

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);  %��������ֵ������theta
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph ���ƴ��ۺ�������ͼ������������������
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);%���Ϊ2����ɫ��
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones.����һ�£�X�ĵ�һ����ȫ������1
% Thus, it does not need to be normalized.���ԣ�����������x0û�Ž���֮ǰ��
temp=[1650,3];      %���������
temp = temp - mu;   %ʹת����ľ�ֵΪ0
temp=temp ./ sigma;
price = [1, temp] *theta; % You should change this



% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n'); %�����淽������

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house   
% ====================== YOUR CODE HERE ======================
price = [1, 1650,3] *theta; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
     
     %���ַ�������ֱ�Ϊ$293081.476856��$293081.464335

