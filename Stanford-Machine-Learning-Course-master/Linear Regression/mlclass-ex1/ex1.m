
%% Machine Learning Online Class - Exercise 1: Linear Regression线性回归

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
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
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%对于此练习，您不需要更改此文件中的任何代码或除上述文件以外的任何其他文件。

% x refers to the population size in 10,000s  以十万计的人口规模
% y refers to the profit in $10,000s  以十万美元计的收入
%

%% Initialization
clear all; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m 
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = csvread('ex1data1.txt');  %读入样本数据
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);  %调用子程序，绘制样本点

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x    x加一列，第一列全为1
theta = zeros(2, 1); % initialize fitting parameters  定义一个两行一列的向量存储theta值

% Some gradient descent settings
iterations = 1500;   %迭代次数
alpha = 0.01;        %学习率

% compute and display initial cost  计算初始的代价函数
computeCost(X, y, theta)

% run gradient descent  梯度下降算法，同步更新参数
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible 之前画好的样本点依然保留着
plot(X(:,2), X*theta, '-')    %X(:,2)注意第一项全为1，应以第二项为横坐标
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure 不再在这张图上绘制图形

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;   %第一项中的1是x0
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============实现代价函数可视化
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J 
theta0_vals = linspace(-10, 10, 100);    %用于产生-10,10之间的100点行矢量
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's  将J_vals初始化为0的矩阵
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';   %因为在surf中工作方式的原因，需要先转置
% Surface plot  三维表面图
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot   等高线图
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
