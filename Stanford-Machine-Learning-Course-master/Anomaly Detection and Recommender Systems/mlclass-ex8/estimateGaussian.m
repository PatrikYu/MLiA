function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
% mu（i）应该包含第i个特征的数据的平均值，mu为11维的向量
% 而sigma2（i）应该包含第i个特征的方差。

mu = mean(X)';
sigma2 = var(X)' * (m - 1) / m;
%var函数求的是样本方差的无偏估计，分母为(m-1),要转变为方差，需要换算








% =============================================================


end
