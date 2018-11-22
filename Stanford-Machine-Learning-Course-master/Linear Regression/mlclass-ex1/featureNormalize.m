function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms. 特征缩放为均值为1，标准差为0的数据

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));      %X的列数，即特征值个数，mu用来存放每个特征的均值，sigma存放标准差std
sigma = zeros(1, size(X, 2));   

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract 减去it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
m = size(X , 1);  %X的行数，即样本个数
mu = mean(X);
for i = 1 : m,
	X_norm(i, :) = X(i , :) - mu;   %使转换后的均值为0
end

sigma = std(X);
for i = 1 : m,
	X_norm(i, :) = X_norm(i, :) ./ sigma;
end

%mu , sigma , X_norm



% ============================================================

end
