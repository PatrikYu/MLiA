function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms. ��������Ϊ��ֵΪ1����׼��Ϊ0������

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));      %X��������������ֵ������mu�������ÿ�������ľ�ֵ��sigma��ű�׼��std
sigma = zeros(1, size(X, 2));   

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract ��ȥit from the dataset,
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
m = size(X , 1);  %X������������������
mu = mean(X);
for i = 1 : m,
	X_norm(i, :) = X(i , :) - mu;   %ʹת����ľ�ֵΪ0
end

sigma = std(X);
for i = 1 : m,
	X_norm(i, :) = X_norm(i, :) ./ sigma;
end

%mu , sigma , X_norm



% ============================================================

end
