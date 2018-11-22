function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 均值为0，标准差为1
%   This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);  %将X中的每一项减去mu（均值）
% bsxfun强大的、万能的、不同维数的矩阵扩展混合运算,从此告别矩阵运算中的for循环
% 另,matlab里所有以fun为后缀的命令都很好用

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma); % X_norm中每一项 ./ sigma


% ============================================================

end
