function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%   将两个输入特征映射到正则化练习中使用的二次特征。
%
%   Returns a new feature array with more features, comprising of 
%   将原来的两个特征转换为如下的多个多项式特征
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..X1.^6...
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);   % 妙呀
    end
end

end