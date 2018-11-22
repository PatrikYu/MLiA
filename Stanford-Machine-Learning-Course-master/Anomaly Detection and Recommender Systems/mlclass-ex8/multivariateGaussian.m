function p = multivariateGaussian(X, mu, Sigma2)
%MULTIVARIATEGAUSSIAN Computes the probability density function of the
%multivariate gaussian distribution. 计算多元高斯分布的概率密度函数。
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
%    treated as the covariance matrix. If Sigma2 is a vector, it is treated
%    as the \sigma^2 values of the variances in each dimension (a diagonal
%    covariance matrix)
%    MULTIVARIATEGAUSSIAN计算参数mu和Sigma2的多元高斯分布下示例X的概率密度函数。 
%    如果Sigma2是矩阵，则将其视为协方差矩阵。 
%    如果Sigma2是向量，则将其视为每个维度中方差的\ sigma ^ 2值（对角协方差矩阵）

k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1) %sigma是向量
    Sigma2 = diag(Sigma2);  %矩阵对角元素的提取并创建对角阵
end

X = bsxfun(@minus, X, mu(:)');   % @minus实现除法功能
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end