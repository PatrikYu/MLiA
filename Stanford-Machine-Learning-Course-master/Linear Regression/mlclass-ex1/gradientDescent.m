function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);%用J_history记录每次迭代得到的代价函数值

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %在调试时，可以在此输出成本函数（computeCost）和梯度值。

	H = X * theta;  %预测函数H
	T = [0 ; 0];
	for i = 1 : m,
		T = T + (H(i) - y(i)) * X(i,:)';	%代价函数对所有theta求导后得到的向量，然后对所有m个样本累加，见笔记29页
    end                     %X(i,:)'为第i个样本，注意第一项为我们添加的x0即1，第二项才是数据中的特征值，注意要转置
	                        % i=1时   1.0000
                            %         6.1101
	theta = theta - (alpha * T) / m; %同步更新参数，同时更新theta0，theta1
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
