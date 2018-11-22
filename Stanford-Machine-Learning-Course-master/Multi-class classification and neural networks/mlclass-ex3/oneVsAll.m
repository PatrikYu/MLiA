function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);   % 样本个数
n = size(X, 2);   % 每一个样本维数，400，可以理解为特征值个数

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector. 返回一个列向量。
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%       建议使用fmincg作为高级优化函数
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%       fmincg与fminunc类似，但在处理大量参数时效率更高。
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

	for c = 1 : num_labels,    %构造了num_labels个分类器
		initial_theta = zeros(n + 1 , 1);
		options = optimset('GradObj' , 'on' , 'MaxIter' , 50);
		[theta] = ...
			fmincg(@(t)(lrCostFunction(t , X , (y == c) , lambda)), ...
					initial_theta , options);
                % (y == c)实际上选出了y中标签值为c的类别定义为1，其他的类别定义新标签为0
		all_theta(c,:) = theta';
        % 每个分类器所训练出的theta存入每一行



% =========================================================================


end
