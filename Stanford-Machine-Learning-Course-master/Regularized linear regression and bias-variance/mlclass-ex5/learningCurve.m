function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               The vector numex_vec contains the number of training 
%               examples to use for each calculation of training error and 
%               cross validation error, i.e, error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%       填写此函数以返回error_train中的训练误差和error_val中的交叉验证误差。
%       向量numex_vec包含用于每次计算训练误差和交叉验证误差的训练示例的数量
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%       您应该评估第一个训练样例（即X（1：i，:)和y（1：i））上的训练误差。
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%      对于交叉验证误差，您应该对整个交叉验证集（Xval和yval）进行评估。

% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%    如果使用代价函数来计算训练和交叉验证误差，注意要把lamba值设为0
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%     但是当执行训练过程以获得theta值时，要加上lamba值

% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
val_size = size(Xval,1);

for i = 1 : m,
	[theta] = trainLinearReg([ones(i , 1) X(1:i , :)], y(1:i), lambda);
    % 用前i个样本训练theta
	[error_train(i), grad] = linearRegCostFunction([ones(i , 1) X(1:i, :)], y(1:i), theta, 0);
    % 对前i个训练样本评估训练误差，注意此时lamba为0
	[error_val(i), grad] = linearRegCostFunction([ones(val_size , 1) Xval], yval, theta, 0);
    % 对所有交叉验证样本评估交叉验证误差
end






% -------------------------------------------------------------

% =========================================================================

end
