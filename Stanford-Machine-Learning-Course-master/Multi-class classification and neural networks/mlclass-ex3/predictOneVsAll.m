function p = predictOneVsAll(all_theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters all_theta
%   p = PREDICT(all_theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(all_theta'*x) >= 0.5, predict 1)
%   大于0.5，预测为1

m = size(X, 1);
num_labels = size(all_theta, 2);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);   % 对测试样本的预测分类，（样本总个数）维向量

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       建议使用max函数选出类别

C = sigmoid(X*all_theta');  %计算X*all_theta'的S型函数，得到每一个分类器对于测试样本的预测类型
[M , p] = max(C , [] , 2);   % M：每一行（一行对应一个样本）的最大值，p：每一行最大值的索引值，恰好就是类别1-10





% =========================================================================


end
