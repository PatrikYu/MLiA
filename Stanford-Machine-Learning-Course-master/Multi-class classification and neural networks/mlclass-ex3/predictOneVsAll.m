function p = predictOneVsAll(all_theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters all_theta
%   p = PREDICT(all_theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(all_theta'*x) >= 0.5, predict 1)
%   ����0.5��Ԥ��Ϊ1

m = size(X, 1);
num_labels = size(all_theta, 2);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);   % �Բ���������Ԥ����࣬�������ܸ�����ά����

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
%       ����ʹ��max����ѡ�����

C = sigmoid(X*all_theta');  %����X*all_theta'��S�ͺ������õ�ÿһ�����������ڲ���������Ԥ������
[M , p] = max(C , [] , 2);   % M��ÿһ�У�һ�ж�Ӧһ�������������ֵ��p��ÿһ�����ֵ������ֵ��ǡ�þ������1-10





% =========================================================================


end
