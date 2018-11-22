function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%   ��Щ��������չ����Ϊ������ʽ�ˣ������Ҫת����Ȩ�ؾ���
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%   ���صĲ���gradӦ����������ƫ�����ġ�չ����������
%   �������Ҫʹ��fminuc�������Ż��㷨������Ȩ�ؾ���������Ҫ����������չ��Ϊ����
%   �����㷨������Ž��������ת���ؾ��� 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network   ת��ΪȨ�صľ�����ʽ
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly  �ݶȳ�ʼ��Ϊ0
J = 0;
Theta1_grad = zeros(size(Theta1)); 
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%         ִ��ǰ�򴫲��㷨���õ����ۺ���
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%         ִ�з��򴫲��㷨�����ƫ����ֵ
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%               ����Ҫ����ʸ��ӳ��Ϊ1��0�Ķ�����������������������ɱ�������
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%               �������һ��ʵʩ���򴫲������ǽ�����ѵ��ʾ����ʹ��for-loop��ʵʩ���򴫲���
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%         ���Էֱ�������򻯵��ݶȣ�Ȼ���������ӵ�Part 2��Theta1_grad��Theta2_grad�С�

X = [ones(m , 1)  X];

% Part 1: CostFunction
% -------------------------------------------------------------

a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m , 1)  a2];
a3 = sigmoid(a2*Theta2');

temp=eye(num_labels);
ry = temp(y,:);
%�ֱ�������ѡ������temp�еġ���y�С�Ȼ������������γ�һ���µ��������䡢����Ϊk��0-1����
%����y=10����ôȡtemp�е�ʮ����Ϊ�µı�ǩ����[0 0 0 0 0 0 0 0 0 1],ѧϰ��

cost = ry.*log(a3) + (1 - ry).*log(1 - a3);
J = -sum(sum(cost,2)) / m;

reg = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(: , 2:end).^2));%���򻯵����theta0����������

J = J + lambda/(2*m)*reg; %����֮��Ĵ��ۺ���

% -------------------------------------------------------------

% Part 2: Backpropagation algorithm
% -------------------------------------------------------------

%��һ�ַ���û�п�������p142д�Ĳ�̫һ��
delta3 = a3 - ry;
temp=delta3*Theta2; 
delta2 = temp(:,2:end) .* sigmoidGradient(z2);%sigmoidGradient���ڼ���s�ͺ����ĵ�������p142

Delta1 = delta2'*a1;
Delta2 = delta3'*a2;  
% Delta1��2Ϊ������

Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1), Theta1(:,2:end)] / m;
Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1), Theta2(:,2:end)] / m;%theta0�������򻯴���Ŷ
% Theta1_gradΪ���ۺ�����ƫ�������󣬼�p142-143

%�ڶ��ַ���

%G1 = zeros(size(Theta1));
%G2 = zeros(size(Theta2));
%for i = 1 : m,
%	ra1 = X(i,:)';
%	rz2 = Theta1*ra1;
%	ra2 = sigmoid(rz2);
%	ra2 = [1;ra2];
%	rz3 = Theta2*ra2;
%	ra3 = sigmoid(rz3);
	
%	err3 = ra3 - ry(i,:)';
	
%	err2 = (Theta2'*err3)(2:end , 1) .* sigmoidGradient(rz2);
	
%	G1 = G1 + err2 * ra1';
%	G2 = G2 + err3 * ra2';
%end


%Theta1_grad = G1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
%Theta2_grad = G2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;

% =========================================================================

% Unroll gradients  ƫ����չ��Ϊ������ʽ
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end