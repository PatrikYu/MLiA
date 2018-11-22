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
%   这些参数被“展开”为向量格式了，因此需要转换回权重矩阵。
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%   返回的参数grad应该是神经网络偏导数的“展开”向量。
%   如果我们要使用fminuc这样的优化算法来求解出权重矩阵，我们需要将矩阵首先展开为向量
%   利用算法求出最优解后再重新转换回矩阵 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network   转换为权重的矩阵形式
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly  梯度初始化为0
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
%         执行前向传播算法，得到代价函数
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%         执行反向传播算法，算出偏导数值
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%               您需要将此矢量映射为1和0的二进制向量，以用于神经网络成本函数。
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%               如果您第一次实施反向传播，我们建议在训练示例中使用for-loop来实施反向传播。
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%         可以分别计算正则化的梯度，然后将它们添加到Part 2的Theta1_grad和Theta2_grad中。

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
%分别依次挑选出矩阵temp中的“第y行”然后层层叠加起来形成一个新的行数不变、列数为k的0-1矩阵
%例如y=10，那么取temp中第十行作为新的标签，即[0 0 0 0 0 0 0 0 0 1],学习了

cost = ry.*log(a3) + (1 - ry).*log(1 - a3);
J = -sum(sum(cost,2)) / m;

reg = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(: , 2:end).^2));%正则化的项，对theta0步进行正则化

J = J + lambda/(2*m)*reg; %正则化之后的代价函数

% -------------------------------------------------------------

% Part 2: Backpropagation algorithm
% -------------------------------------------------------------

%第一种方法没有看懂，和p142写的不太一样
delta3 = a3 - ry;
temp=delta3*Theta2; 
delta2 = temp(:,2:end) .* sigmoidGradient(z2);%sigmoidGradient用于计算s型函数的导数，见p142

Delta1 = delta2'*a1;
Delta2 = delta3'*a2;  
% Delta1，2为误差矩阵

Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1), Theta1(:,2:end)] / m;
Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1), Theta2(:,2:end)] / m;%theta0不做正则化处理哦
% Theta1_grad为代价函数的偏导数矩阵，见p142-143

%第二种方法

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

% Unroll gradients  偏导数展开为向量形式
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
