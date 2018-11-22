function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);%��J_history��¼ÿ�ε����õ��Ĵ��ۺ���ֵ

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %�ڵ���ʱ�������ڴ�����ɱ�������computeCost�����ݶ�ֵ��

	H = X * theta;  %Ԥ�⺯��H
	T = [0 ; 0];
	for i = 1 : m,
		T = T + (H(i) - y(i)) * X(i,:)';	%���ۺ���������theta�󵼺�õ���������Ȼ�������m�������ۼӣ����ʼ�29ҳ
    end                     %X(i,:)'Ϊ��i��������ע���һ��Ϊ������ӵ�x0��1���ڶ�����������е�����ֵ��ע��Ҫת��
	                        % i=1ʱ   1.0000
                            %         6.1101
	theta = theta - (alpha * T) / m; %ͬ�����²�����ͬʱ����theta0��theta1
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
