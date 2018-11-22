function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3    %当特征值小于3个的时候，决策边界做平面图
    % Only need 2 points to define a line, so choose two endpoints 确定两点
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];  %确定两点横坐标

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1)); %对应的纵坐标

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])   %横坐标为30-100
else                 %当特征值大于3个的时候，用等高线图做决策边界
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;   %特征映射函数到多项式特征
        end
    end
    z = z'; % important to transpose z before calling contour注意要先将z转置

    % Plot z = 0
    % Notice you need to specify the range [0, 0] 注意你需要指定范围[0，0]
    % contour  这个MATLAB函数绘制矩阵Z的等高线图，其中Z被解释为 相对于x-y平面的高度。
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
