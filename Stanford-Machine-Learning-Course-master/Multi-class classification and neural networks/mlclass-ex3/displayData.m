function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid 在漂亮的网格中显示2D数据
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in 如果未传入，则自动设置example_width
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

%exist用于确定某值是否存在
%isempty returns logical 1 (true) if A is an empty array and logical 0 (false) otherwise.

% Gray Image
colormap(gray);    %输出一个灰色系的曲面图

% Compute rows, cols  计算行，列各分为几个小格
[m n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));  %不超过x 的最大整数.(高斯取整)
display_cols = ceil(m / display_rows); %大于x 的最小整数

% Between images padding 图像填充间距
pad = 1;

% Setup blank display 设置空白显示
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
% 将每个示例复制到显示阵列上的一个补丁（格子）中，即将每个数字图像放到之前画的格子中
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image   imagesc用于缩放数据和显示图像对象
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
