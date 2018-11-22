function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers  找到用于选择异常值的最佳阈值（epsilon）
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%   根据验证集（pval）和基础事实（yval）的结果，找到用于选择异常值的最佳阈值。

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;    % F1的值综合反映查全率与查准率

stepsize = (max(pval) - min(pval)) / 1000;  % 分1000步来走

for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               计算选择epsilon的F1分数作为阈值并将值放在F1中。
    %               循环结束时的代码将比较这个epsilon选择的F1分数，
    %               如果它优于当前选择的epsilon。就将其设置为最佳epsilon
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
	% 您可以使用predictions =（pval <epsilon）来获得0和1的二进制向量以及异常值预测的1+
	cvPredictions = pval < epsilon;
	tp = sum((cvPredictions == 1) & (yval == 1)); %成功将异常数据预测为异常
	fp = sum((cvPredictions == 1) & (yval == 0)); %错将正常数据预测为异常
	fn = sum((cvPredictions == 0) & (yval == 1)); %错将异常数据预测为正常
	
	prec = tp / (tp + fp + 1e-10);  %查准率，+ 1e-10是为了防止prec过小，导致计算出现奇异值
	rec = tp / (tp + fn + 1e-10);   %查全率
	F1 = 2 * prec * rec / (prec + rec + 1e-10);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
