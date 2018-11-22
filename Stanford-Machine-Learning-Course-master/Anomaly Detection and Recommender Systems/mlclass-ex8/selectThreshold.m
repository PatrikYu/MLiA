function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers  �ҵ�����ѡ���쳣ֵ�������ֵ��epsilon��
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%   ������֤����pval���ͻ�����ʵ��yval���Ľ�����ҵ�����ѡ���쳣ֵ�������ֵ��

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;    % F1��ֵ�ۺϷ�ӳ��ȫ�����׼��

stepsize = (max(pval) - min(pval)) / 1000;  % ��1000������

for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               ����ѡ��epsilon��F1������Ϊ��ֵ����ֵ����F1�С�
    %               ѭ������ʱ�Ĵ��뽫�Ƚ����epsilonѡ���F1������
    %               ��������ڵ�ǰѡ���epsilon���ͽ�������Ϊ���epsilon
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
	% ������ʹ��predictions =��pval <epsilon�������0��1�Ķ����������Լ��쳣ֵԤ���1+
	cvPredictions = pval < epsilon;
	tp = sum((cvPredictions == 1) & (yval == 1)); %�ɹ����쳣����Ԥ��Ϊ�쳣
	fp = sum((cvPredictions == 1) & (yval == 0)); %����������Ԥ��Ϊ�쳣
	fn = sum((cvPredictions == 0) & (yval == 1)); %���쳣����Ԥ��Ϊ����
	
	prec = tp / (tp + fp + 1e-10);  %��׼�ʣ�+ 1e-10��Ϊ�˷�ֹprec��С�����¼����������ֵ
	rec = tp / (tp + fn + 1e-10);   %��ȫ��
	F1 = 2 * prec * rec / (prec + rec + 1e-10);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
