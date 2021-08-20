function N = histcounts_weight(X, W, edges)
%HISTCOUNTS_WEIGHT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    N = zeros(1, length(edges)-1);
    for ii = 1:length(edges)-1
        N(ii) = N(ii) + sum(W(X>edges(ii) & X<=edges(ii+1)));
    end
end

