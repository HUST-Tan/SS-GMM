function N = histcounts_weight(X, W, edges)
%HISTCOUNTS_WEIGHT 此处显示有关此函数的摘要
%   此处显示详细说明
    N = zeros(1, length(edges)-1);
    for ii = 1:length(edges)-1
        N(ii) = N(ii) + sum(W(X>edges(ii) & X<=edges(ii+1)));
    end
end

