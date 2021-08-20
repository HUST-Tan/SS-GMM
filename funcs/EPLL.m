function E_Pats = EPLL(N_Pats, nSig, model, norm)
%EPLL 此处显示有关此函数的摘要
%   此处显示详细说明

    % Extract parameters
    ps         = sqrt(size(model.Covs,1));
    cls_num    = length(model.w);
    SigmaNoise = eye(ps^2)*(nSig^2); 
    
    % Calculate responsibilites of all pathces for each mixture component
    r = zeros(cls_num, size(N_Pats,2));
    for k=1:cls_num
        Covs = model.Covs(:,:,k) + SigmaNoise;
        r(k,:) = log(model.w(k)) + loggausspdf2(N_Pats, Covs);
    end
    
    % Find the most likely component for each patch
    [~,k_opt] = max(r);
    
    % Perform weiner filtering
    E_Pats = zeros(size(N_Pats));
    for k=1:cls_num
        [U, S] = eig(model.Covs(:,:,k));

        inds  = find(k_opt==k);
        Temp  = N_Pats(:,inds) - model.mu(:,k);
        alpha = U'*Temp;

        switch norm
            case 'L1'
                const  = 1;
                weight = sqrt(diag(S)) + eps;
                weight = const*(nSig^2)./weight;
                alpha  = sign(alpha).*max((abs(alpha) - weight), 0);
            case 'L2'
                weight = S./(SigmaNoise + S + eps);
                alpha = weight*alpha;
            otherwise
                fprintf('Please choose the norm type!');
        end

        Temp = U*alpha;
        Temp = Temp + model.mu(:,k);

        E_Pats(:,inds) = Temp;
    end
    
end

