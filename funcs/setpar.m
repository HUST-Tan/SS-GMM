function Par = setpar(N_Img)
%This function does a coarse noise level estimation for the input noisy 
%image and sets parameters for the following SS-GMM algorithm.

% Convert the input noisy image to (overlapped) noisy patches
ps     = 7;
N_Pats = im2col(N_Img, [ps ps]);

% Calculate the covariance of noisy patches
N_Covs = N_Pats * N_Pats';
N_Covs = N_Covs / size(N_Pats, 2);

% Do eigenvalue decomposition (EVD) for this covariance matrix
[~, S] = eig(N_Covs);
S      = diag(S);

% Regard the smallest eigenvalue as the noise level (coarse estimation)
E_nSig = min(S);
E_nSig = 255 * sqrt(E_nSig);        % Note: the input image is normalized by dividing 255

% Set parameters for SS-GMM with the coarsely estimated noise level
if E_nSig <  20                     % for small noise levels
    Par.ps = 9;                     % patch size
    Par.cn = 50;                    % number of Gaussian components
end

if E_nSig >= 20 && E_nSig < 40      % for middle noise levels
    Par.ps = 9;                     % patch size
    Par.cn = 50;                    % number of Gaussian components
end

if E_nSig > 40                      % for large noise levels
    Par.ps = 11;                     % patch size
    Par.cn = 20;                    % number of Gaussian components
end

end

