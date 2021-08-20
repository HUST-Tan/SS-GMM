function [E_nSig, model] = SS_GMM(N_Img, Par)
%This function learns image prior directly from a single noisy image.

% Convert the input noisy image to (overlapped) noisy patches
ps     = Par.ps;
N_Pats = im2col(N_Img, [ps ps]);

% Remove DC component of each patch
M_Pats = mean(N_Pats);
N_Pats = N_Pats - M_Pats;
        
% Initialize with the EM-GMM algorithm
fprintf('EM-GMM: running ... \n');
tol      = 1e-4; 
maxiter  = 100;
llh_EM   = -inf(1,maxiter);
R        = initialization(N_Pats, Par.cn);   % Initialize R
for iter = 2:maxiter
    % Remove empty clusters
    [~,label(1,:)] = max(R,[],2);
    R = R(:,unique(label));
    Par.cn = size(R, 2);
    % M-Step: update model parameters
    model = maximization(N_Pats, R);
    % E-Step; update R
    [R, llh_EM(iter)] = expectation(N_Pats, model);
    % Print iteration information
    fprintf('Iteration %d, logL: %.2f\n', iter, llh_EM(iter));               
    % Stopping criterion
    if abs(llh_EM(iter)-llh_EM(iter-1)) < tol*abs(llh_EM(iter)); break; end
end

% Start the SS-GMM
fprintf('SS-GMM: running ... \n');
llh_SS   = -inf(1,maxiter);
for iter = 2:maxiter
    % Remove empty clusters
    [~,label(1,:)] = max(R,[],2);
    R = R(:,unique(label));
    Par.cn = size(R, 2);
    % M-Step: update model parameters
    model = maximization(N_Pats, R);
    % Noise estimation and Covariance correction
    [E_nSig, N_Covs] = cov_correction(model);
    model.Covs       = N_Covs;  
    % E-Step; update R
    [R, llh_SS(iter)] = expectation(N_Pats, model);
    % Print iteration information
    fprintf('Iteration %d, logL: %.2f, E_nSig: %2.2f\n', iter, llh_SS(iter), sqrt(E_nSig)*255);           
    % Stopping criterion
    if abs(llh_SS(iter)-llh_SS(iter-1)) < tol*abs(llh_SS(iter)); break; end
end

% Update the Covariance for further processing
E_Covs = zeros(size(N_Covs));
for ii = 1:Par.cn
    % Do EVD
    Cov    = N_Covs(:,:,ii);
    Cov    = (Cov + Cov')/2; % to avoid numerical error
    [U, S] = eig(Cov); 
    % Substract noise level from covariance eigenvalues
    S = S - E_nSig;
    S(S<1e-6) = 1e-6;
    % Do inverse EVD
    Cov = U * S * U';
    Cov = (Cov + Cov')/2;
    E_Covs(:,:,ii) = Cov;
end
model.Covs = E_Covs;  


% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
function [E_nSig, Covs] = cov_correction(model)
% Load model parameters 
w       = model.w;
Covs    = model.Covs;
[d,~,k] = size(Covs);

% Do eigenvalue decomposition (EVD) for each covariance matrix
U_all   = zeros(d, d, k);
S_all   = zeros(d, k);
for ii = 1:k
    Cov    = Covs(:,:,ii);
    Cov    = (Cov + Cov')/2; % to avoid numerical error
    [U, S] = eig(Cov); 
    
    S_all(:,ii)   = diag(S);
    U_all(:,:,ii) = U; 
end

% Estimate the noise level
ratio   = 0.5;
temp    = S_all(2:end,:)*255*255;

[~,idx] = sort(w, 'descend');
temp    = temp(:,idx(1:5));

w      = ones(size(temp));
counts = histcounts_weight(temp(:), w(:), 0:10:10000);
idx    = find(counts >= max(counts)*ratio);
idx    = (idx-1)*10;
if length(idx) == 1
    E_nSig = idx;
else
    idx    = find(temp>=idx(1) & temp<=idx(end));
    E_nSig = sum(temp(idx).*w(idx))/sum(w(idx));  
end
E_nSig = sqrt(E_nSig);
E_nSig = (E_nSig/255)^2;
    
% Correct the covariance eigenvalues
temp    = S_all(2:end,:);
[~, I]  = sort(temp(:));
w       = ones(size(temp));
temp    = temp .* w;
for ii = 1:length(I)
    nSig = sum(temp(I(1:ii)))/sum(w(I(1:ii)));
    if nSig > E_nSig
        break;
    end
end
L    = ii;
temp = S_all(2:end,:);
temp(I(1:L))   = E_nSig;
S_all(2:end,:) = temp;

% Recover each covariance matrix
for ii = 1:k
    S   = diag(S_all(:, ii));
    U   = U_all(:, :, ii);  
    Cov = U * S * U';
    Cov = (Cov + Cov')/2;
    Covs(:,:,ii) = Cov;
end


% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
function R = initialization(X, init)
n = size(X,2);
if isstruct(init)  % init with a model
    R  = expectation(X,init);
elseif numel(init) == 1  % random init k
    k = init;
    label = ceil(k*rand(1,n));
    R = full(sparse(1:n,label,1,n,k,n));
elseif all(size(init)==[1,n])  % init with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
function [R, llh] = expectation(X, model)
w    = model.w;
mu   = model.mu;
Covs = model.Covs;

n = size(X,2);
k = size(mu,2);
R = zeros(n,k);
for i = 1:k
    R(:,i) = loggausspdf(X,mu(:,i),Covs(:,:,i));
end
R   = bsxfun(@plus,R,log(w));
T   = logsumexp(R,2);
R   = exp(bsxfun(@minus,R,T));
llh = sum(T)/n;                             % loglikelihood

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
function model = maximization(X, R)
[d,n] = size(X);
k  = size(R,2);
nk = sum(R,1);
w  = nk/n;
mu = zeros(d,k);                                                        
% mu = bsxfun(@times, X*R, 1./nk);

Covs = zeros(d,d,k);
r = sqrt(R);
for i = 1:k
    Xo = bsxfun(@minus,X,mu(:,i));
    Xo = bsxfun(@times,Xo,r(:,i)');
    Covs(:,:,i) = Xo*Xo'/nk(i)+eye(d)*(1e-6);
end

model.w = w;
model.mu = mu;
model.Covs = Covs;

% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
% ----------------------------------------------------------------------- %
function y = loggausspdf(X, mu, Covs)
d = size(X,1);
X = bsxfun(@minus,X,mu);
[U,p]= chol(Covs);
if p ~= 0
    error('ERROR: Covs is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);                         % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));    % normalization constant
y = -(c+q)/2;
