function [E_Img, psnrs] = EPLL_GMM(N_Img, model, Par)
%This function is to denoise input noisy image with the learned prior (GMM).
% Set/Load hyper-parameters
betas   = [1 4 8 16 32 64];
itermax = 3;
ps      = Par.ps;
nSig0   = Par.nSig; 

% Iteration for Half Quadratic Split (HQS) method
I_Img = N_Img;
psnrs = zeros(itermax, 2);
for iter_HQS = 1:itermax
    % Set beta for this iteration
    beta = betas(iter_HQS);
    nSig = nSig0/beta;
    
    % Extract patches from the noisy image to be processed 
    N_Pats = im2col(I_Img, [ps ps]);
    
    % Remove DC component of each patch
    M_Pats = mean(N_Pats);
    N_Pats = N_Pats - M_Pats;
    
    % Denoise each patch with the EPLL-GMM
    E_Pats = EPLL(N_Pats, sqrt(nSig), model, 'L2'); 
    
    % Add DC component back
    E_Pats = E_Pats + M_Pats;

    % Aggregate the estimated patches into the whole image
    E_Img = scol2im(E_Pats, ps, size(N_Img,1), size(N_Img,2), 'average');

    % Calculate the current estimate for the clean image
    I_Img = (beta*E_Img + N_Img)./(beta+1+eps);
    
    if isfield(Par, 'O_Img')
        % Calculate and Show PNSR result
        psnrs(iter_HQS,1) = csnr( I_Img*255, Par.O_Img*255, 0, 0 );
        psnrs(iter_HQS,2) = csnr( E_Img*255, Par.O_Img*255, 0, 0 );
        fprintf('Iter %d: I_Img PSNR:%2.2f, E_Img PSNR is:%2.2f \n',...
                 iter_HQS, psnrs(iter_HQS,1), psnrs(iter_HQS,2));
    end
end

end

