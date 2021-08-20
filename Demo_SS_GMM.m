clear;
close all

% Add path for auxiliary functions
addpath('./funcs');

% Set the random number generator    
reset(RandStream.getGlobalStream);

% Determine the experiment setting
nSig0   = 15/255;               % noise level

% Load the test image and Genenrate the corresponding noisy image
path_Img = './Monarch.png';
O_Img    = imread(path_Img);
O_Img    = double(O_Img)/255;
N_Img    = O_Img + nSig0 * randn(size(O_Img));   

% Print the psnr of the noisy image to be processed
PSNR  = csnr( O_Img*255, N_Img*255, 0, 0 );
fprintf( 'Noisy Image: nSig = %2.3f, PSNR = %2.2f \n', nSig0*255, PSNR );   

% Set parameters for SS-GMM
Par = setpar(N_Img);

% Learn the image prior with the SS-GMM
[Par.nSig, model] = SS_GMM(N_Img, Par);

% Store the trained model
path_model = 'prior.mat';
save(path_model, 'model', 'Par');

% Denoise the image with learned prior using the EPLL framework
Par.O_Img = O_Img;  % Uncomment to calculate PSNRs for intermediate results
E_Img = EPLL_GMM(N_Img, model, Par);

% Print information of processed result
E_Img(E_Img<0) = 0;
E_Img(E_Img>1) = 1;
PSNR  = csnr( O_Img*255, E_Img*255, 0, 0 );
fprintf( 'Estimated Image: nSig = %2.3f, PSNR = %2.2f \n\n', sqrt(Par.nSig)*255, PSNR ); 

% Store the denoised result in PNG format
imwrite(uint8(E_Img*255), 'img_denoised.png');
    
% Remove path for auxiliary functions
rmpath('./funcs');

















