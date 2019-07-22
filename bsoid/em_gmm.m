function [grp_max, gmm_max, ll_max,maxll] = em_gmm(datspace, kclass, n)
% EMGMM    Perform expectation maximization algorithm to find clusters such that it fits a mixture of Gaussians.
%  
%   [GRP_MAX, GMM_MAX, LL_MAX, MAXLL] = EM_GMM(DATSPACE, KCLASS, N)
%   DATSPACE    Action space or really just any matrix. Rows represents variables (action space axes) and columns indicate observations (time)
%   KCLASS    Maximum number of desired groups to parse action space into (random initialization). Unsupervised learning.
%   N    Iteration number of randomly initialized points to attempt at finding the global optimum.
%
%   GRP_MAX    A row for groups in the time-varying signal.
%   GMM_MAX    Trained Gaussian mixture models with best fitted parameters prior, mean and covariance matrix.
%   LL_MAX    Log(likelihood) for the optimal N. Plot to check if it converged.
%   LL_N    Log(likelihood) for the N iterations.
%
%   Written by Mo Chen (sth4nth@gmail.com).
%   Modified by Alexander Hsu, Date: 070819
%   Contact ahsu2@andrew.cmu.edu

    %% Initialization
    fprintf('Randomly initializing for n loops for fitting action space into Gaussian Mixture Models... \n');
    optloss = 1e-6;
    itmax = 500;
    for n_rand = 1:n
        ll = -inf(1,itmax);
        R = initl(datspace,kclass);
        for iter = 2:itmax
            [~,grp{n_rand}(1,:)] = max(R,[],2);
            R = R(:,unique(grp{n_rand}));   % remove empty clusters
            gmm{n_rand} = maxim(datspace,R);
            [R, ll(iter)] = expect(datspace,gmm{n_rand});
            if abs(ll(iter)-ll(iter-1)) < optloss*abs(ll(iter)) 
                break; 
            end
        end
        lln{n_rand} = ll(2:iter);
        maxll(n_rand) = max(lln{n_rand});
    end
    n_max = find(maxll == max(maxll));
    grp_max = grp{n_max};
    gmm_max = gmm{n_max};
    ll_max = lln{n_max};
return

function R = initl(datspace, kclass)
    n = size(datspace,2); % Number of observations
    if numel(kclass) == 1  % Random initialization of k classes
        grp = ceil(kclass*rand(1,n));
        R = full(sparse(1:n,grp,1,n,kclass,n));
    else
        error('ERROR: init is not valid.');
    end
return

function [R, ll] = expect(datspace, gmm)
    mu = gmm.mu;
    sig = gmm.sig;
    w = gmm.w;
    n = size(datspace,2);
    k = size(mu,2);
    R = zeros(n,k);
    for i = 1:k
        R(:,i) = lgauspdf(datspace,mu(:,i),sig(:,:,i));
    end
    R = bsxfun(@plus,R,log(w));
    T = logsumexp(R,2);
    ll = sum(T)/n; % loglikelihood
    R = exp(bsxfun(@minus,R,T));
return

function gmm = maxim(datspace, R)
    [d,n] = size(datspace);
    k = size(R,2);
    nk = sum(R,1);
    w = nk/n;
    mu = bsxfun(@times, datspace*R, 1./nk);
    sig = zeros(d,d,k);
    r = sqrt(R);
    for i = 1:k
        Xo = bsxfun(@minus,datspace,mu(:,i));
        Xo = bsxfun(@times,Xo,r(:,i)');
        sig(:,:,i) = Xo*Xo'/nk(i)+eye(d)*(1e-6);
    end
    gmm.mu = mu;
    gmm.sig = sig;
    gmm.w = w;
return

function y = lgauspdf(datspace, mu, sig)
    d = size(datspace,1); % dimension
    datspace = bsxfun(@minus,datspace,mu); % difference from each data point to the cluster mean
    [U,p]= chol(sig); % Cholesky factorization of covariance matrix
    %% Double check if the covariance matrix is symmetric postivie definite
    if p ~= 0
        error('ERROR: covariance matrix sigma is not positive definite.');
    end
    Q = U'\datspace;
    M = dot(Q,Q,1);  % M distance after taking dot product
    c = d*log(2*pi)+2*sum(log(diag(U)));   % Gaussian formula for normalization constant
    y = -(c+M)/2;
return

function s = logsumexp(X, dim)
    % subtract the largest in each dim
    y = max(X,[],dim);
    s = y+log(sum(exp(bsxfun(@minus,X,y)),dim));
    i = isinf(y);
    if any(i(:))
        s(i) = y(i);
    end
return