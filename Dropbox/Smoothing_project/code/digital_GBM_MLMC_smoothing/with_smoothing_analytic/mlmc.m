% function [P, Nl] = mlmc(N0,eps,mlmc_l, alpha,beta,gamma)
%
% multi-level Monte Carlo estimation
%
% P      = value
% Nl     = number of samples at each level
%
% N0     = initial number of samples on levels 0,1,2
% eps    = desired accuracy (rms error)
%
% alpha  -> weak error is  O(2^{-alpha*l})
% beta   -> variance is    O(2^{-beta*l})
% gamma  -> sample cost is O(2^{gamma*l})
%
% if alpha, beta are not positive then they will be estimated
%
% mlmc_l = function for level l estimator 
%
% sums = mlmc_fn(l,N)     low-level routine
%
% inputs:  l = level
%          N = number of paths
%
% output: sums(1) = sum(Y)
%         sums(2) = sum(Y.^2)
%         where Y are iid samples with expected value:
%         E[P_0] on level 0
%         E[P_l - P_{l-1}] on level l>0

function [P, Nl] = mlmc(N0,eps,mlmc_l, alpha_0,beta_0,gamma)

  alpha = max(0, alpha_0);
  beta  = max(0, beta_0);

  L             = 2;
  Nl(1:3)       = 0;
  suml(1:2,1:3) = 0;
  dNl(1:3)      = N0;

  while sum(dNl) > 0

%
% update sample sums
%
    for l=0:L
      if dNl(l+1) > 0
        sums        = feval(mlmc_l,l,dNl(l+1));
        Nl(l+1)     = Nl(l+1) + dNl(l+1);
        suml(1,l+1) = suml(1,l+1) + sums(1);
        suml(2,l+1) = suml(2,l+1) + sums(2);
      end
    end

%
% compute absolute average and variance
%
    ml = abs(   suml(1,:)./Nl);
    Vl = max(0, suml(2,:)./Nl - ml.^2);

%
% fix to cope with possible zero values for ml and Vl
% (can happen in some applications when there are few samples)
%
    for l = 3:L+1
      ml(l) = max(ml(l), 0.5*ml(l-1)/2^alpha);
      Vl(l) = max(Vl(l), 0.5*Vl(l-1)/2^beta);
    end

%
% use linear regression to estimate alpha, beta if not given
%
    if alpha_0 <= 0
      A     = repmat((1:L)',1,2).^repmat(1:-1:0,L,1);
      x     = A \ log2(ml(2:end))';
      alpha = max(0.5,-x(1));
    end

    if beta_0 <= 0
      A    = repmat((1:L)',1,2).^repmat(1:-1:0,L,1);
      x    = A \ log2(Vl(2:end))';
      beta = max(0.5,-x(1));
    end
%
% set optimal number of additional samples
%
    Cl  = 2.^(gamma*(0:L));
    Ns  = ceil(2 * sqrt(Vl./Cl) * sum(sqrt(Vl.*Cl)) / eps^2);
    dNl = max(0, Ns-Nl);
%
% if (almost) converged, estimate remaining error and decide 
% whether a new level is required
%
    if sum( dNl > 0.01*Nl ) == 0
      range = -2:0;
      rem = max(ml(L+1+range).*2.^(alpha*range)) / (2^alpha - 1);

      if rem > eps/sqrt(2)
        L       = L+1;
        Vl(L+1) = Vl(L) / 2^beta;
        Nl(L+1) = 0;
        suml(1:4,L+1) = 0;

        Cl  = 2.^(gamma*(0:L));
        Ns  = ceil(2 * sqrt(Vl./Cl) * sum(sqrt(Vl.*Cl)) / eps^2);
        dNl = max(0, Ns-Nl);
      end
    end
  end

%
% finally, evaluate multilevel estimator
%
  P = sum(suml(1,:)./Nl);
end