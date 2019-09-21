%
% function figs = mlmc_test(mlmc_fn, M, N,L, N0,Eps, nvert)
%
% multilevel Monte Carlo test routine
%
% [sum1, sum2] = mlmc_fn(l,N)     low-level routine
%
% inputs:  l = level
%          N = number of paths
%
% output: sum1(1) = sum(Pf-Pc)
%         sum1(2) = sum((Pf-Pc).^2)
%         sum1(3) = sum((Pf-Pc).^3)
%         sum1(4) = sum((Pf-Pc).^4)
%         sum2(1) = sum(Pf)
%         sum2(2) = sum(Pf.^2)
%
% M      = refinement cost factor (2^gamma in general MLMC Thm)
%
% N      = number of samples for convergence tests
% L      = number of levels for convergence tests
%
% N0     = initial number of samples for MLMC calcs
% Eps    = desired accuracy array for MLMC calcs
%
% nvert  = 1 for 1x2 plots for slides
%          2 for 2x2 plots for papers
%          3 for 3x2 plots for full set

function figs = mlmc_test(mlmc_fn,M, N,L, N0,Eps, nvert)

%
% first, convergence tests
%

rng('default');    % reset random number generator

del1 = [];
del2 = [];
var1 = [];
var2 = [];
kur1 = [];
chk1 = [];
cost = [];

L = 0:L;

for l = L
  disp(sprintf('l = %d',l))
  tic;
  [sum1, sum2] = feval(mlmc_fn,l,N);
  cost = [ cost toc ];
  sum1 = sum1/N;
  sum2 = sum2/N;
  kurt = (sum1(4) - 4*sum1(3)*sum1(1) + 6*sum1(2)*sum1(1)^2 - 3*sum1(1)^4) ...
       / (sum1(2)-sum1(1)^2)^2;
  del1 = [del1 sum1(1)];
  del2 = [del2 sum2(1)];
  var1 = [var1 sum1(2)-sum1(1)^2 ];
  var2 = [var2 sum2(2)-sum2(1)^2 ];
  var2 = max(var2, 1e-12);  % fix for cases with var=0
  kur1 = [kur1 kurt ];

  if l==0
    check = 0;
  else
    check = abs(       del1(l+1)  +      del2(l)  -      del2(l+1)) ...
      /    ( 3.0*(sqrt(var1(l+1)) + sqrt(var2(l)) + sqrt(var2(l+1)) )/sqrt(N));
  end
  chk1 = [chk1 check];
end

%
% use linear regression to estimate alpha, beta and gamma
%

range = max(2,floor(0.4*length(L))):length(L);
fprintf('\nestimates of key MLMC Theorem parameters based on linear regression: \n')
pa = polyfit(L(range),log2(abs(del1(range))),1);  alpha = -pa(1);
fprintf('alpha = %f  (exponent for MLMC weak convergence)\n',alpha)
pb = polyfit(L(range),log2(abs(var1(range))),1);  beta = -pb(1);
fprintf('beta  = %f  (exponent for MLMC variance) \n',beta)
gamma = log2(cost(end)/cost(end-1));
fprintf('gamma = %f  (exponent for MLMC cost) \n\n',gamma)

if max(chk1) > 1
fprintf('WARNING: maximum consistency error = %f \n',max(chk1))
  fprintf('indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied \n\n')
end

if kur1(end) > 100
  fprintf('WARNING: kurtosis on finest level = %f \n',kur1(end))
  fprintf('indicates MLMC correction dominated by a few rare paths; \n')
  fprintf('for information on the connection to variance of sample variances, \n')
  fprintf('see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n\n')
end

%
% plot figures
%


% figs(2) = figure; 
% plot(L(2:end)-1e-9,kur1(2:end),'--*')
% xlabel('level l'); ylabel('kurtosis');

figs(1) = figure; 
pos=get(gcf,'pos'); pos(3:4)=pos(3:4).*[1.0 0.75*nvert]; set(gcf,'pos',pos);

set(0,'DefaultAxesColorOrder',[0 0 0]);
set(0,'DefaultAxesLineStyleOrder','-*|--*|--o|--x|--d|--s');
% subplot(nvert,2,1)
plot(L,log2(var2),L(2:end),log2(var1(2:end)),L,-2+log2(2.^(-L)), L(2:end),polyval(pb,L(2:end)),'-r')
% hold on
% plot(L,log2(var2),L(2:end),log2(var1(2:end)), ...
%                 )
xlabel('level l'); ylabel('log_2 variance');
legend('P_l','P_l- P_{l-1}','2^{-l}', strcat('\beta=',num2str(round(-pb(1),1))),'Location','northeast')
log2(abs(del2))
set(0,'DefaultAxesLineStyleOrder','-*|--*|--o|--x|--d|--s');
% subplot(nvert,2,2)

figs(2) = figure; 
abs(del2)
abs(del1(2:end))
plot(L,log2(abs(del2)),L(2:end),log2(abs(del1(2:end))),L,-6+log2(2.^(-L)),L(2:end),polyval(pa,L(2:end)),'-r')
% plot(L,log2(abs(del2)),L(2:end),log2(abs(del1(2:end))), ...
%                       L(2:end),polyval(pa,L(2:end)),'-r')
xlabel('level l'); ylabel('log_2 |mean|');
legend('P_l','P_l- P_{l-1}','2^{-l}', strcat('\alpha=',num2str(round(-pa(1),1))),'Location','northeast')

if nvert==3
    figs(7) = figure; 
%   subplot(3,2,3)
  plot(L(2:end)-1e-9,chk1(2:end),'--*')
  xlabel('level l'); ylabel('consistency check');

%   subplot(3,2,4)
figs(3) = figure; 
  plot(L(2:end)-1e-9,kur1(2:end),'--*')
  xlabel('level l'); ylabel('kurtosis');
end

% if nvert==1
%   figs(2) = figure;
%   pos=get(gcf,'pos'); pos(3:4)=pos(3:4).*[1.0 0.75]; set(gcf,'pos',pos);
% end


% second, mlmc complexity tests

figs(4) = figure; 
rng('default');    % reset random number generator

Nls = [];
ls  = [];
maxl = 0;

for i = 1:length(Eps)
  eps = Eps(i);
  fprintf('eps = %f \n',eps)
  gamma = log2(M); 
  [P, Nl] = mlmc(N0,eps,mlmc_fn,alpha,beta,gamma);
  l = length(Nl)-1;
  maxl = max(l,maxl);
  mlmc_cost(i) = (1+1/M)*sum(Nl.*M.^(0:l));
  std_cost(i)  = sum((2*var2(end)/eps^2).*M.^(0:l));

 fprintf(' mlmc_cost = %d, std_cost = %d, savings = %f \n',...
           mlmc_cost(i), std_cost(i), std_cost(i)/mlmc_cost(i))
  Nls(1:l+1,i) = Nl;
  ls(1:l+1,i)  = 0:l;
end



disp(' ');

for i = 1:length(Eps)
  Nls(end:maxl,i) = Nls(end,i);
  ls(end:maxl,i)  = ls(end,i);
end


% plot figures

figs(5) = figure; 
set(0,'DefaultAxesLineStyleOrder','--o|--x|--d|--*|--s');
% 
% subplot(nvert,2,2*nvert-1)
semilogy(ls, Nls)
xlabel('level l'); ylabel('N_l');



% subplot(nvert,2,2*nvert)
figs(6) = figure; 
set(0,'DefaultAxesLineStyleOrder','-*|--*')
loglog(Eps,Eps.^2.*std_cost(:)', Eps,Eps.^(2.1).*mlmc_cost(:)')
xlabel('accuracy \epsilon'); ylabel('\epsilon^{2.1} Cost');
legend('Std MC','MLMC','Location','southwest')


