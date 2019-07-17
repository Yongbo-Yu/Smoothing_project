%
% These are similar to the MLMC tests for the MCQMC06 paper
% using a Milstein discretisation with 2^l timesteps on level l
%
% The figures are slightly different due to 
% -- change in MSE split
% -- change in cost calculation
% -- different random number generation
% -- switch to S_0=100
%

function mcqmc06

close all; clear all;

addpath('..');

N0   = 200;    % initial samples on coarse levels
Lmin = 2;      % minimum refinement level
Lmax = 10;     % maximum refinement level
 
for option = 6:6
  if (option==1) 
    fprintf(1,'\n ---- European call Milstein---- \n');
    N      = 20000;    % samples for convergence tests
    L      = 8;        % levels for convergence tests 
    Eps    = [ 0.005 0.01 0.02 0.05 0.1 ];
  elseif (option==2) 
    fprintf(1,'\n ---- Asian call  Milstein ---- \n');
    N      = 20000;    % samples for convergence tests
    L      = 8;        % levels for convergence tests 
    Eps    = [ 0.005 0.01 0.02 0.05 0.1 ];
  elseif (option==3) 
    fprintf(1,'\n ---- lookback call  Milstein ---- \n');
    N      = 20000;    % samples for convergence tests
    L      = 10;       % levels for convergence tests 
    Eps    = [ 0.005 0.01 0.02 0.05 0.1 ];
  elseif (option==4) 
    fprintf(1,'\n ---- digital call  Milstein ---- \n');
    N      = 200000;   % samples for convergence tests
    L      = 8;        % levels for convergence tests 
    Eps    = [ 0.01 0.02 0.05 0.1 0.2 ];
  elseif (option==5) 
    fprintf(1,'\n ---- barrier call  Milstein ---- \n');
    N      = 200000;   % samples for convergence tests
    L      = 8;        % levels for convergence tests 
    Eps    = [ 0.005 0.01 0.02 0.05 0.1 ];
  elseif (option==6) 
    fprintf(1,'\n ---- digital call Euler ---- \n');
    N      = 200000;   % samples for convergence tests
    L      = 8;        % levels for convergence tests 
    Eps    = [ 0.01 0.02 0.05 0.1 0.2 ]; 
  end

  filename = ['mcqmc06_' num2str(option)];
  fp = fopen([filename '.txt'],'w');
  mlmc_test(@mcqmc06_l, N,L, N0,Eps,Lmin,Lmax, fp, option);
  fclose(fp);

%
% print exact analytic value, based on S0=K
%
  T   = 1;
  r   = 0.05;
  sig = 0.2;
  K   = 100;
  B   = 0.85*K;

  d1  = (r+0.5*sig^2)*T / (sig*sqrt(T));
  d2  = (r-0.5*sig^2)*T / (sig*sqrt(T));

  if (option==1)
    val = K*( normcdf(d1) - exp(-r*T)*normcdf(d2) );
  elseif (option==2)
    val = NaN;
  elseif (option==3)
    k   = 0.5*sig^2/r;
    val = K*( normcdf(d1) - normcdf(-d1)*k - exp(-r*T)*(normcdf(d2) - normcdf(d2)*k) );
  elseif (option==4)
    val = K*exp(-r*T)*normcdf(d2);
  elseif (option==5)
    k   = 0.5*sig^2/r;
    d3  = (2*log(B/K) + (r+0.5*sig^2)*T) / (sig*sqrt(T));
    d4  = (2*log(B/K) + (r-0.5*sig^2)*T) / (sig*sqrt(T));
    val = K*(                         normcdf(d1) - exp(-r*T)*normcdf(d2)  ...
            - (K/B)^(1-1/k)*( (B/K)^2*normcdf(d3) - exp(-r*T)*normcdf(d4) ) );
  elseif (option==6)
    val = K*exp(-r*T)*normcdf(d2);      
  end

  if isnan(val)
    fprintf(1,'\n Exact value unknown \n\n');
  else
    fprintf(1,'\n Exact value: %f \n\n',val);
  end

%
% plot results
%
  nvert = 3;
  mlmc_plot(filename, nvert);

  if(nvert==1)
    figure(1)
    print('-deps2',[filename 'a.eps'])
    figure(2)
    print('-deps2',[filename 'b.eps'])
  else
    print('-deps2',[filename '.eps'])
  end

%
% now do 100 MLMC calcs in parallel
%
  filename = ['mcqmc06_' num2str(option) '_100'];
  fp = fopen([filename '.txt'],'w');
  mlmc_test_100(@mcqmc06_l, val, N0,Eps,Lmin,Lmax, fp, option);

  fclose(fp);

%
% plot results
%
  mlmc_plot_100(filename);
  print('-deps2',[filename '.eps'])
end


%-------------------------------------------------------
%
% level l estimator
%

function [sums, cost] = mcqmc06_l(l,N, option)

K   = 100;
T   = 1;
r   = 0.05;
sig = 0.2;
B   = 0.85*K;

nf = 2^l;
nc = nf/2;

hf = T/nf;
hc = T/nc;

sums(1:6) = 0;

for N1 = 1:10000:N
  N2 = min(10000,N-N1+1);

  X0 = K;

  Xf = X0*ones(1,N2);
  Xc = Xf;

  Af  = 0.5*hf*Xf;
  Ac  = 0.5*hc*Xc;

  Mf  = Xf;
  Mc  = Xc;

  Bf  = 1;
  Bc  = 1;

  if l==0
    dWf = sqrt(hf)*randn(1,N2);
    Lf  = log(rand(1,N2));
    dIf = sqrt(hf/12)*hf*randn(1,N2);

    Xf0 = Xf;
    Xf  = Xf + r*Xf*hf + sig*Xf.*dWf ...
             + 0.5*sig^2*Xf.*(dWf.^2-hf);
    vf  = sig*Xf0;
    Af  = Af + 0.5*hf*Xf + vf.*dIf(1,:);
    Mf  = min(Mf,0.5*(Xf0+Xf-sqrt((Xf-Xf0).^2-2*hf*vf.^2.*Lf)));
    Bf  = Bf.*(1-exp(-2*max(0,(Xf0-B).*(Xf-B)./(hf*vf.^2))));
  else
    for n = 1:nc
      dWf = sqrt(hf)*randn(2,N2);
      Lf  = log(rand(2,N2));
      dIf = sqrt(hf/12)*hf*randn(2,N2);
      for m = 1:2
        Xf0 = Xf;
        Xf  = Xf + r*Xf*hf + sig*Xf.*dWf(m,:) + 0.5*sig^2*Xf.*(dWf(m,:).^2-hf);
        vf  = sig*Xf0;
        Af  = Af + hf*Xf + vf.*dIf(m,:);
        Mf  = min(Mf,0.5*(Xf0+Xf-sqrt((Xf-Xf0).^2-2*hf*vf.^2.*Lf(m,:))));
        Bf  = Bf.*(1-exp(-2*max(0,(Xf0-B).*(Xf-B)./(hf*vf.^2))));
      end

      dWc = dWf(1,:) + dWf(2,:);
      ddW = dWf(1,:) - dWf(2,:);

      Xc0 = Xc;
      Xc  = Xc + r*Xc*hc + sig*Xc.*dWc + 0.5*sig^2*Xc.*(dWc.^2-hc);

      vc  = sig*Xc0;
      Ac  = Ac + hc*Xc + vc.*(sum(dIf,1) + 0.25*hc*ddW);
      Xc1 = 0.5*(Xc0 + Xc + vc.*ddW);
      Mc  = min(Mc, 0.5*(Xc0+Xc1-sqrt((Xc1-Xc0).^2-2*hf*vc.^2.*Lf(1,:))));
      Mc  = min(Mc, 0.5*(Xc1+Xc -sqrt((Xc -Xc1).^2-2*hf*vc.^2.*Lf(2,:))));
      Bc  = Bc .*(1-exp(-2*max(0,(Xc0-B).*(Xc1-B)./(hf*vc.^2))));
      Bc  = Bc .*(1-exp(-2*max(0,(Xc1-B).*(Xc -B)./(hf*vc.^2))));
    end
    Af = Af - 0.5*hf*Xf;
    Ac = Ac - 0.5*hc*Xc;
  end

  if option==1
    Pf  = max(0,Xf-K);
    Pc  = max(0,Xc-K);
  elseif option==2
    Pf  = max(0,Af-K);
    Pc  = max(0,Ac-K);
  elseif option==3
    Pf  = Xf - Mf;
    Pc  = Xc - Mc;
  elseif option==4
    if(l==0)
      Pf  = K*normcdf((Xf0+r*Xf0*hf-K)./(sig*Xf0*sqrt(hf)));
      Pc  = Pf;
    else
      Pf  = K*normcdf((Xf0+r*Xf0*hf-K)./(sig*Xf0*sqrt(hf)));
      Pc  = K*normcdf((Xc0+r*Xc0*hc+sig*Xc0.*dWf(1,:)-K)./(sig*Xc0*sqrt(hf)));
    end
  elseif option==5
    Pf  = Bf.*max(0,Xf-K);
    Pc  = Bc.*max(0,Xc-K);
  elseif option==6
      S0  = 100;   % initial asset value
      
     
     Sf = S0*ones(1,N2);
     Sc = Sf;
     Pc = zeros(1,N2);

     if l==0
        dWf = sqrt(hf)*randn(1,N2);
        Sf  = Sf + r*Sf*hf + sig*Sf.*dWf;

      else
        for n = 1:nc
            dWc = zeros(1,N2);
            for m = 1:2
                  dWf = sqrt(hf)*randn(1,N2);
                  dWc = dWc + dWf;
                  Sf  = Sf + r*Sf*hf + sig*Sf.*dWf;
            end
        Sc = Sc + r*Sc*hc + sig*Sc.*dWc;
        end
      end
      Pf = exp(-r*T)*10*0.5*(1+sign(Sf-K));
      if (l>0)
          Pc = exp(-r*T)*10*0.5*(1+sign(Sc-K));
      end
  end

  dP  = exp(-r*T)*(Pf-Pc);
  Pf  = exp(-r*T)*Pf;

  if l==0
    dP = Pf;
  end

  sums(1) = sums(1) + sum(dP);
  sums(2) = sums(2) + sum(dP.^2);
  sums(3) = sums(3) + sum(dP.^3);
  sums(4) = sums(4) + sum(dP.^4);
  sums(5) = sums(5) + sum(Pf);
  sums(6) = sums(6) + sum(Pf.^2);
end

cost = N*nf;   % cost defined as number of fine timesteps


