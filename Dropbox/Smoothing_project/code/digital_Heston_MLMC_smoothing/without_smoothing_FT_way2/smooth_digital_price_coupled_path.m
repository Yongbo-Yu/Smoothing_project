function [Pf,Pc] = smooth_digital_price_coupled_path(Nsteps,Ms)

 S0=100;
 K=100;
 T=1;
 rho=-0.9;
  V0=0.04;
kappa= 1.0;
 xi=0.1;
 theta=0.0025;
   
dtc=T/Nsteps;

dtf=dtc/2;
nc=Nsteps;
Sf = S0*ones(1,Ms);
Sc = Sf;
Vf = V0*ones(1,Ms);
Vc = Vf;
M=2;
for n = 1:nc
      dWc = zeros(2,Ms);
      for m = 1:M
        dWf = sqrt(dtf)*randn(2,Ms);
        dWc = dWc + dWf;
        dW_v_f=dWf(1,:);
        dW_s_f= rho *dWf(1,:) + sqrt(1-rho^2) * dWf(2,:);
        Sf=Sf.*(1+sqrt(Vf).*dW_s_f);
        Vf=Vf- kappa *dtf* max(Vf,0)+ xi *sqrt(max(Vf,0)).*dW_v_f+ kappa*theta*dtf;
        Vf=max(Vf,0);
      end
      dW_v_c=dWc(1,:);
      dW_s_c= rho *dWc(1,:) + sqrt(1-rho^2) * dWc(2,:);
      Sc=Sc.*(1+sqrt(Vc).*dW_s_c);
      Vc=Vc- kappa *dtc* max(Vc,0)+ xi *sqrt(max(Vc,0)).*dW_v_c+ kappa*theta*dtc;
      Vc=max(Vc,0);

end

   
Pf=  0.5*(1+sign(Sf(end,:)-K));
Pc=  0.5*(1+sign(Sc(end,:)-K));
end
