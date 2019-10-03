function [QoI ] = smooth_digital_price_single_path(Nsteps,Ms)
 S0=100;
 K=100;
 T=1;
 rho=-0.9;
  V0=0.04;
kappa= 1.0;
 xi=0.1;
 theta=0.0025;
   
dt=T/Nsteps;
hf=dt;



Sf = S0*ones(1,Ms);
Vf = V0*ones(1,Ms);
      
        
for n =2:1:Nsteps+1
   dWf = sqrt(hf)*randn(2,Ms);
   dW_v=dWf(1,:);
   dW_s= rho *dWf(1,:) + sqrt(1-rho^2) * dWf(2,:); 
   Sf=Sf.*(1+sqrt(Vf).*dW_s);
    Vf=Vf- kappa *dt* max(Vf,0)+ xi *sqrt(max(Vf,0)).*dW_v+ kappa*theta*dt;
   Vf=max(Vf,0);
end

 QoI = 0.5*(1+sign(Sf-K));
end

