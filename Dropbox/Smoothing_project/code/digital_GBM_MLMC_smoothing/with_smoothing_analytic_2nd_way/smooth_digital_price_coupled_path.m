function [Pf,Pc] = smooth_digital_price_coupled_path(Nsteps,Ms)

mean = zeros(2*Nsteps-1,1);
covariance= eye(2*Nsteps-1);
y =mvnrnd(mean, covariance,Ms);

y_1f=y(:,1:2*Nsteps-1);



eps=10^-10;   
[bar_z_f,bar_z_c]=newtons_method_coupled(0,y_1f,Nsteps,eps,Ms);

Pf= 1-cdf('Normal',bar_z_f,0,1);
Pc= 1-cdf('Normal',bar_z_c,0,1);
end
