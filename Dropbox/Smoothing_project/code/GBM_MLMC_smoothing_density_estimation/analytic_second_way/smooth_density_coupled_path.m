function [Pf,Pc] = smooth_density_coupled_path(Nsteps,Ms)

mean = zeros(2*Nsteps-1,1);
covariance= eye(2*Nsteps-1);
y =mvnrnd(mean, covariance,Ms);

y_1f=y(:,1:2*Nsteps-1);



eps=10^-10;   
[bar_z_f,bar_z_c]=newtons_method_coupled(0,y_1f,Nsteps,eps,Ms);

Pf= (1/(sqrt(2 *pi)))*exp(-bar_z_f.^2/2);
Pc= (1/(sqrt(2 *pi)))* exp(-bar_z_c.^2/2);
end
