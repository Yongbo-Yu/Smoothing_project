function [QoI ] = smooth_digital_price_single_path(Nsteps,Ms)
mean =zeros(Nsteps,1);
covariance= eye(Nsteps);
y =mvnrnd(mean, covariance,Ms);
y_1=y(:,2:Nsteps);
eps=10^-10;

bar_z=newtons_method(0,y_1,Nsteps,eps,Ms);
QoI =1-cdf('Normal',bar_z,0,1);
end

