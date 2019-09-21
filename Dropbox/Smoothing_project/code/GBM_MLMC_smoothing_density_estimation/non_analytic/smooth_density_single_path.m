function [QoI ] = smooth_density_single_path(Nsteps,Ms)
mean =zeros(Nsteps-1,1);
covariance= eye(Nsteps-1);
y =mvnrnd(mean, covariance,Ms);
y_1=y(:,1:Nsteps-1);
eps=10^-10;

bar_z=newtons_method(0,y_1,Nsteps,eps,Ms);

[x, w] = GaussLaguerre(32,0);
X_1l=zeros(length(x),Ms);
for i=1:32
X_1l(i,:)=stock_price_trajectory_1D_BS(bar_z-x(i),y_1,Nsteps,Ms);
end
X_1l =round( X_1l,2);
QoI_left=w'*(payoff_den(X_1l).*1/(sqrt(2 *pi)).*exp(-(bar_z-x).^2/2).*exp(x));

X_1r=zeros(length(x),Ms);
for i=1:32
X_1r(i,:)=stock_price_trajectory_1D_BS(bar_z+x(i),y_1,Nsteps,Ms);
end
X_1r =round( X_1r,2);
QoI_right=w'*(payoff_den(X_1r)*(1/(sqrt(2 *pi))).*exp(-(bar_z+x).^2/2).*exp(x));

QoI=QoI_left+QoI_right;

end

