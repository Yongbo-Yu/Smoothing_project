function [QoI ] = smooth_digital_price_single_path_digital(Nsteps,Ms)
mean =zeros(2*Nsteps-1,1);
covariance= eye(2*Nsteps-1);
y =mvnrnd(mean, covariance,Ms);

y1=y(:,1:Nsteps);
y2=zeros(Ms,Nsteps);
y2(:,1)=zeros(Ms,1);

y2(:,2:Nsteps)=y(:,Nsteps+1:2*Nsteps-1);
y2s=y2(:,2:Nsteps);
      
eps=10^-10;
bar_z=newtons_method(y2(:,1)',y2s,y1(:,1),y1(:,2:Nsteps),Nsteps,eps,Ms);

[x, w] = GaussLaguerre(128,0);
X_1l=zeros(length(x),Ms);
for i=1:128
X_1l(i,:)=stock_price_trajectory_1D_heston(bar_z-x(i),y2s,y1(:,1),y1(:,2:Nsteps),Nsteps,Ms);
end

QoI_left=w'*(payoff_digital(X_1l).*1/(sqrt(2 *pi)).*exp(-(bar_z-x).^2/2).*exp(x));

X_1r=zeros(length(x),Ms);
for i=1:128
X_1r(i,:)=stock_price_trajectory_1D_heston(bar_z+x(i),y2s,y1(:,1),y1(:,2:Nsteps),Nsteps,Ms);
end

QoI_right=w'*(payoff_digital(X_1r)*(1/(sqrt(2 *pi))).*exp(-(bar_z+x).^2/2).*exp(x));

QoI=QoI_left+QoI_right;
end

