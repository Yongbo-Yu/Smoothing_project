function [QoI ] = smooth_digital_price_single_path(Nsteps,Ms)
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

QoI=(1/(sqrt(2 *pi)))*exp(-bar_z.^2/2);
end

