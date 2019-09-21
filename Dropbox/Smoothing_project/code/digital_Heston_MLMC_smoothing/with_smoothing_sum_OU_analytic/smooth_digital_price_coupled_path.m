function [Pf,Pc] = smooth_digital_price_coupled_path(Nsteps,Ms)

mean = zeros(4*Nsteps-1,1);
covariance= eye(4*Nsteps-1);
y =mvnrnd(mean, covariance,Ms);


y1f=y(:,1:2*Nsteps);
y2f=zeros(Ms,2*Nsteps);
y2f(:,1)=zeros(Ms,1);
y2f(:,2:2*Nsteps)=y(:,2*Nsteps+1:4*Nsteps-1);
y2sf=y2f(:,2:2*Nsteps);

    
y1c=y1f(:,1:Nsteps);
y2c=y2f(:,1:Nsteps);
y2sc=y2c(:,2:Nsteps);

eps=10^-10;   
bar_z_f=newtons_method(y2f(:,1)',y2sf,y1f(:,1),y1f(:,2:2*Nsteps),2*Nsteps,eps,Ms);

bar_z_c=newtons_method(y2c(:,1)',y2sc,y1c(:,1),y1c(:,2:Nsteps),Nsteps,eps,Ms);


Pf= 1-cdf('Normal',bar_z_f,0,1);
Pc= 1-cdf('Normal',bar_z_c,0,1);
end
