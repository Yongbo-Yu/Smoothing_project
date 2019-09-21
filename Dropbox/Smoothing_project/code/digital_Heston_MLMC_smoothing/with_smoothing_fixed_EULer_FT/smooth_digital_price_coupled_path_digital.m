function [Pf,Pc] = smooth_digital_price_coupled_path_digital(Nsteps,Ms)
[x, w] = GaussLaguerre(32,0);
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


X_1l_f=zeros(length(x),Ms);
for i=1:32
X_1l_f(i,:)=stock_price_trajectory_1D_heston(bar_z_f-x(i),y2sf,y1f(:,1),y1f(:,2:2*Nsteps),2*Nsteps,Ms);
end

QoI_left_f=w'*(payoff_digital(X_1l_f).*1/(sqrt(2 *pi)).*exp(-(bar_z_f-x).^2/2).*exp(x));

X_1r_f=zeros(length(x),Ms);
for i=1:32
X_1r_f(i,:)=stock_price_trajectory_1D_heston(bar_z_f+x(i),y2sf,y1f(:,1),y1f(:,2:2*Nsteps),2*Nsteps,Ms);
end

QoI_right_f=w'*(payoff_digital(X_1r_f)*(1/(sqrt(2 *pi))).*exp(-(bar_z_f+x).^2/2).*exp(x));

Pf=QoI_left_f+QoI_right_f;



bar_z_c=newtons_method(y2c(:,1)',y2sc,y1c(:,1),y1c(:,2:Nsteps),Nsteps,eps,Ms);



X_1l_c=zeros(length(x),Ms);
for i=1:32
X_1l_c(i,:)=stock_price_trajectory_1D_heston(bar_z_c-x(i),y2sc,y1c(:,1),y1c(:,2:Nsteps),Nsteps,Ms);
end

QoI_left_c=w'*(payoff_digital(X_1l_c).*1/(sqrt(2 *pi)).*exp(-(bar_z_c-x).^2/2).*exp(x));

X_1r_c=zeros(length(x),Ms);
for i=1:32
X_1r_c(i,:)=stock_price_trajectory_1D_heston(bar_z_c+x(i),y2sc,y1c(:,1),y1c(:,2:Nsteps),Nsteps,Ms);
end

QoI_right_c=w'*(payoff_digital(X_1r_c)*(1/(sqrt(2 *pi))).*exp(-(bar_z_c+x).^2/2).*exp(x));

Pc=QoI_left_c+QoI_right_c;
end

