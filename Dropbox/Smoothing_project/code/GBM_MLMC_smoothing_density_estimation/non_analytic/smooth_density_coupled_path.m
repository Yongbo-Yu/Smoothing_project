function [Pf,Pc] = smooth_density_coupled_path(Nsteps,Ms)
[x, w] = GaussLaguerre(32,0);
mean = zeros(2*Nsteps-1,1);
covariance= eye(2*Nsteps-1);
y =mvnrnd(mean, covariance,Ms);

y_1f=y(:,1:2*Nsteps-1);
y_1c=y_1f(:,1:Nsteps-1);


eps=10^-10;   
bar_z_f=newtons_method(0,y_1f,2*Nsteps,eps,Ms);
X_1l_f=zeros(length(x),Ms);
for i=1:32
X_1l_f(i,:)=stock_price_trajectory_1D_BS(bar_z_f-x(i),y_1f,2*Nsteps,Ms);
end
X_1l_f =round( X_1l_f,2);
QoI_left_f=w'*(payoff_den(X_1l_f).*1/(sqrt(2 *pi)).*exp(-(bar_z_f-x).^2/2).*exp(x));

X_1r_f=zeros(length(x),Ms);
for i=1:32
X_1r_f(i,:)=stock_price_trajectory_1D_BS(bar_z_f+x(i),y_1f,2*Nsteps,Ms);
end
X_1r_f =round( X_1r_f,2);

QoI_right_f=w'*(payoff_den(X_1r_f)*(1/(sqrt(2 *pi))).*exp(-(bar_z_f+x).^2/2).*exp(x));

Pf=QoI_left_f+QoI_right_f;

bar_z_c=newtons_method(0,y_1c,Nsteps,eps,Ms);

X_1l_c=zeros(length(x),Ms);
for i=1:32
X_1l_c(i,:)=stock_price_trajectory_1D_BS(bar_z_c-x(i),y_1c,Nsteps,Ms);
end
X_1l_c =round( X_1l_c,2);
QoI_left_c=w'*(payoff_den(X_1l_c).*1/(sqrt(2 *pi)).*exp(-(bar_z_c-x).^2/2).*exp(x));

X_1r_c=zeros(length(x),Ms);
for i=1:32
X_1r_c(i,:)=stock_price_trajectory_1D_BS(bar_z_c+x(i),y_1c,Nsteps,Ms);
end
X_1r_c =round( X_1r_c,2);
QoI_right_c=w'*(payoff_den(X_1r_c)*(1/(sqrt(2 *pi))).*exp(-(bar_z_c+x).^2/2).*exp(x));

Pc=QoI_left_c+QoI_right_c;
end
