Ms=10^4;
Nsteps=[2 4 8 16 32 64 128 256 512];

eps=10^-10; 
difference=zeros(1,length(Nsteps));

parfor i=1:length(Nsteps)
    i
        means = zeros(2*Nsteps(i)-1,1);
covariance= eye(2*Nsteps(i)-1);
y =mvnrnd(means, covariance,Ms);

y_1f=y(:,1:2*Nsteps(i)-1);
y_1c=y_1f(:,1:Nsteps(i)-1);

 
bar_z_f=newtons_method(0,y_1f,2*Nsteps(i),eps,Ms);
bar_z_c=newtons_method(0,y_1c,Nsteps(i),eps,Ms);




difference(i)= abs(mean(bar_z_c)-mean(bar_z_f));
       
        
end
difference
% pa = polyfit(1./Nsteps,difference,1)

loglog(1./Nsteps,difference,'b')
hold on
% loglog(1./Nsteps,polyval(pa,1./Nsteps),'g')
loglog(1./Nsteps,1./Nsteps,'r')

xlabel('\Deltat'); ylabel('|E[y^{*}_{2\Deltat}]-E[y^{*}_{\Deltat}]|');
legend('|E[y^{*}_{2\Deltat}]-E[y^{*}_{\Deltat}]|','\Deltat', 'Location','northeast')
