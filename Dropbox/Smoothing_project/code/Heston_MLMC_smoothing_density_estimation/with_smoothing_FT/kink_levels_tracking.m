Ms=10^4;
Nsteps=[2 4 8 16 32 64 128 256 512];

[x, w] = GaussLaguerre(32,0);
eps=10^-10; 
difference=zeros(1,length(Nsteps));

parfor i=1:length(Nsteps)
    i
        means = zeros(4*Nsteps(i)-1,1);
        covariance= eye(4*Nsteps(i)-1);
        y =mvnrnd(means, covariance,Ms);

        y1f=y(:,1:2*Nsteps(i));
        y2f=zeros(Ms,2*Nsteps(i));
        y2f(:,1)=zeros(Ms,1);
        y2f(:,2:2*Nsteps(i))=y(:,2*Nsteps(i)+1:4*Nsteps(i)-1);
        y2sf=y2f(:,2:2*Nsteps(i));
        
        y1c=y1f(:,1:Nsteps(i));
        y2c=y2f(:,1:Nsteps(i));
        y2sc=y2c(:,2:Nsteps(i));
    
    
        bar_z_f=newtons_method(y2f(:,1)',y2sf,y1f(:,1),y1f(:,2:2*Nsteps(i)),2*Nsteps(i),eps,Ms);

        bar_z_c=newtons_method(y2c(:,1)',y2sc,y1c(:,1),y1c(:,2:Nsteps(i)),Nsteps(i),eps,Ms);
       
     
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
