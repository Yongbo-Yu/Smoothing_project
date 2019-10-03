function [X_1f,dbbf,X_1c,dbbc]=stock_price_trajectory_1D_BS_coupled(y1,y,Nsteps,Ms)
        T=1;
        dtc=T/Nsteps;
        dtf=T/(2*Nsteps);
        S0=100;
        sigma=0.2;
        
        bb=brownian_increments(y1,y,2*Nsteps,Ms);
        dWf=zeros(2*Nsteps,Ms);
        dWc=zeros(Nsteps,Ms);
        j=1;
        for i=1:1:size(bb,2)-1
            dWf(i,:)= bb(:,i+1)-bb(:,i) ;
            if mod(i,2)==0
               dWc(j,:)= dWf(i,:)+dWf(i-1,:);
               j=j+1;
            end
        end
      
        
        dbbf=dWf-(dtf/sqrt(T))*y1; 
        dbbc=dWc-(dtc/sqrt(T))*y1;
        Xf=zeros(2*Nsteps+1,Ms);
        Xf(1,:)=S0*ones(1,Ms);
        for n =2:1:2*Nsteps+1
            Xf(n,:)=Xf(n-1,:).*(1+sigma*dWf(n-1,:));
        end
        
        Xc=zeros(Nsteps+1,Ms);
        Xc(1,:)=S0*ones(1,Ms);
        for n =2:1:Nsteps+1
            Xc(n,:)=Xc(n-1,:).*(1+sigma*dWc(n-1,:));
        end
         
        X_1f=Xf(end,:);
        X_1c=Xc(end,:);
 end
        

