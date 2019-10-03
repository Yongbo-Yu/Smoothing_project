function [X_1,dbb]=stock_price_trajectory_1D_BS(y1,y,Nsteps,Ms)
        T=1;
        dt=T/Nsteps;
        S0=1;
        sigma=0.2;
        
        bb=brownian_increments(y1,y,Nsteps,Ms);
        dW=zeros(Nsteps,Ms);
     
        for i=1:1:size(bb,2)-1
            dW(i,:)= bb(:,i+1)-bb(:,i) ;
        end
        dbb=dW-(dt/sqrt(T))*y1; 
        X=zeros(Nsteps+1,Ms);
        X(1,:)=S0*ones(1,Ms);
        for n =2:1:Nsteps+1
            X(n,:)=X(n-1,:).*(1+sigma*dW(n-1,:));
        end
         
        X_1=X(end,:);
 end
        

