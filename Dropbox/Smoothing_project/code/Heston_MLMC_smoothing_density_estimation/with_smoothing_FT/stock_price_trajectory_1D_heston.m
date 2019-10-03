function [X_1,dbb_s,V]=stock_price_trajectory_1D_heston(y1,y,yv1,yv,Nsteps,Ms)
        T=1;
        dt=T/Nsteps;
        S0= 1;
% %          %set1
%         rho=-0.9;
%         V0=0.04;
%         kappa= 1.0;
%         xi=0.1;
%         theta=0.0025;
%    %set2
%         rho=-0.3;
%         V0=0.09;
%         kappa= 2.7778 ;
%         xi=1;
%         theta=0.09;
  
%          set3
         rho=-0.9;
        V0=0.04;
        kappa= 0.5;
        xi=0.01;
        theta=0.04;
 
        bb=brownian_increments(y1,y,Nsteps,Ms);
        bb_v=brownian_increments(yv1,yv,Nsteps,Ms);
        dW=zeros(Nsteps,Ms);
        dW_v=zeros(Nsteps,Ms);
     
        for i=1:1:size(bb,2)-1
            dW(i,:)= bb(:,i+1)-bb(:,i) ;
        end
     
        for i=1:1:size(bb_v,2)-1
            dW_v(i,:)= bb_v(:,i+1)-bb_v(:,i) ;
        end
        
        dW_s= rho *dW_v + sqrt(1-rho^2) * dW; 
     
        y1s= rho *yv1' + sqrt(1-rho^2)*y1;
      
        dbb_s=dW_s-(dt/sqrt(T)).*y1s; 
        
      
        X=zeros(Nsteps+1,Ms);
        X(1,:)=S0*ones(1,Ms);
        
        V=zeros(Nsteps+1,Ms);
        V(1,:)=V0*ones(1,Ms);
      
        for n =2:1:Nsteps+1
            X(n,:)=X(n-1,:).*(1+sqrt(V(n-1,:)).*dW_s(n-1,:));
            V(n,:)=V(n-1,:)- kappa *dt* max(V(n-1,:),0)+ xi *sqrt(max(V(n-1,:),0)).*dW_v(n-1,:)+ kappa*theta*dt;
            V(n,:)=max(V(n,:),0);
        end
         
        X_1=X(end,:);
 end
        

