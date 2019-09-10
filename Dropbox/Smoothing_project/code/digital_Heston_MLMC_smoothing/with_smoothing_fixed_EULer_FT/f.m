function [Py,dPy] = f(y1,y,yv1,yv,Nsteps,Ms)

    T=1;
    dt=T/Nsteps;
    K=100;
    S0=100;
    rho=-0.9;
        
    [X_1,dbb,V]=stock_price_trajectory_1D_heston(y1,y,yv1,yv,Nsteps,Ms);
    
    y1s= rho *yv1' + sqrt(1-rho^2)*y1;
    
    fi=1+(sqrt(V(1:Nsteps,:))/sqrt(T)).*y1s*(dt)+sqrt(V(1:Nsteps,:)).*dbb;
    product=prod(fi,1);
    s=sqrt(V(1:Nsteps,:))./fi;
    summation=sum(s);
    Py=product-(K/S0);
   
    dPy=(1/sqrt(T))*(dt).*product.*summation;
     
end
        