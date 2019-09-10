function [Py,dPy] = f(y1,y,Nsteps,Ms)

    T=1;
    dt=T/Nsteps;
    sigma=0.2;
    K=100;
    S0=100;
        
    [X,dbb]=stock_price_trajectory_1D_BS(y1,y,Nsteps,Ms);
    fi=1+(sigma/sqrt(T))*y1*(dt)+sigma*dbb;
    product=prod(fi,1);
    summation=sum(1./fi);
    Py=product-(K/S0);
    dPy=(sigma/sqrt(T))*(dt).*product.*summation;
end
        