function [Pyf,dPyf,Pyc,dPyc] = f_coupled(y1,y,Nsteps,Ms)

    T=1;
    dtc=T/Nsteps;
    dtf=T/(2*Nsteps);
    sigma=0.2;
    K=1;
    S0=1;
        
    [Xf,dbbf,Xc,dbbc]=stock_price_trajectory_1D_BS_coupled(y1,y,Nsteps,Ms);
    fi_f=1+(sigma/sqrt(T))*y1*(dtf)+sigma*dbbf;
    fi_c=1+(sigma/sqrt(T))*y1*(dtc)+sigma*dbbc;
    product_f=prod(fi_f,1);
    product_c=prod(fi_c,1);
    summation_f=sum(1./fi_f);
    summation_c=sum(1./fi_c);
    Pyf=product_f-(K/S0);
    Pyc=product_c-(K/S0);
    dPyf=(sigma/sqrt(T))*(dtf).*product_f.*summation_f;
    dPyc=(sigma/sqrt(T))*(dtc).*product_c.*summation_c;
end
          