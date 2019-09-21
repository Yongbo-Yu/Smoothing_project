function x0 = newtons_method(x0,y,Nsteps,eps,Ms)
        delta = dx(x0,y,Nsteps,Ms);
        while delta > eps
            [P_value,dP]=f(x0,y,Nsteps,Ms);
            x0 = x0 - 0.1*P_value./dP;
            delta = dx(x0,y,Nsteps,Ms);
        end
end