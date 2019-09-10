function x0 = newtons_method(x0,y,yv1,yv,Nsteps,eps,Ms)

        delta = dx(x0,y,yv1,yv,Nsteps,Ms);
        while delta > eps
        
            [P_value,dP]=f(x0,y,yv1,yv,Nsteps,Ms);
            x0 = x0 - 0.1*P_value./dP;
          
            delta = dx(x0,y,yv1,yv,Nsteps,Ms);
        end
end

  