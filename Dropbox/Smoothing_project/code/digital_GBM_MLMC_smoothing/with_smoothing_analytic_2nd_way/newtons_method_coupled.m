function [x0f,x0c] = newtons_method_coupled(x0,y,Nsteps,eps,Ms)
        x0f=x0;
        x0c=x0;
        deltaf = dx_coupled(x0,y,Nsteps,Ms);
        while deltaf > eps
            [P_value_f,dP_f,P_value_c,dP_c]=f_coupled(x0f,y,Nsteps,Ms);
            x0f = x0f - 0.1*P_value_f./dP_f;
            deltaf= dx_coupled(x0f,y,Nsteps,Ms);
        end
        [deltaf,deltac] = dx_coupled(x0,y,Nsteps,Ms);
         while deltac > eps
            [P_value_f,dP_f,P_value_c,dP_c]=f_coupled(x0c,y,Nsteps,Ms);
            x0c = x0c - 0.1*P_value_c./dP_c;
            [deltaf,deltac]= dx_coupled(x0c,y,Nsteps,Ms);
        end
end