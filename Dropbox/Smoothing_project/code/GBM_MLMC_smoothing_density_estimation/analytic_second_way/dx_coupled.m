function [delta_f,delta_c] = dx_coupled(x,y,Nsteps,Ms)
 [Pf,dPyf,Pc,dPyc] = f_coupled(x,y,Nsteps,Ms);
 delta_f= abs(0-Pf);
 delta_c= abs(0-Pc);
end