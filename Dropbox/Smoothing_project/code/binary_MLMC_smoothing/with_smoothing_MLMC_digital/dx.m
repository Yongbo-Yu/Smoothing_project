function delta = dx(x,y,Nsteps,Ms)
 [P1,dP1]=f(x,y,Nsteps,Ms);
 delta= abs(0-P1);
end