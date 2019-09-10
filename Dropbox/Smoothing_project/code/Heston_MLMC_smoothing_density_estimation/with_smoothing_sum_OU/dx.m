function delta = dx(x,y,yv1,yv,Nsteps,Ms)
 [P1,dP1]=f(x,y,yv1,yv,Nsteps,Ms);
 delta= abs(0-P1);
end