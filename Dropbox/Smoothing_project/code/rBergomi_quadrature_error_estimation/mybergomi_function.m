function [f]=mybergomi_function(x),
    xi=0.235^2;
    HIn= 0.07;
    e=1.9;
    r=-0.9;
    t=1.0;
    k=1.0;
    NIn=8;
    MIn=1;
    z=RBergomi( xi,  HIn, e,  r,  t, k,  NIn, MIn);
    f=zeros(1,1);
    W1 = x(1:NIn);
    W2 = x(NIn+1:2*NIn);
	f(1)=z.ComputePayoffRT_single(W1,W2);
    
end
