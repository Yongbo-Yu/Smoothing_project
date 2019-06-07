x=0.235^2;
HIn= 0.07;
e=1.9;
r=-0.9;
t=1.0;
k=1.0;
NIn=2;
MIn=1;
z=RBergomi( x,  HIn, e,  r,  t, k,  NIn, MIn);



a=zeros(10^7,1);

for m=1:10^7
    W1 = randn(NIn,1);
    W2 = randn(NIn,1);
	a(m)=z.ComputePayoffRT_single(W1,W2);
end

mean(a)
std(a)/sqrt(10^7)