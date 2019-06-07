% ===========================================================
%  knots and weights of the Clenshaw-Curtis quadrature formula
% ===========================================================
%
%   [x,w] = knots_CC(nn,x_a,x_b)
%   nn: number of knots
%   x_a,x_b: interval containing the nodes
%   x: knots   [row vector]
%   w: weights [row vector]

function [w] = scheme_CC_weights(nn,x_a,x_b)

if nn==1
    x=(x_a+x_b)/2; w=1;

elseif mod(nn,2)==0
    error('error in knots_CC: Clenshaw-Curtis formula \n use only odd number of points')

else
n=nn-1;

N=[1:2:n-1]'; l=length(N); m=n-l; K=[0:m-1]';
g0=-ones(n,1); g0(1+l)=g0(1+l)+n; g0(1+m)=g0(1+m)+n;
g=g0/(n^2-1+mod(n,2));
end_N=length(N);
v0=[2./N./(N-2); 1/N(end_N); zeros(m,1)];
end_v0=length(v0);
v2=-v0(1:end_v0-1)-v0(end_v0:-1:2);
wcc=ifft(v2+g);
w=[wcc;wcc(1,1)]'/2;
x=(cos([0:n]*pi/n));
x = (x_b-x_a)/2*x + (x_a+x_b)/2;
end
