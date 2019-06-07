function [a, b] = coefherm(n)
if (n <= 1),  disp(' n deve essere maggiore di 1 '); return; end
a=zeros(n,1); b=zeros(n,1); b(1)=sqrt(4.*atan(1.));
for k=2:n, b(k)=0.5*(k-1); end

