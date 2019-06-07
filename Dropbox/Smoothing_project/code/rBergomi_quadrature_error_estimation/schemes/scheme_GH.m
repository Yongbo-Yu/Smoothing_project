function [x,w]=scheme_GH(n,mi,sigma)

% [x,w]=zpherm(n,mi,sigma) calculates the collocation points (x) 
% and the weights (w) for the gaussian integration 
% w.r.t to the weight function 
% rho(x)=1/sqrt(2*pi*sigma) *exp( (x-mi)^2 / (2*sigma^2) ) 
% i.e. the density of a gaussian random variable 
% with mean mi and standard deviation sigma


if (n==1) 
      % the point (traslated if needed) 
      x=mi;
      % the weight is 1:
      w=1;
      return
end

% calculates the values of the recursive relation
[a,b]=coefherm(n); 

% builds the matrix
JacM=diag(a)+diag(sqrt(b(2:n)),1)+diag(sqrt(b(2:n)),-1);

% calculates points and weights from eigenvalues / eigenvectors of JacM
[w,x]=eig(JacM); 
x=diag(x); 
%scal=sqrt(pi); 
%w=w(1,:)'.^2*scal;
w=w(1,:)'.^2;
[x,ind]=sort(x); w=w(ind);

% modifies points according to mi, sigma (the weigths are unaffected)

x=mi + sqrt(2)*sigma*x;

