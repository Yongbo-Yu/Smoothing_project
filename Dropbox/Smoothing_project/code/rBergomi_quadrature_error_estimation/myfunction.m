function [f]=myfunction(x),
    f=zeros(2,1);
    f(1)=1.0./(1.0+0.8*sum(exp(x)));
    f(2)=1.0./(0.5+0.01*sum(exp(x)));
end
