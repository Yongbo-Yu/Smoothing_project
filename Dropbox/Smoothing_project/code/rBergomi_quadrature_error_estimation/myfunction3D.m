function [f]=myfunction3D(x),
    f=sum(x.^8.0)/(1.0+0.1*sum(x.^2));
end
