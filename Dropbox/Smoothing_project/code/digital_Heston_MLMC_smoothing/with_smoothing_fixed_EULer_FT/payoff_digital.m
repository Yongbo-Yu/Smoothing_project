function [out] = payoff_digital(x)
K=100;

out=0.5*(1+sign(x-K));
end

