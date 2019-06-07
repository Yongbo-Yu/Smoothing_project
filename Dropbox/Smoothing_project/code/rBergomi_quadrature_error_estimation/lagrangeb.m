function [l]=lagrangeb(knots,x,x_j)
l=1;
for k=1:1:length(knots)
    if ~(x_j-knots(k))<10^(-10)
        l=l*(x-knots(k))/(x_j-knots(k)); 
    end
end