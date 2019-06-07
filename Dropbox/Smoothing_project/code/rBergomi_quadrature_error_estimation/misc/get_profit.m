function [I_margin]=get_profit(I,profit)

    I_profit=zeros(size(I,1),1);
    for i=1:1:size(I,1)
        idx=I(i,:);
        I_profit(i)=rank(idx);
    end
end