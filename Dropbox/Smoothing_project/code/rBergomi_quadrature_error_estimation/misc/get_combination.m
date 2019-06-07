function [C_comb]=get_combination(I_delta)
    d=size(I_delta,2);

    M_comb=delta_to_comb(I_delta);
    C_comb=sum(M_comb,1)';

    %D_comb=zeros(size(I_delta,1),1);
    %for i=1:1:size(I_delta,1)
    %   idx=I_delta(i,:);
    %    D_comb(i)=c_coeff(I_delta,idx);
    %end

end