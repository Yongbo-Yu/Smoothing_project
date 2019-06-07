function [I_margin,R_margin]=get_margin_rank(I,rank)
    d=size(I,2);

    I_margin=[];
    R_margin=[];
    
    e_vec=eye(d);
    for i=1:1:size(I,1)
        idx=I(i,:);
        for j=1:1:size(e_vec,1)
            idx_e=idx+e_vec(j,:);
            idx_e_rank=rank(idx_e);
            I_margin=[I_margin;idx_e];
            R_margin=[R_margin,idx_e_rank];
        end
    end
end