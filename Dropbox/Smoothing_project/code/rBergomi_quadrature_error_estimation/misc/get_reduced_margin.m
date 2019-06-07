function [I_reduced_margin]=get_reduced_margin(I)
    d=size(I,2);

    I_reduced_margin=[];
    e_d=eye(d);
    for i=1:1:size(I,1)
        idx=I(i,:);
        include=1;
        for j=1:1:size(e_d,1)
            idx_e=idx+e_d(j,:);
            if ismember(idx_e,I,'rows')
                include=0;
                break;
            end
        end
        if include
            I_reduced_margin=[I_reduced_margin;idx];
        end
    end
end