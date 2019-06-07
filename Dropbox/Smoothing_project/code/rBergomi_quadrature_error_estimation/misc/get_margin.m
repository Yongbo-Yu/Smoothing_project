function [I_margin]=get_margin(I)
    d=size(I,2);

    I_margin=[];
    e_vec=eye(d);
    for i=1:1:size(I,1)
        idx=I(i,:);
        for j=1:1:size(e_vec,1)
            idx_e=idx+e_vec(j,:);
            if ~ismember(idx_e,I,'rows')
                if isempty(I_margin) || ~ismember(idx_e,I_margin,'rows')
                    I_margin=[I_margin;idx_e];
                end
            end
        end
    end
end