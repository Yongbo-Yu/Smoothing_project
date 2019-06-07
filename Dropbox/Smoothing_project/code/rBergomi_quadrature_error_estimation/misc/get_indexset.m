function [I,I_add]=get_indexset(sg_struct,adm,profit,num_add)

    d=sg_struct.N+sg_struct.d_ind;
    I=sg_struct.I;
    
    if isempty(I),
            I=ones(1,d);
    end
        
    keepexplore=1;
    added=0;
    I_margin_prev=[];
    while keepexplore
        I_add=[];
        [I_margin]=get_margin(I);
        %I_profit=profit(I_margin);
        
        I_adm=[];P_adm=[];
        for i=1:1:size(I_margin,1)
            idx=I_margin(i,:);
            if adm(idx)
                I_adm=[I_adm;idx];
                P_adm=[P_adm;profit(idx)];
            end
        end
        [P_adm,I_sort]=sort(P_adm,'descend');
        I_adm=I_adm(I_sort,:);
        for j=1:1:size(I_adm,1)
            idx=I_adm(j,:);
            if added<num_add || num_add==-1
                if ~ismember(idx,I,'rows')
                    I_add=[I_add;I_adm(j,:)];
                    added=added+1;
                end
            end
        end
        I=[I;I_add];

        if isempty(I_add)
            keepexplore=0;
        else
            I_margin_prev=I_margin;
        end
    end
end