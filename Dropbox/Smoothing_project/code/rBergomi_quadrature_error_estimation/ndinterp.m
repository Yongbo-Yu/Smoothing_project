function [Y]=ndinterp(sg_struct,X)
    mapObj=sg_struct.mapObj;
    I=sg_struct.I;
    %I=sg_struct.I_delta;
    C=sg_struct.C;
    knots=sg_struct.knots_lvls;

    Y=zeros(size(X,1),1);
    for i=1:1:size(I,1)
        idx=I(i,:);
        M_idx=i2m(idx,'CC');
        [M1_idx,M2_idx]=ndgrid(1:M_idx(1),1:M_idx(2));
        Y_i=zeros(size(X,1),1);
        for j=1:numel(M1_idx)  
            x_j=[knots(idx(1),M1_idx(j),1),knots(idx(2),M2_idx(j),2)];
            x_j_str=[sprintf('%.14f',x_j(1)),',',sprintf('%.14f',x_j(2))];
            y_j=mapObj(x_j_str);
            for k=1:1:size(X,1),
                x_k=X(k,:);
                lb=1;
                for d=1:1:length(x_j)
                    lb=lb*lagrangeb(knots(idx(d),1:M_idx(d),d),x_k(d),x_j(d));
                end
                Y_i(k)=Y_i(k)+y_j*lb;
            end
            
        end
        Y=Y+C(i).*Y_i;
    end
end