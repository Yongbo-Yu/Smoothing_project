function [Y_out,Z,Y]=sgquad(sg_struct)
    evalObj=sg_struct.evalObj;
    I=sg_struct.I;
    C=sg_struct.C;
    knots=sg_struct.knots_lvl;
    weights_lvl=sg_struct.weights_lvl;
   
    d=sg_struct.N+sg_struct.d_ind;
    Y=zeros(size(I,1),2);
    Z=zeros(size(I,1),2);
    Y_out=zeros(1,2);
    M=delta_to_comb(I);
    for i=1:1:size(I,1)
        idx=I(i,:);
        M_i=i2m(idx,sg_struct.scheme);
        [M_i_new,M_sort]=sort(M_i);
        TP_set=@(i) i./M_i_new(1:length(i));
        M_idx_new=multiidx_gen(size(M_i,2),TP_set,1,1);
        M_idx=zeros(size(M_idx_new));
        M_idx(:,M_sort)=M_idx_new;
        
        Y_i=zeros(1,2);
        for j=1:1:size(M_idx,1)  
            x_j=zeros(size(idx));
            x_j_str=[];
            for n=1:1:size(I,2)
                x_j(n)=knots(n,idx(n),M_idx(j,n));
                x_j_str=[x_j_str,sprintf('%.14f',x_j(n)),','];
                if n==size(I,2)
                   x_j_str=[x_j_str,sprintf('%.14f',x_j(n))];
                end
            end
            y_j=evalObj(x_j_str);

            lb=1;
            for d=1:1:length(x_j)
            	if sg_struct.scheme(d)=='C'
                	lb=lb*weights_lvl(d,idx(d),M_idx(j,d));
                end
                if sg_struct.scheme(d)=='H'
                	lb=lb*weights_lvl(d,idx(d),M_idx(j,d));
                end
            end
            Y_i=Y_i+y_j'*lb;
        end
        Y(i,:)=Y_i;
        Y_out=Y_out+C(i)*Y_i;  

    end

    for i=1:1:size(I,1)
        for j=1:1:size(I,1)
        	Z(i,:)=Z(i,:)+M(i,j).*Y(j,:);
        end
    end
    
    %Y_out=zeros(size(X,1),2);
    %size(Z)
    %size(C')
    %C=repmat(C,2,2);
    %size(C)
    %C.*Z'
    %Y_out=Y_out+C*Z;
    %size(Y_out)
end