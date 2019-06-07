function [evalObj,count]=y_values(sg_struct)
    evalObj=sg_struct.evalObj;
    func=sg_struct.func;
    I=sg_struct.I;
    knots=sg_struct.knots_lvl;
    
    count=zeros(size(I,1),1);
    for i=1:1:size(I,1) 
        idx=I(i,:);
        M_i=i2m(idx,sg_struct.scheme);
        [M_i_new,M_sort]=sort(M_i);
        TP_set=@(i) i./M_i_new(1:length(i));
        M_idx_new=multiidx_gen(size(M_i,2),TP_set,1,1);
        M_idx=zeros(size(M_idx_new));
        M_idx(:,M_sort)=M_idx_new;
        for j=1:size(M_idx,1)
            x_j=zeros(size(idx));
            x_j_str=[];
            for n=1:1:size(I,2)
                x_j(n)=knots(n,idx(n),M_idx(j,n));
                x_j_str=[x_j_str,sprintf('%.14f',x_j(n)),','];
                if n==size(I,2)
                   x_j_str=[x_j_str,sprintf('%.14f',x_j(n))];
                end
            end
            if ~evalObj.isKey(x_j_str)
                for n=1:1:sg_struct.N
                    if sg_struct.scheme(n)=='C'  
                        x_j(n)=sg_struct.scheme_detail{n}(1)+(sg_struct.scheme_detail{n}(2)-sg_struct.scheme_detail{n}(1))*x_j(n);
                    end
                    if sg_struct.scheme(n)=='H'  
                        x_j(n)=sg_struct.scheme_detail{n}(1)+sg_struct.scheme_detail{n}(2)*x_j(n);
                    end
                end
                evalObj(x_j_str)=func(x_j);
                count(i)=count(i)+1;
            end
        end
    end
end