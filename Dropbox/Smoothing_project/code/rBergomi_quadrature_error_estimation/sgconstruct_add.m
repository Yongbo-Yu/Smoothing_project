function [mapObj,I,I_delta,C,knots_lvls,w]=sgconstruct_add(sg_struct,idx)
    % generate Smolyak grid with level w
    N=sg_struct.N;
    w=sg_struct.w;
    func=sg_struct.func;
    X=sg_struct.X;
    knots_lvls=sg_struct.knots_lvls;
    
    I_delta=sg_struct.I_delta;
    I_delta=[I_delta;idx];
    
    %I_adm=admissibleset(I);
    E_adm=[];W_adm=[];
    %for k=1:1:size(I_adm,1)
    %    
    %end
    %P_adm=E_adm./W_adm;
    %[P_sort,I_sort]=sortrows(P_adm);
    %idx=I_adm(I_sort(1));
    %I_delta=[I;idx];
    %smolyakset=@(i) sum(i-1);
    %I_delta=multiidx_gen(N,smolyakset,w,1);
    
    w_new=max(I_delta(:));
    w_new
    % compute normalized CC knots
    if w_new>w
        w=w_new;
        sg_struct.w=w_new;
        knots_lvls=zeros(w+1,i2m(w+1),3);
        for l=1:1:(w+1)
            knots=knots_CC(i2m(l),0,1);
            knots=repmat(knots,N,1)';
            knots=repmat(X(1,:),size(knots,1),1)+knots.*(repmat(X(2,:),size(knots,1),1)-repmat(X(1,:),size(knots,1),1));
            knots_lvls(l,1:i2m(l),:)=knots;
        end
    end

    % scale CC knots wrt [a,b]^N, given by ab

    % index set for interpolation with coefficients C
    [I,C]=index_set_I(I_delta);

    % pre-compute function values
    mapObj = containers.Map;
    mapObj=y_values(mapObj,func,I,knots_lvls);

end