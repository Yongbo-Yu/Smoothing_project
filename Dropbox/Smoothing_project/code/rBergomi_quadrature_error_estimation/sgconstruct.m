function [mapObj,I,C,knots_lvls]=sgconstruct(N,w,func,ab)
    % generate Smolyak grid with level w
    smolyakset=@(i) max(i-1);%sum(i-1);
    I_delta=multiidx_gen(N,smolyakset,w,1);

    % compute normalized CC knots
    knots_lvls=zeros(w+1,i2m(w+1),3);
    for l=1:1:(w+1)
        knots=knots_CC(i2m(l),0,1);
        knots=repmat(knots,N,1)';
        knots=repmat(ab(1,:),size(knots,1),1)+knots.*(repmat(ab(2,:),size(knots,1),1)-repmat(ab(1,:),size(knots,1),1));
        knots_lvls(l,1:i2m(l),:)=knots;
    end

    % scale CC knots wrt [a,b]^N, given by ab

    % index set for interpolation with coefficients C
    [I,C]=index_set_I(I_delta);

    % pre-compute function values
    mapObj = containers.Map;
    mapObj=y_values(mapObj,func,I,knots_lvls);

end