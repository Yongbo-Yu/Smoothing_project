function [mapObj,I,I_delta,count_delta,C,knots_lvls]=sgconstruct_all(N,w,func,X)
    % generate Smolyak grid with level w
    smolyakset=@(i) sum(i-1);%max(i-1);%sum(i-1); %
    I_delta=multiidx_gen(N,smolyakset,w,1);

    % compute normalized CC knots
    knots_lvls=zeros(w+1,i2m(w+1,'CC'),N);
    for l=1:1:(w+1)
        knots=knots_CC(i2m(l,'CC'),0,1);
        knots=repmat(knots,N,1)';
        knots=repmat(X(1,:),size(knots,1),1)+knots.*(repmat(X(2,:),size(knots,1),1)-repmat(X(1,:),size(knots,1),1));
        knots_lvls(l,1:i2m(l,'CC'),:)=knots;
    end

    % scale CC knots wrt [a,b]^N, given by ab

    % index set for interpolation with coefficients C
    [I,C]=index_set_I(I_delta);

    % pre-compute function values
    mapObj = containers.Map;
    [mapObj,count_delta]=y_values(mapObj,func,I_delta,knots_lvls);
    %mapObj=y_values(mapObj,func,I,knots_lvls);
end