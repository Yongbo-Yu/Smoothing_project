function [evalObj,knots_lvl,weights_lvl]=sgbuild(sg_struct)
    max_lvl=max(max(sg_struct.I));
    knots_lvl=zeros(sg_struct.N,max_lvl,i2m(max_lvl,sg_struct.scheme));
    weights_lvl=zeros(sg_struct.N,max_lvl,i2m(max_lvl,sg_struct.scheme));
    for i=1:1:sg_struct.N
        if sg_struct.scheme(i)=='C'
            for j=1:1:max_lvl
                knots_lvl(i,j,1:i2m(j,'C'))=scheme_CC(i2m(j,'C'),0,1);%;
                weights_lvl(i,j,1:i2m(j,'C'))=scheme_CC_weights(i2m(j,'C'),0,1);
            end
        end
        if sg_struct.scheme(i)=='H'
            for j=1:1:max_lvl
                [x,w]=scheme_GH(i2m(j,'H'),0,1);
                knots_lvl(i,j,1:i2m(j,'H'))=x;
                weights_lvl(i,j,1:i2m(j,'H'))=w;
            end
        end
    end
    sg_struct.knots_lvl=knots_lvl;
    sg_struct.weights_lvl=weights_lvl;

    C=get_combination(sg_struct.I);
    evalObj = containers.Map;
    sg_struct.evalObj=evalObj;
    [evalObj,count_delta]=y_values(sg_struct);
    sg_struct.evalObj=evalObj;
end