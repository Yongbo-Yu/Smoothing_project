function [M_comb]=delta_to_comb(I)
    d=size(I,2);
    S_0 = dec2bin(0:(2^d)-1)-'0';
    S_0 = -1.0*S_0;
    S = dec2bin(0:(2^d)-1)-'0';
    S = -1.0*S;
    S(S>-0.5)=1;
    
    M_comb=zeros(size(I,2),size(I,2));
    for i=1:1:size(I,1)
        idx=I(i,:);
        for s=1:1:size(S_0,1),
        	idx_new=idx+S_0(s,:);
            if all(idx_new>0)==1  
                [tf,index]=ismember(idx_new,I,'rows');
                M_comb(i,index)=prod(S(s,:));
            end
        end
    end
end