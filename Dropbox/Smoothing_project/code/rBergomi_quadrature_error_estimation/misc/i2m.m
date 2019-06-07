function [m] = i2m(i,scheme)

m=zeros(size(i));
for k=1:1:size(i,1)
for n=1:1:size(i,2)
    if scheme(n)=='CC',
        if i(k,n)==1,
            m(k,n)=1;
        else,
            m(k,n)=2^(i(k,n)-1)+1;
        end
    end
    if scheme(n)=='H',
    	m(k,n)=i(k,n);
    end
end
end

end