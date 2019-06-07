function multi_idx = multiidx_gen(L,rule,w,base,multiidx,multi_idx)

if nargin==3
      base = 0;
      multiidx=[];
      multi_idx=[];
elseif nargin==4
      multiidx=[];
      multi_idx=[];
end

if length(multiidx)~=L   
      i=base;
      while rule([multiidx, i]) <= w
            multi_idx = multiidx_gen(L,rule,w,base,[multiidx, i],multi_idx);
            i=i+1;
      end
else  
      multi_idx=[multi_idx; multiidx];
end