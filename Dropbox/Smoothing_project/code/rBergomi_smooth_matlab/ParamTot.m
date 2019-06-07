classdef ParamTot
    
% Public    

   properties
       H;
       eta;
       rho;
       T;
       K;
       xi;
     
   end
   methods
      
      function obj =  ParamTot(h, e, r, t, k,  x)
            if nargin == 0
                 obj.xi = 0.0;
            else
                
                  obj.H=h;
                  obj.eta=e;
                  obj.rho=r;
                  obj.T=t;
                  obj.K=k;
                  obj.xi=x;                 
            end
      end
      
   end
end