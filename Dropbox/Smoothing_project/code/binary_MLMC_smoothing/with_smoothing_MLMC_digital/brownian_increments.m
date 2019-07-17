function [bb]=brownian_increments(y1,y,Nsteps,Ms)
        T=1;
        d=log2(Nsteps);
        
        t=linspace(0, T, Nsteps+1); 
        h=Nsteps;
        j_max=1;
        bb= zeros(Ms,Nsteps+1);
        
        bb(:,h+1)=sqrt(T)*y1;
       
   
        for k=1:1:d
            i_min=fix(h/2);
            i=i_min+1;
            
            l=1;
            r=h+1;
            for j=1:1:j_max
               
%                 h
%                 i
%                 r
%                 l
         
                
                a=((t(r)-t(i))* bb(:,l)+(t(i)-t(l))*bb(:,r))/(t(r)-t(l));
                
                b=sqrt((t(i)-t(l))*(t(r)-t(i))/(t(r)-t(l)));
              
                
                bb(:,i)=a+b*y(:,i-1);
                i=i+h;
                l=l+h;
                r=r+h;
             end
             j_max=2*j_max;
             h=i_min; 
           
        end
end