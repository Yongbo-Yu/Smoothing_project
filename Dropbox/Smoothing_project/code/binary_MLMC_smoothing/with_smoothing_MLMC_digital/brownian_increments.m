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
            i=i_min;
            
            l=0;
            r=h;
            for j=1:1:j_max
               
%                 h
%                 i
%                 r
%                 l
         
                
                a=((t(r+1)-t(i+1))* bb(:,l+1)+(t(i+1)-t(l+1))*bb(:,r+1))/(t(r+1)-t(l+1));
                
                b=sqrt((t(i+1)-t(l+1))*(t(r+1)-t(i+1))/(t(r+1)-t(l+1)));
              
                
                bb(:,i+1)=a+b*y(:,i);
                i=i+h;
                l=l+h;
                r=r+h;
             end
             j_max=2*j_max;
             h=i_min; 
           
        end
end