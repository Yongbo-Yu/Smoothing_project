classdef RBergomi
    
% Public    

   properties
       nDFT;
       M;
       N;
       par;
     
   end
   methods
      
      function obj =  RBergomi(x, HIn, e, r, t,k, NIn,  MIn)
            if nargin == 0
                 obj.N = 0;
                 obj.M = 0;
                 obj.nDFT = 0;
                 obj.par = ParamTot(); 
                  
            else
                
                 obj.N = NIn;
                 obj.nDFT = 2 * obj.N - 1;
                 obj.M = MIn;
                 obj.par = ParamTot(HIn, e, r, t, k, x);
                  
                  
            end
      end
      
      function Wtilde = updateWtilde(obj, W1, W1perp, H)
         Gamma=getGamma(obj,H);
         s2H = sqrt(2.0 * H);
         rhoH = s2H/(H + 0.5);
         W1hat = linearComb(obj,rhoH/s2H, W1, sqrt(1.0 - rhoH * rhoH)/s2H, W1perp);
         xC=copyToComplex(obj,W1);
         yC=copyToComplex(obj,Gamma);
          % Doing forward transformation
         xHat = fft(xC);
         yHat = fft(yC);
%          multiply xHat and yHat and save in zHat
	     zHat=complexMult(obj,xHat, yHat);
         zC = ifft( zHat);          
%        read out the real part, re-scale by 1/nDFT
         Y2=zeros(obj.N,1);
         Y2=copyToReal(obj,Y2,zC);         
         Y2=scaleVector(obj,Y2, 1.0/obj.nDFT); 
%          
%     Wtilde = (Y2 + W1hat) * sqrt(2*H) * dt^H ??
	     Wtilde = linearComb(obj,sqrt(2.0 * H) * ((1.0 / obj.N)^ H), Y2,sqrt(2.0 * H) * ((1.0 / obj.N)^H), W1hat);
                              
      end
      
      function WtildeScaled = scaleWtilde(obj,Wtilde,T, H)
             WtildeScaled=zeros(obj.N,1);
         for i=1:obj.N 
             WtildeScaled(i) = (T^H) * Wtilde(i);
         end
       
      end
      
      function ZScaled = scaleZ(obj, Z,sdt)
         ZScaled=zeros(obj.N,1);
         for i=1:obj.N 
             ZScaled(i) = sdt * Z(i);
         end
      end
        
      function v= updateV(obj,  WtildeScaled,  h, e,  dt)
         v=zeros(obj.N,1); 
         v(1) = obj.par.xi;
         for i=2:obj.N    
            v(i) = obj.par.xi * exp(e * WtildeScaled(i - 1) - 0.5 * e * e* (((i - 2) * dt)^(2 * h)));          
         end
      end
      function y= scaleVector(obj, x,s)
            y=zeros(length(x),1);
            for i=1:length(x)
                y(i)=x(i)*s;
            end
      end
            
      function Gamma= getGamma(obj, H)
         Gamma=zeros(obj.N,1); 
         alpha = H - 0.5;
	     Gamma(1) = 0.0;
         for i=2:obj.N    
            Gamma(i) = ( (i)^(alpha + 1.0) - ((i-1)^(alpha + 1.0))) /(alpha + 1.0);         
         end
      end
         
      function xc= copyToComplex(obj, x)
       
         xc=zeros(length(x),1);
         for i=1:length(x)  
             	xc(i) = complex(x(i),0.0);
         end
         for i=length(x)+1: obj.nDFT 
             	xc(i) = complex(0.0,0.0);
               
         end
       
      end 
      
      function x= copyToReal(obj,x, xc)
         for i=1:length(x)  
             	x(i) = real(xc(i));
         end
       
       
      end 
      
      function z= complexMult(obj, x,y)
         z=zeros(obj.nDFT,1);
         for i=1:obj.nDFT 
             	z(i)=fftw_c_mult(obj,x(i), y(i));
         end
       
       
      end 

      function c= fftw_c_mult(obj, a,b)
         c_real = real(a) * real(b) - imag(a) * imag(b);
	     c_imag = real(a) * imag(b) + imag(a) * real(b);
         c=complex(c_real, c_imag);
      end 
      
    
      
      function QoI= intVdt(obj,v,dt)
         QoI= dt * sum(v);
      end
      function IsvdW= intRootVdW(obj, v,  W1,  sdt)
          IsvdW = 0.0;
            for i=1:length(v)
		      IsvdW = IsvdW + sqrt(v(i)) * sdt * W1(i);
            end
      end 
      
      function z=linearComb(obj,a,  x, b, y)
           z=zeros(length(x),1);
         
           for i=1:length(x)
           
               z(i) = a*x(i) + b*y(i);
           end
      end
      
      function BS_price= BS_call_price( obj,S0, K, tau, sigma,  r)
             d1 = (log(S0/K) + (r+0.5*sigma*sigma)*tau)/(sigma*sqrt(tau));
             d2 = d1 - sigma*sqrt(tau);
             BS_price=Phi(obj,d1)*S0 - Phi(obj,d2)*K*exp(-r*tau);
          
      end
       function Phi_price= Phi( obj,value)
           
           Phi_price=0.5 * erfc(-value * (1/sqrt(2)));
             
          
       end
      
      
      
      function price= updatePayoff(obj, W1, W1perp)
          dt = obj.par.T / obj.N;
          sdt = sqrt(dt);
	      
          Wtilde=updateWtilde(obj, W1, W1perp, obj.par.H);
         
          WtildeScaled=scaleWtilde(obj,Wtilde, obj.par.T, obj.par.H);
      
          v=updateV(obj,WtildeScaled, obj.par.H, obj.par.eta, dt);
          
          Ivdt = intVdt(obj,v, dt);
          
	      IsvdW = intRootVdW(obj,v, W1, sdt);
          
	      BS_vol = sqrt((1.0 -  obj.par.rho* obj.par.rho) * Ivdt);
          BS_spot = exp( - 0.5 *  obj.par.rho* obj.par.rho * Ivdt +  obj.par.rho * IsvdW );
          price=BS_call_price(obj,BS_spot,  obj.par.K, 1.0, BS_vol,0);
      end 
      
      function payoff= ComputePayoffRT_single(obj,W1,  W1perp)
          payoff = updatePayoff(obj,  W1, W1perp);	    
      end 
      
  

   end
end