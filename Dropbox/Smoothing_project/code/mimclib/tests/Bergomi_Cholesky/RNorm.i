%module RNorm
%{   
    #include"RNorm.h"
    
%}

%include"RNorm.h"
%include "std_vector.i"
namespace std {
  %template(Vector) vector<double>;
}

