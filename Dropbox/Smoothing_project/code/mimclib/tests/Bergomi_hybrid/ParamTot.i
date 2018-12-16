%module ParamTot
%{   
    #include"ParamTot.h"
    
%}

%include"ParamTot.h"
%include "std_vector.i"
namespace std {
  %template(Vector) vector<double>;
}

