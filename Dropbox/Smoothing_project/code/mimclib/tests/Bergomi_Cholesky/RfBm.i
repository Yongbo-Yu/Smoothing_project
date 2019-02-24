%module RfBm
%{   
  
    #include "RfBm.h"
    
%}

%include "std_vector.i"
// Instantiate templates used by example
namespace std {
  %template(Vector) vector<double>;
  %template(VVector) vector< vector<double> >;
}


%include "RfBm.h"

