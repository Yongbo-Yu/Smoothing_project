%module RBergomi
%{    
   
    #include "RBergomi.h"
%}

%include "std_vector.i"
// Instantiate templates used by example
namespace std {
  %template(Vector) vector<double>;
}

// Include the header file with above prototypes
%include "RBergomi.h"





