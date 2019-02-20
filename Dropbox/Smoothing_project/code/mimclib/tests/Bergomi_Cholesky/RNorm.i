%module RNorm
%{   
    #include"RNorm.h"
    
%}

%include"RNorm.h"
%include "std_vector.i"
%include "stdint.i"
namespace std {
  %template(Vector) vector<double>;
}


%{
  typedef std::mt19937_64 MTGenerator;
  typedef std::normal_distribution<double> normDist;;
%}

typedef std::mt19937_64 MTGenerator;
typedef std::normal_distribution<double> normDist;
