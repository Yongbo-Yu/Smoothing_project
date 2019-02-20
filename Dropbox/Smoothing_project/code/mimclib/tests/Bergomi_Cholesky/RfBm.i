%module RfBm
%include typemaps.i
%apply Vector &OUTPUT {Vector & Wtilde};

%{   
    #define SWIG_FILE_WITH_INIT
    #include "RfBm.h"
    
%}
%include "RfBm.h"

