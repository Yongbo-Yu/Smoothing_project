%module RBergomi
%{    

   
    #include "RBergomi.h"
 
    
    
%}

%include "std_vector.i"
namespace std {
  %template(Vector) vector<double>;
}

%include "numpy.i"

%init {
  import_array();
}

//include typemaps.i
//apply double* IN_ARRAY1 &OUTPUT { Vector  & W1 };
//apply double* IN_ARRAY1 &OUTPUT { Vector  & W1perp };

%apply (double* IN_ARRAY1, double* IN_ARRAY1)         \
{(Vector  & W1 , Vector   & W1perp  )}

class RBergomiST {
public:
	//*structors
	RBergomiST();
	// seed is an optional parameter
	RBergomiST(double x, Vector HIn, Vector e, Vector r, Vector t, Vector k, int NIn, long MIn,
			std::vector<uint64_t> seed);
	~RBergomiST();

	// Compute payoff (a la Romano-Touzi) for an array of Brownian increments
	std::vector<Vector> ComputePayoffRT(const std::vector<Vector> & W1Arr, const std::vector<Vector> & W1perpArr);
	// Compute one single value (corresponding to the very first entry in each parameter array)
	double ComputePayoffRT_single(const Vector & W1, const Vector & W1perp);
	// auxiliary methods
	// getters and setters
	long getM() const;
	void setM(long m);
	int getN() const;
	void setN(int n);
	double getXi() const;
};





