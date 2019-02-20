/*
 * rBergomi.h
 *
 *  Created on: 26 Jan 2018
 *      Author: bayerc
 */

#ifndef RBERGOMI_H_
#define RBERGOMI_H_

#include <vector>
#include <numeric>
#include <math.h>
//include <algorithm>
// #include "RNorm.h"
// #include "RfBm.h"

typedef std::vector<double> Vector;

/*
 Compute the call option price in the rough Bergomi model, with r = 0.
 */

// void mc_bayer_roughbergomi_cholesky(double S0, double eta, double H, double rho,
// 		double xi, double K, double T, int M, int N, double* price,
// 		double* stat);

//  * Re-interpreting K as moneyness, we may have S0 = 1, without loss of generality.
 

// void mc_bayer_roughbergomi_moneyness_cholesky(double eta, double H, double rho,
// 		double xi, double K, double T, int M, int N, double* price,
// 		double* stat);



// Compute value on one single trajectory
double updatePayoff_cholesky(Vector& Wtilde, const Vector& W1,
		Vector& v, double eta, double H, double rho, double xi,
		double T, double K, int N);

// compute v
void compute_V(Vector& v, const Vector& Wtilde, double H, double eta, double xi,
		double dt);

// compute \int_0^T v_t dt
double intVdt(const Vector & v, double dt);

// compute \int_0^T \sqrt{v_t} dW_t, with W = W1
double intRootVdW(const Vector & v, const Vector & W1, double sdt);

// BS Call price formula
double BS_call_price(double S0, double K, double tau, double sigma, double r =
		0.0);

// Auxiliary functions

// Normal CDF
double pnorm(double value);

// z = a*x+b*y
template<typename T>
inline std::vector<T> linearComb(T a, const std::vector<T>& x, T b,
		const std::vector<T>& y) {
	std::vector<T> z(x.size());
	for (int i = 0; i < z.size(); ++i)
		z[i] = a * x[i] + b * y[i];
	return z;
}

// x = s*x
template<typename T>
inline void scaleVector(std::vector<T>& x, T s) {
	for (auto& v : x)
		v = v * s;
}


#endif /* RBERGOMI_H_ */
