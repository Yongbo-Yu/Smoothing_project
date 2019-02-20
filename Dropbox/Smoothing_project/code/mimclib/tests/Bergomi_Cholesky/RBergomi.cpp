/*
 * rBergomi.cpp
 *
 *  Created on: 26 Jan 2018
 *      Author: bayerc
 */

#include "RBergomi.h"

// void mc_bayer_roughbergomi_cholesky(double S0, double eta, double H, double rho,
// 		double xi, double K, double T, int M, int N, double* price,
// 		double* stat) {
// 	mc_bayer_roughbergomi_moneyness_cholesky(eta, H, rho, xi, K / S0, T, M, N, price,
// 			stat);
// 	*price = *price * S0;
// 	*stat = *stat * S0;
// }


// void mc_bayer_roughbergomi_moneyness_cholesky(double eta, double H, double rho,
// 		double xi, double K, double T, int M, int N, double* price,
// 		double* stat) {
// 	// Prepare random number generator
// 	const uint64_t seed = 123456789;
// 	RNorm rnorm(seed);
// 	RfBm rfbm(N, H, &rnorm);

// 	// Allocate memory for Gaussian random numbers and the random vector v (instantaneous variance)
// 	// Note that W1, W1perp correspond to UNNORMALIZED increments of Brownian motions,
// 	// i.e., are i.i.d. standard normal.
// 	Vector W1(N);
// 	Vector Wtilde(N);
// 	Vector v(N);

// 	double mean = 0.0; // will eventually be the mean
// 	double mu2 = 0.0; // will become second moment
// 	double var; // will eventually become variance (i.e., MC error).

// 	// The big loop which needs to be parallelized in future
// 	for (int m = 0; m < M; ++m) {
// 		// generate W and Wtilde
// 		rfbm(W1, Wtilde);

// 		double payoff = updatePayoff_cholesky(Wtilde, W1, v, eta, H, rho, xi, T, K,
// 				N);
// 		mean += payoff;
// 		mu2 += payoff * payoff;
// 	}

// // compute mean and variance
// 	mean = mean / M;
// 	mu2 = mu2 / M;
// 	var = mu2 - mean * mean;

// // price = mean, stat = sqrt(var) / sqrt(M)
// 	*price = mean;
// 	*stat = sqrt(var / M);

// }


// This function should be fed to MISC

double updatePayoff_cholesky(Vector& Wtilde, const Vector& W1,
		Vector& v, double eta, double H, double rho, double xi,
		double T, double K, int N){
	double dt = T / N;
	double sdt = sqrt(dt);
	scaleVector(Wtilde, pow(T, H)); // scale Wtilde for time T
	compute_V(v, Wtilde, H, eta, xi, dt); // compute instantaneous variance v
// now compute \int v_s ds, \int \sqrt{v_s} dW_s with W = W1
	double Ivdt = intVdt(v, dt);
	double IsvdW = intRootVdW(v, W1, sdt);
// now compute the payoff by inserting properly into the BS formula
	double BS_vol = sqrt((1.0 - rho * rho) * Ivdt);
	double BS_spot = exp(-0.5 * rho * rho * Ivdt + rho * IsvdW);
	return BS_call_price(BS_spot, K, 1.0, BS_vol);
}

// Note that Wtilde plays the role of the old WtildeScaled!
void compute_V(Vector& v, const Vector& Wtilde, double H, double eta, double xi,
		double dt) {
	v[0] = xi;
	for (int i = 1; i < v.size(); ++i)
		v[i] = xi
				* exp(
						eta * Wtilde[i - 1]
								- 0.5 * eta * eta * pow(i * dt, 2 * H));
}

double intVdt(const Vector & v, double dt) {
	return dt * std::accumulate(v.begin(), v.end(), 0.0);
}

double intRootVdW(const Vector & v, const Vector & W1, double sdt) {
	double IsvdW = 0.0;
	for (int i = 0; i < v.size(); ++i)
		IsvdW += sqrt(v[i]) * sdt * W1[i];
	return IsvdW;
}



double pnorm(double value) {
	return 0.5 * erfc(-value * M_SQRT1_2);
}

double BS_call_price(double S0, double K, double tau, double sigma, double r) {
	double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * tau)
			/ (sigma * sqrt(tau));
	double d2 = d1 - sigma * sqrt(tau);
	return pnorm(d1) * S0 - pnorm(d2) * K * exp(-r * tau);
}

