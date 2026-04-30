#include "cr/cr_utils.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_psi.h>
#include <vector>
#include <armadillo>

namespace helfem {
  namespace cr {

    // unnormalised incomplete beta function, including support for negative arguments
    double beta_inc(double a, double b, double x) {
      //std::cout << "beta_inc_{" << x << "}(" << a << "," << b << ")\n";
      double asymp_threshold = strtod("1e-2",NULL);
      if ((a > 0) && (x == 0.0)) {
	return 0.0;
      } else if ((a > 0) && (b > 0)) {
	return gsl_sf_beta(a,b)*gsl_sf_beta_inc(a,b,x);
      } else if ((b == 0) && (1-x < asymp_threshold)) { // the 1e-3 threshold is not actually correct
	double eps = 1-x;
	double res = -(1 - a*eps)*(gsl_sf_log(eps) + gsl_sf_psi(a) + M_EULER);
	std::cout << "Warning: using asymptotic form of beta_inc(" << a << "," << b << "," << x << "): [eps=" << eps << ", a=" << a << "]" <<  "\t\t -(1 - a*eps)*(gsl_sf_log(eps) + gsl_sf_psi(a) + M_EULER) = " << res << "\n";
	return res; // logarithmic asymptote of incomplete beta function when x -> 1 from below
	// return (1-α*(1+2k)*ε) * (-1)*(Log[ε] + PolyGamma[α*(1+2k)] + EulerGamma);
      } else if ((b < 0) && (1-x < asymp_threshold)) { // the 1e-3 threshold is not actually correct
	double eps = 1-x;
	double res = -(1 - eps*a)*pow(eps,b)/b;
	std::cout << "Warning: using asymptotic form of beta_inc(" << a << "," << b << "," << x << "): [eps=" << eps << ", a=" << a << ", b=" << b << "]" <<  "\t\t -(1 - eps*a)*pow(eps,b)/b = " << res << "\n";
	return res; // (non-logarithmic) asymptote of incomplete beta function when x -> 1 from below
	// return ((1 - ε*α*(1+k+l)) * ε^(α*(l-k))/(α*(k-l)));
      } else {
	std::cout << "Warning: using hypergeometric form of beta_inc(" << a << "," << b << "," << x << "):\t\t (" << x << "^" << a << ")/" << a << " * 2F1(" << a << "," << 1-b << "," << a+1 << "," << x << ") = ";
	double res = pow(x,a)/a*gsl_sf_hyperg_2F1(a,1-b,a+1,x);
	std::cout << res << "\n";
	return res; // hypergeometric form of incomplete beta function, appropriate when b<0 and x is not too close to 1
      }
    }

    int minusonepow(int n){
      return (1 - 2*(n % 2));
    }

    double powsum(int k, const arma::vec & x) {
      return arma::sum(arma::pow(x, k));
    }

    // vector of elementary symmetric polynomials e_k(x_n), using Newton's identities (this might not be optimally efficient)
    arma::vec es_poly(int k, const arma::vec & x) {

      arma::vec ek(k+1);

      ek[0] = 1;

      if (k == 0) {
	return ek;
      }

      ek[1] = arma::sum(x);

      if (k == 1) {
	return ek;
      }

      arma::vec pk(k+1);

      pk[0] = 1;

      for (int j = 1; j <= k; j++) {
	pk[j] = powsum(j, x);
      }

      for (int j = 2; j <= k; j++) {
	double ej = 0.0;

	for (int i = 1; i <= j; i++) {
	  ej += minusonepow(i-1) * ek[j-i] * pk[i];
	}

	ek[j] = pow(j,-1)*ej;
      }

      return ek;

    }


    
  }
}
