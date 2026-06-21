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
      double asymp_threshold = strtod("1e-2",NULL); // this threshold is not actually correct
      if ((a > 0) && (x == 0.0)) {
	return 0.0;
      } else if ((a > 0) && (b > 0)) {
	return gsl_sf_beta(a,b)*gsl_sf_beta_inc(a,b,x);
      } else if ((b == 0) && (1-x < asymp_threshold)) {
	double eps = 1-x;
	double res = -(1 - a*eps)*(gsl_sf_log(eps) + gsl_sf_psi(a) + M_EULER);
	std::cout << "Warning: using asymptotic form of beta_inc(" << a << "," << b << "," << x << "): [eps=" << eps << ", a=" << a << "]" <<  "\t\t -(1 - a*eps)*(gsl_sf_log(eps) + gsl_sf_psi(a) + M_EULER) = " << res << "\n";
	return res; // logarithmic asymptote of incomplete beta function when x -> 1 from below
	// return (1-α*(1+2k)*ε) * (-1)*(Log[ε] + PolyGamma[α*(1+2k)] + EulerGamma);
      } else if ((b < 0) && (1-x < asymp_threshold)) {
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

    arma::vec genlaguerre_n(int n, double a, double x) {
      // produce a vector of Laguerre polynomials of order 0...n, each
      // evaluated with the same argument (x) and parameter (a)
      // (see e.g. genlaguerreCN in common.m)
      if (a < -1) {
	return {arma::datum::nan};
      }

      if (n < 0) {
	return {};
      }

      if (n == 0) {
	return {1.0};
      }

      if (n == 1) {
	return {1.0, -x+a+1};
      }

      arma::vec bi2(n+1);
      bi2[0] = 1.0;

      double d(-x/(a+1));
      double p(d+1);
      arma::vec p2(n+1);
      p2[0] = 1.0;
      p2[1] = p;

      int k;
      //int kk;

      for (int kk = 0; kk <= n-2; kk++) {
	k = kk+1;
	d=-x/(k+a+1)*p + (k/(k+a+1))*d;
	p=d+p;
	p2[k+1] = p;
      }

      double bi(1.0);
      for (int i = 1; i <= n; i++) {
	bi = bi*(a+i)/i;
	bi2[i] = bi;
      }

      return bi2 % p2;
    }

    arma::vec jacobi_n(int n, double a, double b, double x) {
      // A vector of Jacobi polynomials of orders 0...n, each
      // evaluated with the same argument (x) and parameters (a,b)
      if (n == 0) {
	return {1.0};
      }

      if (n == 1) {
	return {1.0, 0.5*(2*(a+1)+(a+b+2)*(x-1))};
      }

      arma::vec bi2(n+1);
      bi2[0] = 1.0;

      //double d(-x/(a+1));
      //double p(d+1);
      arma::vec p2(n+1);

      double d((a+b+2)*(x - 1) / (2*(a+1)));
      double p(d + 1);
      double t;
      int k;

      p2[0] = 1.0;
      p2[1] = p;

      for (int kk = 0; kk <= n-2; kk++) {
	k = kk+1;
	t = 2*k+a+b;
	d = ((t*(t+1)*(t+2))*(x-1)*p + 2*k*(k+b)*(t+2)*d) / (2*(k+a+1)*(k+a+b+1)*t);
	p = d + p;
	p2[k+1] = p;
      }
      
      double bi(1.0);
      for (int i = 1; i <= n; i++) {
	bi = bi*(a+i)/i;
	bi2[i] = bi;
      }

      return bi2 % p2;
    }

    arma::vec jacobi_norm_n(size_t n, double a, double b, double x) {
      // std::cout << "computing jacobi_norm_n with n=" << n << " a=" << a << " b=" << b << " x=" << x << "\n";
      double h0(pow(2,0.5*(1+a+b))*sqrt(gsl_sf_beta(a+1,b+1)));
      // std::cout << "h0=" << h0 << "\n";

      arma::vec ret(n+1);

      ret[0] = 1/h0;

      if (n == 0) {
	return ret;
      }

      ret[1] = ((1+(a+b)/2)*x + (a-b)/2)*sqrt((3+a+b)/(1+a+b+a*b))*ret[0];

      if (n == 1) {
	return ret;
      }

      double An,Bn,Cn,id;

      for (size_t i = 2; i <= n; i++){
	id = double(i);
	An = ((a + b + 2*id)*sqrt(((-1 + a + b + 2*id)*(1 + a + b + 2*id))/(id*(a + id)*(b + id)*(a + b + id))))/2;
	Bn = ((a - b)*(a + b)*sqrt(((-1 + a + b + 2*id)*(1 + a + b + 2*id))/(id*(a + id)*(b + id)*(a + b + id))))/(2*(-2 + a + b + 2*id));
	Cn = ((-1 + a + id)*(-1 + b + id)*(a + b + 2*id)*sqrt(((-1 + id)*id*(-1 + a + b + id)*(a + b + id)*(1 + a + b + 2*id))/((-1 + a + id)*(a + id)*(-1 + b + id)*(b + id)*(-3 + a + b + 2*id))))/(id*(a + b + id)*(-2 + a + b + 2*id));
	ret[i] = (An*x + Bn)*ret[i-1] - Cn*ret[i-2];
      }

      return ret;
    }

    void print_mat_dims(std::string s, arma::mat M) {
      std::cout << s << ".n_rows=" << M.n_rows << " " << s << ".n_cols=" << M.n_cols << "\n";
    }

    void print_vec_dims(std::string s, arma::vec V) {
      std::cout << s << ".n_elem=" << V.n_elem << "\n";
    }


  }
}
