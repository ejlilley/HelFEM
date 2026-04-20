#include <vector>
#include <armadillo>


namespace helfem {
  namespace cr {

    double beta_inc(double a, double b, double x);

    int minusonepow(int n);

    double powsum(int k, const arma::vec & x);

    arma::vec es_poly(int k, const arma::vec & x);

  }
}
