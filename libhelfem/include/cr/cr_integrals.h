#include <armadillo>
#include "cr_spherical_potentials.h"
#include <PolynomialBasis.h>
#include "RadialBasis.h"

namespace helfem {
  namespace cr {

    // arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L);
    arma::mat twoe_integral_pairs(double rmin, double rmax, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int nmax, int L, double rs);

    arma::mat twoe_integral(const polynomial_basis::FiniteElementBasis & fem, IknlTable & iknl, int L, double rs, size_t iel);

    arma::vec psi_monomial_coeffs(int i, int j, const arma::vec & r0);

    arma::vec phi_monomial_coeffs(int i, const arma::vec & r0);

    double psi_prefactor(int i, int j, const arma::vec & r);

    double phi_prefactor(int i, const arma::vec & r);
    
  }
}

