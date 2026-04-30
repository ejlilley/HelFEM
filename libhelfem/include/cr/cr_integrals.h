#include <armadillo>
#include "cr_spherical_potentials.h"
#include <PolynomialBasis.h>
#include "RadialBasis.h"

namespace helfem {
  namespace cr {

    // arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L);
    arma::mat twoe_integral_pairs(double rmin, double rmax, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int nmax, int L, double rs);

    arma::mat twoe_integral(const polynomial_basis::FiniteElementBasis & fem, IknlTable & iknl, int L, double rs, size_t iel);

    arma::mat twoe_integral_wrk(double rmin, double rmax, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & pb, IknlTable & iknl, int L, double rs);

    arma::mat twoe_integral_quadrature(const polynomial_basis::FiniteElementBasis & fem, IknlTable & iknl, int L, double rs, size_t iel, const arma::vec & x, const arma::vec & wx);

    arma::vec psi_monomial_coeffs(int i, int j, const arma::vec & r0);

    arma::vec phi_monomial_coeffs(int i, const arma::vec & r0);

    double psi_prefactor(int i, int j, const arma::vec & r);

    double phi_prefactor(int i, const arma::vec & r);
    
    // "IBF" meaning 'Integral over Basis Function' (as opposed to "Ik" meaning 'Integral over r^k')
    arma::vec IBF0l_quadrature(double rmin, double rmax, int l, const arma::vec & xq, const arma::vec & wq, helfem::cr::PhinlTable phinl, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, double rs);

    arma::mat IBFnl_quadrature(double rmin, double rmax, const arma::vec & xq, const arma::vec & wq, helfem::cr::PhinlTable phinl, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, double rs);

  }
}

