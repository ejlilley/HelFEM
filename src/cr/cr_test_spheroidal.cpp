#include <typeinfo>
#include <iostream>
#include <iomanip>

#include <cfloat>
#include "../../libhelfem/include/cr/cr_integrals.h"
//#include "cr_spherical_potentials.h"
#include "../../libhelfem/include/cr/cr_utils.h"
#include "../../libhelfem/include/cr/cr_spheroidal_potentials.h"
#include "../../libhelfem/include/FiniteElementBasis.h"
#include "../../libhelfem/include/PolynomialBasis.h"
#include "../../libhelfem/src/utils.h"
//#include "PolynomialBasis.h"
// #include "RadialBasis.h"
#include <helfem.h>
#include <quadrature.h>
#include "chebyshev.h"
#include "lobatto.h"
#include <gsl/gsl_sf_coupling.h>

#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/diis.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/timer.h"
#include "../general/scf_helpers.h"
// #include "../general/gaunt.h"
// #include "../atomic/basis.h"
#include "../diatomic/basis.h"

// A program to test the accuracy of the CR method by using it to
// compute the equivalents of the primitive TEIs (a quantity which is
// not actually required in basis.coulomb_cr())

using namespace helfem;

static double factorial_ratio(int pmax, int pmin) {
  // Check consistency of arguments
  if(pmax < pmin)
    return 1.0/factorial_ratio(pmin, pmax);
  
  // Calculate ratio
  double r=1.0;
  for(int p=pmax;p>pmin;p--)
    r*=p;
  
  return r;
}


int main(int argc, char **argv) {
  cmdline::parser parser;

  parser.add<int>("nmax", 0, "Nmax", false, 50);
  parser.add<int>("L", 0, "L", false, 0);
  parser.add<int>("M", 0, "M", false, 0);
  parser.add<int>("nelem", 0, "Nelem", false, 3);
  parser.add<int>("nnodes", 0, "nnodes", false, 3);
  parser.add<int>("nquad", 0, "nquad", false, 50);
  parser.add<double>("Rmax", 0, "Rmax", false, 10.0);
  parser.add<double>("Rhalf", 0, "Rhalf", false, 1.0);
  parser.parse_check(argc, argv);

  double Rhalf(parser.get<double>("Rhalf"));
  int Nmax(parser.get<int>("nmax"));
  int L(parser.get<int>("L"));
  int M(parser.get<int>("M"));
  int Nelem(parser.get<int>("nelem"));
  int nnodes(parser.get<int>("nnodes"));
  int nquad(parser.get<int>("nquad"));
  double Rmax(parser.get<double>("Rmax"));


  int primbas = 4;

  auto pb(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,nnodes)));
  //polynomial_basis::PolynomialBasis *pb =  polynomial_basis::get_basis(4,nnodes);

  arma::vec x; arma::vec wx;
  
  // ::lobatto_compute(nquad, x, wx);
  chebyshev::chebyshev(nquad, x, wx);

  helfem::cr::PhinlmTable phinlm(2,Nmax,L,M,Rhalf);
  // helfem::cr::PhinlmTable phinlm0(0,Nmax,L,M,Rhalf);

  int igrid(4); double zexp(1.0);

  double mumax(utils::arcosh(Rmax/Rhalf));

  arma::vec bval(atomic::basis::normal_grid(Nelem, mumax, igrid, zexp));

  std::cout.precision(15);
  std::cout.setf(std::iostream::fixed);

  std::cout << "type:" << typeid(pb).name() << "\n";
  std::cout << "id:" << pb->get_id() << "\n";
  std::cout << "nnodes:" << pb->get_nnodes() << "\n";
  std::cout << "nprim:" << pb->get_nprim() << "\n";
  std::cout << "nbf:" << pb->get_nbf() << "\n";
  std::cout << "nodes:" << pb->get_nodes() << "\n";
  // std::cout << "bval:" << bval << "\n";
  bval.raw_print(std::cout, "bval:");
  std::cout << "nquad:" << nquad << "\n";
  std::cout << "Rhalf:" << Rhalf << "\n";
  std::cout << "Rmax:" << Rmax << "\n";
  std::cout << "mumax:" << mumax << "\n";
  std::cout << "Nelem:" << Nelem << "\n";
  std::cout << "Nmax:" << Nmax << "\n";
  std::cout << "L:" << L << "\n";
  std::cout << "M:" << M << "\n";

  // x.raw_print(std::cout, "quadrature points:");
  // wx.raw_print(std::cout, "quadrature weights:");

  helfem::polynomial_basis::FiniteElementBasis fem(pb, bval, false, false, true, true);

  diatomic::basis::RadialBasis radial;
  radial=diatomic::basis::RadialBasis(fem, nquad);

  arma::ivec lmmax; lmmax.ones(M+1);
  arma::ivec lval, mval;
  diatomic::basis::lm_to_l_m(lmmax,lval,mval);
  int Z1(1); int Z2(1); int lpad(10);
  diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, pb, nquad, bval, lval, mval, lpad);
  basis.compute_tei(false);

  helfem::legendretable::LegendreTable legtab;
  legtab=legendretable::LegendreTable(L+lpad,L,M);
  arma::vec chmu(radial.get_chmu_quad());
  for(size_t i=0;i<chmu.n_elem;i++) {
    legtab.compute(chmu(i));
  }


  std::vector<arma::mat> cr_twoe0(Nelem);
  std::vector<arma::mat> cr_twoe2(Nelem);
  std::vector<arma::mat> cr_twoe(Nelem);

  std::vector<arma::mat> cr_twoe0_orig(Nelem);
  std::vector<arma::mat> cr_twoe2_orig(Nelem);


  double LMfac((4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/factorial_ratio(L+std::abs(M),L-std::abs(M))));

  std::cout << "LMfac=" << LMfac << "\n";

  for(int iel = 0; iel < Nelem; iel++) {
    
    cr_twoe0[iel] = helfem::cr::twoe_integral_quadrature_diatomic(fem, phinlm, L, M, 0, iel, x, wx);
    // std::cout << "cr_twoe0[" << iel << "].n_rows = " << cr_twoe0[iel].n_rows << "    cr_twoe0[" << iel << "].n_cols = " << cr_twoe0[iel].n_cols << "\n";
    // int r(cr_twoe0[iel].n_rows);
    // cr_twoe0[iel].row(r-1).raw_print(std::cout, "cr_twoe0:");
    // cr_twoe2[iel] = helfem::cr::twoe_integral_quadrature_diatomic(fem, phinlm, L, M, 2, iel, x, wx);

    // cr_twoe0_orig[iel] = helfem::cr::twoe_integral_quadrature_diatomic(fem, phinlm0, L, M, 0, iel, x, wx);
    // cr_twoe2_orig[iel] = helfem::cr::twoe_integral_quadrature_diatomic(fem, phinlm0, L, M, 2, iel, x, wx);


    // std::cout << "cr_twoe0/cr_twoe0_orig=\n" << cr_twoe0[iel]/cr_twoe0_orig[iel];

    // cr_twoe0[iel].raw_print(std::cout, "cr_twoe0:");

    arma::mat cr_twoe00 = (cr_twoe0[iel] * arma::trans(cr_twoe0[iel]))/(LMfac);
    // arma::mat cr_twoe02 = (cr_twoe0[iel] * arma::trans(cr_twoe2[iel]))/(LMfac);
    // arma::mat cr_twoe20 = (cr_twoe2[iel] * arma::trans(cr_twoe0[iel]))/(LMfac);
    // arma::mat cr_twoe22 = (cr_twoe2[iel] * arma::trans(cr_twoe2[iel]))/(LMfac);

    // std::cout << "cr_twoe00=\n" << cr_twoe00;

    arma::mat prim_tei00 = radial.twoe_integral(0,0,iel,L,M,legtab);
    // arma::mat prim_tei02 = radial.twoe_integral(0,2,iel,L,M,legtab);
    // arma::mat prim_tei20 = radial.twoe_integral(2,0,iel,L,M,legtab);
    // arma::mat prim_tei22 = radial.twoe_integral(2,2,iel,L,M,legtab);

    // std::cout << "prim_tei00=\n" << prim_tei00;
    

    arma::mat ones_mat;
    ones_mat.copy_size(cr_twoe00);
    ones_mat.ones();

    std::cout << "log|1-cr_twoe00/prim_tei00|=\n" << arma::log10(arma::abs(ones_mat - cr_twoe00 / prim_tei00));
    // std::cout << "log|1-cr_twoe02/prim_tei02|=\n" << arma::log10(arma::abs(ones_mat - cr_twoe02 / prim_tei02));
    // std::cout << "log|1-cr_twoe20/prim_tei20|=\n" << arma::log10(arma::abs(ones_mat - cr_twoe20 / prim_tei20));
    // std::cout << "log|1-cr_twoe22/prim_tei22|=\n" << arma::log10(arma::abs(ones_mat - cr_twoe22 / prim_tei22));
    arma::mat ratio(cr_twoe00 / prim_tei00);
    // std::cout << "cr_twoe00/prim_tei00=\n" << ratio;
    std::cout.precision(15);
    std::cout.setf(std::iostream::fixed);
    ratio.raw_print(std::cout, "cr_twoe00/prim_tei00=");

    cr_twoe00.raw_print(std::cout, "cr_twoe00:");
    prim_tei00.raw_print(std::cout, "prim_tei00:");

  }

    return 0;
}


