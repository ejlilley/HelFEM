#include <typeinfo>
#include <string>
#include <iostream>
#include <iomanip>

#include <cfloat>
#include "../../libhelfem/include/cr/cr_integrals.h"
//#include "cr_spherical_potentials.h"
#include "../../libhelfem/include/cr/cr_utils.h"
//#include "PolynomialBasis.h"
#include "RadialBasis.h"
#include <helfem.h>
#include <quadrature.h>
#include "chebyshev.h"
#include "lobatto.h"


#include "../general/cmdline.h"
#include "../general/checkpoint.h"
#include "../general/constants.h"
#include "../general/diis.h"
#include "../general/dftfuncs.h"
#include "../general/elements.h"
#include "../general/timer.h"
#include "../general/scf_helpers.h"


#include "../atomic/basis.h"

using namespace helfem;



int main(int argc, char* argv[]) {

  // std::cout << argc << "\n";
  // std::cout << argv[0] << "\n";
  // std::cout << argv[1] << "\n";
  // std::cout << argv[2] << "\n";
  // std::cout << argv[3] << "\n";
  // std::cout << argv[4] << "\n";
  
  if (argc < 5) {
    std::cout << "Usage: ./cr_check_expressions <CR> <alpha> <nmax> <lmax>\n";
    std::cout << "       where <CR> is the type of basis function to use for the Coulomb Resolution (0: Zhao, 1: Slater, etc.)\n";
    std::cout << "             <alpha> is the fractional scaling parameter (affects asymptotic slope near 0) (default: 0.5)\n";
    std::cout << "             <nmax> test functions with n=0...nmax\n";
    std::cout << "             <lmax> test functions with l=0...lmax\n";
    return 0;
  }

  int CR = std::stoi(argv[1]);

  if (CR != 0 && CR != 1) {
    std::cout << "need CR = 0 or 1\n";
    return 1;
  }

  double alpha = std::stod(argv[2]);

  if (alpha < 0.5) {
    std::cout << "need alpha >= 0.5\n";
    return 1;
  }

  int nmax = std::stoi(argv[3]);

  if (nmax < 0) {
    std::cout << "need nmax > 0\n";
    return 1;
  }

  int lmax = std::stoi(argv[4]);

  if (lmax < 0) {
    std::cout << "need lmax > 0\n";
    return 1;
  }

  helfem::cr::PhinlTable phinl(CR,nmax,lmax,alpha);

  arma::vec r_test;

  for (std::string line; std::getline(std::cin, line);) {
    double r_value = std::stod(line);
    if (r_value < 0.0) {
      std::cout << "all r values must be > 0\n";
      return 1;
    }
    r_test.resize(r_test.n_rows + 1);
    r_test[r_test.n_rows-1] = r_value;
    phinl.compute(r_value);
  }

  std::cout << "         r = " << arma::trans(r_test) << "\n";

  for (int l = 0; l <= lmax; l++) {
    for (int n = 0; n <= nmax; n++) {
      std::cout << "phinl(" << n << "," << l << ") = " << arma::trans(phinl.get_Phinl(n,l,r_test)); //  << "\n";
    }
  }

  std::cout << "\n\n";

  //arma::vec foo(helfem::cr::genlaguerre_n(nmax, lmax+0.5, r_test[0]));
  //
  //std::cout << "Laguerre test: " << foo << "\n";

  // int l = 0;
  // arma::vec k_test = {0.0, 1.0, 2.0, 3.0};
  // arma::vec s_test = k_test + 0.5*arma::ones(k_test.n_elem);
  // //arma::mat pnl_test(nmax+1,k_test.n_elem);
  // arma::mat pnl_test(phinl.pnl_mat(nmax,l,s_test));
  // 
  // std::cout << "  k = " << arma::trans(k_test);
  // std::cout << "  s = " << arma::trans(s_test) << "\n";
  // std::cout << "pnl(0..n,l) = \n" << pnl_test << "\n";

  return 0;
}
