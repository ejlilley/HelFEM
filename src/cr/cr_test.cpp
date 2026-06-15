#include <typeinfo>


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

using namespace helfem;

// double psi_monomial_coeff(int i, int j, int k, const arma::vec & r0) {
//   //std::cout << "r0: " << r0 << "\n";
//   int n = r0.n_elem;
//   //std::cout << "j: " << j << "\n";
//   arma::vec r1(r0); r1.shed_row(i);
//   arma::vec r2(r0); r2.shed_row(j);
//   //std::cout << "r: " << r << "\n";
//   arma::vec r3 = arma::join_rows(r1,r2).as_col();
//   //std::cout << "r3: " << r3 << "\n";
//   arma::vec esp = cr::es_poly(2*n-2-k, r3);
//   //std::cout << "esp: " << esp << "\n";
//   return cr::minusonepow(2*n-2-k)*esp[k];
// }


    double gaunt_coefficient(int L, int M, int l, int m, int lp, int mp) {
      if(M != m+mp)
        // Coefficient vanishes if l values don't match
        return 0.0;
      if(L < abs(l-lp) || L > l+lp)
        // Can't couple
        return 0.0;

      // First, compute square root factor
      double res=sqrt((2*L+1)*(2*l+1)*(2*lp+1)/(4.0*M_PI));
      // Plug in (l1 l2 l3 | 0 0 0), GSL uses half units
      res*=gsl_sf_coupling_3j(2*L,2*l,2*lp,0,0,0);
      // Plug in (l1 l2 l3 | m1 m2 m3)
      res*=gsl_sf_coupling_3j(2*L,2*l,2*lp,-2*M,2*m,2*mp);
      // and the phase factor
      res*=pow(-1.0,M);

      return res;
    }

    double modified_gaunt_coefficient(int lj, int mj, int L, int M, int li, int mi) {
      static const double const0(2.0/3.0*sqrt(M_PI));
      static const double const2(4.0/15.0*sqrt(5.0*M_PI));

      // Coupling through Y_0^0
      double cpl0(gaunt_coefficient(L,M,0,0,L,M)*gaunt_coefficient(lj,mj,li,mi,L,M));

      // Coupling through Y_2^0
      double cpl2=0.0;
      for(int Lp=std::max(std::max(L-2,0),std::abs(M));Lp<=L+2;Lp++)
        cpl2+=gaunt_coefficient(Lp,M,2,0,L,M)*gaunt_coefficient(lj,mj,li,mi,Lp,M);

      return const0*cpl0+const2*cpl2;
    }



double eval_psi(int i, int j, const arma::vec & rnodes, double r) {
  //std::cout << "eval_psi(" << i << "," << j << "):\n";
  int n = rnodes.n_elem;
  //std::cout << "rnodes: " << rnodes << "\n";
  arma::vec bijk = cr::psi_monomial_coeffs(i, j, rnodes);
  //std::cout << "bijk: " << bijk << "\n";

  double prefactor = cr::psi_prefactor(i, j, rnodes);
  double res = 0.0;

  for (int k = 0; k <= 2*n-1; k++) {
    res += pow(r,k)*bijk[k];
  }

  //std::cout << "psi: res = " << res << "\n";
  //std::cout << "psi: prefactor = " << prefactor << "\n";
  //std::cout << "psi: prefactor*res = " << prefactor*res << "\n";

  return prefactor*res;
}

double eval_phi(int i, const arma::vec & rnodes, double r) {
  //std::cout << "eval_phi(" << i << "):\n";
  int n = rnodes.n_elem;
  arma::vec bik = cr::phi_monomial_coeffs(i, rnodes);
  //std::cout << "bik: " << bik << "\n";
  double prefactor = cr::phi_prefactor(i, rnodes);
  double res = 0.0;

  for (int k = 0; k <= n; k++) {
    res += pow(r,k)*bik[k];
  }

  //std::cout << "phi: res = " << res << "\n";
  //std::cout << "phi: prefactor = " << prefactor << "\n";
  //std::cout << "phi: prefactor*res = " << prefactor*res << "\n";


  return prefactor * res;
}

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


int main(void) {

  // phinl.compute(1.0); phinl.compute(2.0); phinl.compute(3.0); phinl.compute(4.0);
  // 
  // //double res = phinl.get_Phinl(1,1,0.3);
  // arma::vec rs = {1.0,2.0,3.0,4.0};
  // arma::vec res;
  // 
  // res = phinl.get_Phinl(0,0,rs);
  // std::cout << "phinl_(0,0)(" << arma::trans(rs) << ") = " << arma::trans(res) << "\n";
  // 
  // res = phinl.get_Phinl(1,1,rs);
  // std::cout << "phinl_(1,1)(" << arma::trans(rs) << ") = " << arma::trans(res) << "\n";
  // 
  // res = phinl.get_Phinl(2,2,rs);
  // std::cout << "phinl_(2,2)(" << arma::trans(rs) << ") = " << arma::trans(res) << "\n";
  // 
  // res = phinl.get_Phinl(3,3,rs);
  // std::cout << "phinl_(3,3)(" << arma::trans(rs) << ") = " << arma::trans(res) << "\n";
  // 
  // res = phinl.get_Phinl(4,4,rs);
  // std::cout << "phinl_(4,4)(" << arma::trans(rs) << ") = " << arma::trans(res) << "\n";




//   arma::vec rs = {1.0, 2.0, 3.0, 4.0, 5.0};
// 
//   arma::vec::iterator it     = rs.begin();
//   arma::vec::iterator it_end = rs.end();
// 
//   for(; it != it_end; ++it)  {
//     iknl.compute(*it);
//   }
// 
//   arma::vec res;
// 
//   res = iknl.get_Iknl(2,50,50,rs);
//   std::cout << "iknl_(2,50,50)(" << arma::trans(rs) << ") = " << arma::trans(res) << "\n";




//   arma::vec x(100);
// 
//   for (int i = 1; i <= 20; i ++) {
//     x[i] = i;
//   }
// 
//   arma::vec ep = helfem::cr::es_poly(20, x);
// 
//   std::cout << "e_n(x) = " << arma::trans(ep) << "\n";

  //pb = helfem::polynomial_basis::PolynomialBasis(4,5);

  //auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(4,5)));

  //std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(4,5));

  

  //auto poly(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(4,5)));

  //std::cout << "type:" << typeid(poly).name() << "\n";


//   int primbas = 4;
//   int nnodes = 3;
//   int quadnodes = 20; // should really have nnodes > 10, and then quadnodes > nnodes, but use small values for testing purposes
//   
//   auto pb(std::shared_ptr<const polynomial_basis::PolynomialBasis>(polynomial_basis::get_basis(primbas,nnodes)));
//   //polynomial_basis::PolynomialBasis *pb =  polynomial_basis::get_basis(4,nnodes);
//   std::cout << "type:" << typeid(pb).name() << "\n";
//   std::cout << "id:" << pb->get_id() << "\n";
//   std::cout << "nnodes:" << pb->get_nnodes() << "\n";
//   std::cout << "nprim:" << pb->get_nprim() << "\n";
//   std::cout << "nbf:" << pb->get_nbf() << "\n";
//   std::cout << "nodes:" << pb->get_nodes() << "\n";
//  
//   int nprim = pb->get_nprim();
// 
// 
// //  //  arma::vec x = {1.0,2.0,3.0};
// //  //  arma::vec wx = {1.0,1.0,1.0};
// //
//   arma::vec x; arma::vec wx;
//  
// //  //chebyshev::chebyshev(nnodes, x, wx);
//   ::lobatto_compute(quadnodes, x, wx);
//   
//   // arma::mat bf(pb->eval_dnf(x,0,5.0));
//   // std::cout << "bf:\n" << bf << "\n";
// 
// 
// 
// // 
// //   arma::vec nodes = pb->get_nodes();
// // 
// //   arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
// //   for(size_t fi=0;fi<bf.n_cols;fi++)
// //     for(size_t fj=0;fj<bf.n_cols;fj++)
// //       bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);
// //   arma::vec wp(wx*rlen);
// //   for(size_t i=0;i<bfprod.n_cols;i++)
// //     bfprod.col(i)%=wp;
// // 
// //   std::cout << "bfprod:\n" << bfprod << "\n";
// 
// 
// 
// //  double rmin = 2.7; double rmax = 3.1;
// //
// //  double rmid = 0.5*(rmax + rmin); double rlen = 0.5*(rmax - rmin);
// //
// //  std::cout << "rmin: " << rmin << "\trmax: " << rmax << "\trmid: " << rmid << "\trlen: " << rlen << "\n";
// //
// //  std::shared_ptr<const polynomial_basis::PolynomialBasis> p(pb);
// //
// //  int nprim = p->get_nprim();
// //
//   int Nmax = 50;
//   int L = 0;
//   int M = 0;
// 
// //  double rs = 1.5;
// 
//   double Rhalf = 1.0;
// 
// //  helfem::cr::PhinlTable phinl(0,Nmax,L,0.5);
// //  helfem::cr::IknlTable iknl(0,2*nprim-2,Nmax,L,0.5);
// 
//   helfem::cr::PhinlmTable phinlm(0,Nmax,L,M,Rhalf);
// 
// //  arma::mat ints_orig(helfem::cr::twoe_integral_wrk(rmin, rmax, p, iknl, L, rs));
// //  
// //  std::cout << "ints (original method):\n" << ints_orig << "\n";
// //  
// //  // arma::vec ints(helfem::cr::IBF0l_quadrature(rmin, rmax, L, x, wx, phinl, p, rs));
// //  // 
// //  // std::cout << "ints:\n" << ints << "\n";
// //  // 
// //  // std::cout << "ratio:\n" << ints/ints_orig << "\n";
// //
// //
// //  arma::mat ints(helfem::cr::IBFnl_quadrature(rmin, rmax, x, wx, phinl, p, rs));
// //  
// //  arma::mat intsL(ints.cols((Nmax+1)*L,(Nmax+1)*(L+1)-1));
// 
// //  //std::cout << "ints:\n" << ints << "\n";
// //
// //  std::cout << "ints:\n" << intsL << "\n";
// //
// //  std::cout << "ratio:\n" << intsL/ints_orig << "\n";
// 
// 
// //  double Lfac=(4.0*M_PI/(2*L+1))/2.0; // the extra /2.0 here is mysterious!
// //
// //  arma::mat quad_twoe = quadrature::twoe_integral(rmin, rmax, x, wx, p, L);
// //
// //  // std::cout << "quadrature twoe n rows: " << quad_twoe.n_rows << "\n";
// //  // std::cout << "quadrature twoe n cols: " << quad_twoe.n_cols << "\n";
// //  // std::cout << "quadrature twoe:\n" << quad_twoe << "\n";
// //
// //  arma::mat quad_twoe_full(nprim*nprim,nprim*nprim,arma::fill::zeros);
// //
// //  for (int i = 0; i < nprim; i++) {
// //    for (int j = 0; j < nprim; j++) {
// //      for (int k = 0; k < nprim; k++) {
// //	for (int l = 0; l < nprim; l++) {
// //	  // std::cout << "ijkl: " << i << "," << j << "," << k << "," << l << "\n";
// //	  quad_twoe_full(i + j*nprim, k + l*nprim) = Lfac*(quad_twoe(i + j*nprim, k + l*nprim) + quad_twoe(k + l*nprim, i + j*nprim));
// //	}
// //      }
// //    }
// //  }
// //
// //  std::cout << "quadrature twoe full n rows: " << quad_twoe_full.n_rows << "\n";
// //  std::cout << "quadrature twoe full n cols: " << quad_twoe_full.n_cols << "\n";
// //  std::cout << "quadrature twoe full:\n" << quad_twoe_full << "\n";
// //
// //  // arma::mat quad_twoe_inner = quadrature::twoe_inner_integral(rmin, rmax, x, wx, p, L);
// //  // 
// //  // std::cout << "quadrature inner twoe n rows: " << quad_twoe_inner.n_rows << "\n";
// //  // std::cout << "quadrature inner twoe n cols: " << quad_twoe_inner.n_cols << "\n";
// //  // std::cout << "quadrature inner twoe:\n" << quad_twoe_inner << "\n";
// //
// //
// //  arma::mat cr_twoe_pairs = cr::twoe_integral_pairs(rmin, rmax, p, Nmax, L, rs);
// //
// //  std::cout << "cr twoe pairs n rows: " << cr_twoe_pairs.n_rows << "    ";
// //  std::cout << "cr twoe pairs n cols: " << cr_twoe_pairs.n_cols << "\n";
// //  std::cout << "cr twoe pairs:\n" << cr_twoe_pairs << "\n";
// //
// //
// //
// //  int ndeg = 2*nprim - 2;
// //
// //  cr::IknlTable iknl(0,ndeg,Nmax,L,0.5);
// //
//   int Nelem = 2; double Rmax = 20.0;
//   int igrid(4); double zexp(1.0);
// //  int Z = 2; int Zl = 0; int Zr = 0;
// //  int finitenuc = 0; double Rrms = 0.0; int igrid = 4; double zexp = 2.0;
// //  int Nelem0 = 0; int igrid0 = 4; double zexp0 = 2.0;
// //  double Rhalf = 0.0; bool add_conf = true; double shift_conf = 0.0;
// //
// 
//   //arma::vec bval=atomic::basis::form_grid((modelpotential::nuclear_model_t) finitenuc, Rrms, Nelem, Rmax, igrid, zexp, Nelem0, igrid0, zexp0, Z, Zl, Zr, Rhalf, add_conf, shift_conf);
//   double mumax(utils::arcosh(Rmax/Rhalf));
//   arma::vec bval(atomic::basis::normal_grid(Nelem, mumax, igrid, zexp));
// 
//   helfem::polynomial_basis::FiniteElementBasis fem(pb, bval, false, false, true, true);
// 
//   diatomic::basis::RadialBasis radial;
//   radial=diatomic::basis::RadialBasis(fem, quadnodes);
// 
//   arma::ivec lmmax; lmmax.ones(M+1);
//   arma::ivec lval, mval;
//   diatomic::basis::lm_to_l_m(lmmax,lval,mval);
//   int Z1(1); int Z2(1); int lpad(10);
//   diatomic::basis::TwoDBasis basis(Z1, Z2, Rhalf, pb, quadnodes, bval, lval, mval, lpad);
//   basis.compute_tei(false);
// 
//   helfem::legendretable::LegendreTable legtab;
//   legtab=legendretable::LegendreTable(L+lpad,L,M);
//   arma::vec chmu(radial.get_chmu_quad());
//   for(size_t i=0;i<chmu.n_elem;i++) {
//     legtab.compute(chmu(i));
//   }
// 
// 
//   //arma::cube cr_twoe(Nelem,nprim*nprim,Nmax+1);
//   int Ncr = (L+1)*(M+1)*(Nmax+1);
// 
//   //arma::cube cr_twoe(nprim*nprim,Ncr,Nelem);
//   //std::cout << "cr_twoe.n_rows=" << cr_twoe.n_rows << "\n";
//   //std::cout << "cr_twoe.n_cols=" << cr_twoe.n_cols << "\n";
//   //std::cout << "cr_twoe.n_slices=" << cr_twoe.n_slices << "\n";
// 
//   std::vector<arma::mat> cr_twoe0(Nelem);
//   std::vector<arma::mat> cr_twoe2(Nelem);
//   std::vector<arma::mat> cr_twoe(Nelem);
// 
//   int gmax(10);
//   int lrval(10);
//   int midval(10);
//   int Mmax(2);
//   // gaunt::Gaunt gaunt;
//   // gaunt=gaunt::Gaunt(lrval,Mmax,midval,Mmax,lrval,Mmax);
// 
//   int lk(0); int mk(0); int ll(0); int ml(0);
//   // double cpl0(gaunt.mod_coeff(lk,mk,L,M,ll,ml));
//   double cpl0(modified_gaunt_coefficient(lk,mk,L,M,ll,ml));
//   // std::cout << "cpl0=" << cpl0 << "\n";
//   // double cpl2(gaunt.coeff(lk,mk,L,M,ll,ml));
//   double cpl2(gaunt_coefficient(lk,mk,L,M,ll,ml));
//   // std::cout << "cpl2=" << cpl2 << "\n";
//   double LMfac((4.0*M_PI*std::pow(Rhalf,5)*std::pow(-1.0,M)/factorial_ratio(L+std::abs(M),L-std::abs(M))));
// 
//   // std::cout << "lk=" << lk << " ";
//   // std::cout << "mk=" << mk << " ";
//   // std::cout << "L=" << L << " ";
//   // std::cout << "M=" << M << " ";
//   // std::cout << "ll=" << ll << " ";
//   // std::cout << "ml=" << ml << " ";
//   // std::cout << "\n";
//   // std::cout << "cpl0=" << cpl0 << " ";
//   // std::cout << "\n";
//   // std::cout << "cpl2=" << cpl2 << " ";
//   // std::cout << "\n";
//   // std::cout << "LMfac=" << LMfac << "\n";
// 
// 
//   for(int iel = 0; iel < Nelem; iel++) {
// 
//     cr_twoe0[iel] = helfem::cr::twoe_integral_quadrature_diatomic(fem, phinlm, L, M, 0, iel, x, wx);
//     cr_twoe2[iel] = helfem::cr::twoe_integral_quadrature_diatomic(fem, phinlm, L, M, 2, iel, x, wx);
//     // std::cout << "cr_twoe0[" << iel << "]=\n" << cr_twoe0[iel];
//     // std::cout << "cr_twoe2[" << iel << "]=\n" << cr_twoe2[iel];
// 
//     std::cout << "cr_twoe0[" << iel << "].n_rows = " << cr_twoe0[iel].n_rows << "    cr_twoe0[" << iel << "].n_cols = " << cr_twoe0[iel].n_cols << "\n";
//     // std::cout << "cr_twoe[" << iel << "]=\n" << cr_twoe[iel];
// 
// 
//     arma::mat cr_twoe00 = (cr_twoe0[iel] * arma::trans(cr_twoe0[iel]))/(LMfac);
//     arma::mat cr_twoe02 = (cr_twoe0[iel] * arma::trans(cr_twoe2[iel]))/(LMfac);
//     arma::mat cr_twoe20 = (cr_twoe2[iel] * arma::trans(cr_twoe0[iel]))/(LMfac);
//     arma::mat cr_twoe22 = (cr_twoe2[iel] * arma::trans(cr_twoe2[iel]))/(LMfac);
//     // std::cout << "cr_twoe00=\n" << cr_twoe00;
// 
//     // arma::mat Iijnlm(cpl2*cr_twoe2[iel] - cpl0*cr_twoe0[iel]);
//     // cr_twoe[iel] = LMfac*(Iijnlm * arma::trans(Iijnlm));
// 
//     arma::mat prim_tei00 = radial.twoe_integral(0,0,iel,L,M,legtab);
//     arma::mat prim_tei02 = radial.twoe_integral(0,2,iel,L,M,legtab);
//     arma::mat prim_tei20 = radial.twoe_integral(2,0,iel,L,M,legtab);
//     arma::mat prim_tei22 = radial.twoe_integral(2,2,iel,L,M,legtab);
// 
//     // std::cout << "prim_tei00=\n" << prim_tei00;
// 
//     arma::mat ones_mat;
//     ones_mat.copy_size(cr_twoe00);
//     ones_mat.ones();
// 
//     std::cout << "cr_twoe00/prim_tei00=\n" << arma::log10(ones_mat - cr_twoe00 / prim_tei00);
//     std::cout << "cr_twoe02/prim_tei02=\n" << arma::log10(ones_mat - cr_twoe02 / prim_tei02);
//     std::cout << "cr_twoe20/prim_tei20=\n" << arma::log10(ones_mat - cr_twoe20 / prim_tei20);
//     std::cout << "cr_twoe22/prim_tei22=\n" << arma::log10(ones_mat - cr_twoe22 / prim_tei22);
// 
// 
//   }






//
//  std::cout << "cr twoe n rows: " << cr_twoe.n_rows << "    ";
//  std::cout << "cr twoe n cols: " << cr_twoe.n_cols << "    ";
//  std::cout << "cr twoe n slices: " << cr_twoe.n_slices << "\n";
//  std::cout << "cr twoe:\n" << cr_twoe << "\n";
//


  size_t n(1000); double a(3.2); double b(4.6);
//  //arma::vec xs({0.0, 0.5, 1.5, 3.5});
  double z(0.2);

  arma::vec Pn(cr::jacobi_norm_n(n,a,b,z));

  std::cout << "P_{0.." << n << "}^{(" << a << "," << b << ")}" << "(" << z << ")=\n" << Pn;

//
//  int Nmax = 20; int Lmax = 10; int Mmax = 10;
//
//  helfem::cr::PhinlmTable phinlm(0,Nmax,Lmax,Mmax,1.0);
//
//  double mu(1.3); phinlm.compute(mu);
//
//  for (int l = 0; l <= Lmax; l++) {
//    for (int m = 0; m <= Mmax && m <= l; m++) {
//      for (int n = 0; n <= Nmax; n++) {
//	std::cout << "phinlm(" << n << "," << l << "," << m << ")(" << mu << ") = " << phinlm.get_Phinlm(n,l,m,mu) << "\n";
//      }
//    }
//  }
//
//  arma::cube Nnlm(phinlm.get_Nnlm());
//
//  for (int l = 0; l <= Lmax; l++) {
//    for (int m = 0; m <= Mmax && m <= l; m++) {
//      for (int n = 0; n <= Nmax; n++) {
//	std::cout << "Nnlm(" << n << "," << l << "," << m << ") = " << Nnlm(n,l,m) << "\n";
//      }
//    }
//  }
//
//  for (int l = 0; l <= Lmax; l++) {
//    for (int m = 0; m <= Mmax && m <= l; m++) {
//      for (int n = 0; n <= Nmax; n++) {
//	std::cout << "normed phinlm(" << n << "," << l << "," << m << ")(" << mu << ") = " << phinlm.get_Phinlm(n,l,m,mu)/sqrt(Nnlm(n,l,m)) << "\n";
//      }
//    }
//  }
//
//
//  return 0;


//    arma::vec rnodes(rmid*arma::ones<arma::vec>(nodes.n_elem)+rlen*nodes);
//  
//    std::cout << "rnodes: " << rnodes << "\n";
//  
//  //   double Cij = psi_prefactor(0, 1, r);
//  // 
//  //   std::cout << "Cij: " << Cij << "\n";
//  
//  //  arma::vec bijk = psi_monomial_coeffs(0, 1, r)/psi_prefactor(0, 1, r);
//  
//  //  std::cout << "bijk: " << bijk << "\n";
//  
//  
//    double r0 = 1.234; double x0 = (r0 - rmid)/rlen;
//  
//    std::cout << "r0: " << r0 << "\n";
//    std::cout << "x0: " << x0 << "\n";
//  
//    //double epp = eval_psi(0, 1, nodes, r0);
//  
//    //int ii = 1;
//  
//    //double phi_x0 = eval_phi(ii, nodes, x0);
//    //std::cout << "phi(x0): " << phi_x0 << "\n";
//  
//    //double phi_r0 = eval_phi(ii, rnodes, r0);
//    //std::cout << "phi(r0): " << phi_r0 << "\n";
//  
//  
//    arma::vec x0a = {x0};
//    arma::mat dnf = pb->eval_dnf(x0a, 0, 1.0);
//  
//    //std::cout << "dnf: " << dnf << "\n";
//    //double epp2 = dnf[0]*dnf[1];
//    //std::cout << "epp2: " << epp2 << "\n";
//  
//    arma::vec phi(nnodes);
//  
//    for (int i = 0; i < nnodes; i++) {
//        phi[i] = eval_phi(i, rnodes, r0);
//    }
//  
//    std::cout << "dnf: " << dnf << "\n";
//    std::cout << "phi: " << arma::trans(phi) << "\n";
//  
//  
//  
//    arma::mat phi_prod(nnodes,nnodes,arma::fill::zeros);
//    arma::mat psi(nnodes,nnodes,arma::fill::zeros);
//    arma::mat dnf_prod(nnodes,nnodes,arma::fill::zeros);
//  
//    for (int i = 0; i < nnodes; i++) {
//      for (int j = 0; j < nnodes; j ++) {
//        dnf_prod(i,j) = dnf[i]*dnf[j];
//        phi_prod(i,j) = phi[i]*phi[j];
//        psi(i,j) = eval_psi(i, j, rnodes, r0);
//      }
//    }
//    
//    std::cout << "dnf prod: " << dnf_prod << "\n";
//    std::cout << "phi prod: " << phi_prod << "\n";
//    std::cout << "psi prod: " << psi << "\n";

  return 0;
}

