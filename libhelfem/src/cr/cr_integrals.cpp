#include "cr/cr_integrals.h"
#include "cr/cr_utils.h"

namespace helfem {
  namespace cr {

    arma::vec psi_monomial_coeffs(int i, int j, const arma::vec & r0) {
      //std::cout << "r0: " << r0 << "\n";
      int n = r0.n_elem;
      //std::cout << "j: " << j << "\n";
      arma::vec r1(r0); r1.shed_row(i);
      arma::vec r2(r0); r2.shed_row(j);
      //std::cout << "r1: " << r1 << "\n";
      //std::cout << "r2: " << r2 << "\n";
      arma::vec r3 = arma::join_rows(r1,r2).as_col();
      //std::cout << "r3: " << r3 << "\n";
      arma::vec esp = cr::es_poly(2*n-2, r3);
      //std::cout << "esp: " << esp << "\n";
      arma::vec signs(2*n-1, arma::fill::zeros);
      for (int k = 0; k < 2*n-1; k++) {
	signs[k] = cr::minusonepow(k);
      }
      //std::cout << "signs: " << signs << "\n";
      return arma::reverse(signs % esp);
    }
    
    arma::vec phi_monomial_coeffs(int i, const arma::vec & r0) {
      //std::cout << "r0: " << r0 << "\n";
      int n = r0.n_elem;
      //std::cout << "j: " << j << "\n";
      arma::vec r1(r0); r1.shed_row(i);
      //arma::vec r2 = r0; r2.shed_row(j);
      //std::cout << "r1: " << r1 << "\n";
      //std::cout << "r2: " << r2 << "\n";
      //arma::vec r3 = arma::join_rows(r1,r2).as_col();
      //std::cout << "r3: " << r3 << "\n";
      arma::vec esp = cr::es_poly(n-1, r1);
      //std::cout << "esp: " << esp << "\n";
      arma::vec signs(n, arma::fill::none);
      for (int k = 0; k < n; k++) {
	signs[k] = cr::minusonepow(k);
      }
      //std::cout << "signs: " << signs << "\n";
      return arma::reverse(signs % esp);
    }


    double psi_prefactor(int i, int j, const arma::vec & r) {
      int n = r.n_elem;
      
      double res = 1.0;
      
      for (int k = 0; k < n; k++) {
	if (k == i) {
	  continue;
	} else {
	  res *= r[k] - r[i];
	}
      }
      
      for (int k = 0; k < n; k++) {
	if (k == j) {
	  continue;
	} else {
	  res *= r[k] - r[j];
	}
      }
      
      return pow(res, -1.0);
    }
    
    double phi_prefactor(int i, const arma::vec & r) {
      int n = r.n_elem;
      
      double res = 1.0;
      
      for (int k = 0; k < n; k++) {
	if (k == i) {
	  continue;
	} else {
	  res *= r[k] - r[i];
	}
      }
      
      return pow(res, -1.0);
    }



//     arma::mat twoe_integral(double rmin, double rmax, const arma::vec & x, const arma::vec & wx, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int L) {
    arma::mat twoe_integral_pairs(double rmin, double rmax, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, int nmax, int L, double rs) {

      double rmid = 0.5*(rmax + rmin);
      double rlen = 0.5*(rmax - rmin);

      arma::vec nodes = poly->get_nodes();
      arma::vec rnodes(rmid*arma::ones<arma::vec>(nodes.n_elem)+rlen*nodes);

      int nnodes = rnodes.n_elem;
      int nprim = poly->get_nprim();
      int ndeg = 2*nprim - 2; // maximum polynomial degree of product of basis funs (= Kmax)

      cr::IknlTable iknl(0,ndeg,nmax,L,0.5);

      iknl.compute(rmin/rs);
      iknl.compute(rmax/rs);

      arma::mat Nnl(iknl.get_Nnl());

      arma::cube bijk(nnodes,nnodes,ndeg+1);


      for (int i = 0; i < nnodes; i++) {
	for (int j = 0; j < nnodes; j++) {
	  bijk.tube(i,j) = psi_prefactor(i, j, rnodes)*psi_monomial_coeffs(i, j, rnodes);
	}
      }


      arma::cube cijnl(nprim,nprim,nmax+1,arma::fill::zeros);

      for (int n = 0; n <= nmax; n++) {
	for (int i = 0; i < nprim; i++) {
	  for (int j = 0; j < nprim; j++) {
	    for (int k = 0; k <= ndeg; k++) {
	      cijnl(i,j,n) += bijk(i,j,k) * pow(rs,k+0.5) * (iknl.get_Iknl(k,n,L,rmax/rs) - iknl.get_Iknl(k,n,L,rmin/rs));
	    }
	  }
	}
      }


      
      arma::mat twoe_ints(nprim*nprim,nprim*nprim,arma::fill::zeros);
      
      for (int i = 0; i < nprim; i++) {
	for (int j = 0; j < nprim; j++) {
	  for (int k = 0; k < nprim; k++) {
	    for (int l = 0; l < nprim; l++) {
	      //twoe_ints(i + j*nprim, k + l*nprim) = arma::sum(cijnl.tube(i,j) % cijnl.tube(k,l));
	      arma::vec foo(cijnl.tube(i,j));
	      arma::vec bar(cijnl.tube(k,l));
	      arma::vec baz = foo % bar / Nnl.col(L);
	      //std::cout << "two_ints(" << i << "," << j << "," << k << "," << l << ") = " << arma::sum(baz) << "\n";
	      twoe_ints(i + j*nprim, k + l*nprim) = arma::sum(baz);
	    }
	  }
	}
      }
      
      return twoe_ints;

    }


    //    arma::mat twoe_integral_quadrature(const polynomial_basis::FiniteElementBasis & fem, IknlTable & iknl, int L, double rs, size_t iel, const arma::vec & x, const arma::vec & wx) {
    arma::mat twoe_integral_quadrature(const polynomial_basis::FiniteElementBasis & fem, PhinlTable & phinl, int L, double rs, size_t iel, const arma::vec & x, const arma::vec & wx) {
      double rmin(fem.element_begin(iel));
      double rmax(fem.element_end(iel));

      std::shared_ptr<const polynomial_basis::PolynomialBasis> pb(fem.get_basis(iel));
      
      //arma::vec x; arma::vec wx; int quadnodes = 20;
      //
      //::lobatto_compute(quadnodes, x, wx);

      //cr::PhinlTable phinl(iknl.phinl);

      //int Nmax = iknl.get_Nmax();
      int Nmax = phinl.get_Nmax();

      arma::mat ints(IBFnl_quadrature(rmin, rmax, x, wx, phinl, pb, rs));
  
      return ints.cols((Nmax+1)*L,(Nmax+1)*(L+1)-1);
    }
    

    arma::mat twoe_integral_quadrature_diatomic(const polynomial_basis::FiniteElementBasis & fem, PhinlmTable & phinlm, int L, int M, int alpha, size_t iel, const arma::vec & xq, const arma::vec & wq) {
      std::shared_ptr<const polynomial_basis::PolynomialBasis> poly(fem.get_basis(iel));

      arma::vec w(wq);

      double mumin=fem.element_begin(iel);
      double mumax=fem.element_end(iel);

      double mumid = 0.5*(mumax + mumin);
      double mulen = 0.5*(mumax - mumin);
      int nprim = poly->get_nprim();
      int nq = w.n_rows; // number of integral weights
      
      arma::vec mu = (mumid*arma::ones<arma::vec>(xq.n_rows) + mulen*xq);

      // std::cout << "mu=\n" << mu;

      arma::mat bf(poly->eval_dnf(xq,0,mulen));
      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);

      for(size_t fi=0;fi<bf.n_cols;fi++)
	for(size_t fj=0;fj<bf.n_cols;fj++)
	  bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);

      arma::vec shmu(arma::sinh(mu));
      arma::vec chmu(arma::pow(arma::cosh(mu),alpha));

      w %= shmu;
      if (alpha != 0)
	w %= chmu;


      std::cout.precision(15);
      std::cout.setf(std::iostream::fixed);

      // mu.t()..raw_print(std::cout, "mu:");

      // w.t().raw_print(std::cout, "w:");

      // bfprod.col(bfprod.n_cols-1).t().raw_print(std::cout, "bfprod.col(imax):");


      for(size_t i=0;i<bfprod.n_cols;i++) { // multiply weights
        bfprod.col(i) %= w;
      }

      //std::cout << "bfprod=\n" << bfprod;

      arma::cube Nnlm(phinlm.get_Nnlm());

      int Nmax(phinlm.get_Nmax());

      for (int j = 0; j < nq; j++) {
	phinlm.compute(mu[j]);
      }

      //arma::mat phimu((Lmax+1)*(Mmax+1)*(Nmax+1),nq);

      arma::mat phi_n_mu(Nmax+1,nq);

      //for (int j = 0; j < nq; j++) {
	// evaluate (a cube of) Phinlm at each mu, normalised by sqrt(Nnlm)
	// arma::cube phinlm_cube(phinlm.get_Phinlm(mu[j]));
	//phimu.col(j) = arma::vectorise(phinlm_cube/arma::sqrt(Nnlm));
	//}


      for (int n = 0; n <= Nmax; n++) {

	arma::vec phimu_vec(phinlm.get_Phinlm(n, L, M, mu));
	//std::cout << "phimu_vec=\n" << phimu_vec;
	double norm(sqrt(Nnlm(L,abs(M),n)));
	// std::cout << "norm=" << norm << "\n";
	phi_n_mu.row(n) = arma::trans(phimu_vec)/norm;

	// std::cout << "loop n=" << n << "    norm=" << norm << "\n";
	// phimu_vec.raw_print(std::cout, "phimu_vec:");
      }



      // phi_n_mu.row(Nmax).t().raw_print(std::cout, "phi_n_mu.row(Nmax):");


      //std::cout << "phi_n_mu=\n" << phi_n_mu;

      arma::mat ints(Nmax+1,bfprod.n_cols);
      //arma::mat ints((Lmax+1)*(Mmax+1)*(Nmax+1),bfprod.n_cols);
      // so first index is flattened (n,l,m) (labelling CR basis functions)
      // & second index is flattened (i,j) (labelling in-element polynomial basis functions)

      // bfprod.raw_print(std::cout, "bfprod:");
      // phi_n_mu.raw_print(std::cout, "phi_n_mu:");



      double Rh(phinlm.get_Rh());

      ints = mulen/Rh * (phi_n_mu * bfprod);

      //ints.replace(arma::datum::nan, 0);

      return arma::trans(ints);
    }




    arma::mat twoe_integral(const polynomial_basis::FiniteElementBasis & fem, IknlTable & iknl, int L, double rs, size_t iel) {
      double rmin(fem.element_begin(iel));
      double rmax(fem.element_end(iel));

      std::shared_ptr<const polynomial_basis::PolynomialBasis> pb(fem.get_basis(iel));

      return twoe_integral_wrk(rmin, rmax, pb, iknl, L, rs);

      




//       arma::vec nodes = pb->get_nodes();
//       arma::vec rnodes(rmid*arma::ones<arma::vec>(nodes.n_elem)+rlen*nodes);
// 
//       //std::cout << "nodes =\n" << nodes << "\n";
//       //std::cout << "rnodes =\n" << rnodes << "\n";
// 
//       int nnodes = rnodes.n_elem;
//       size_t nprim = pb->get_nprim();
//       size_t nbf = pb->get_nbf();
//       size_t ndeg = 2*nprim - 2; // maximum polynomial degree of product of basis funs (= Kmax)
//       int Nmax = iknl.get_Nmax(); // maximum radial order of coulomb resolution: n = (0,...,Nmax)
//       arma::uvec en(pb->get_enabled());
//       
//       //std::cout << "cr::twoe_integral nnodes=" << nnodes << "\n";
//       //std::cout << "cr::twoe_integral nprim=" << nprim << "\n";
//       //std::cout << "cr::twoe_integral nbf=" << nbf << "\n";
//       //std::cout << "cr::twoe_integral ndeg=" << ndeg << "\n";
//       //std::cout << "cr::twoe_integral enabled=" << en << "\n";
// 
//       arma::mat Nnl(iknl.get_Nnl());
// 
//       iknl.compute(rmin/rs);
//       iknl.compute(rmax/rs);
// 
//       //arma::mat bijk(nnodes*nnodes,ndeg+1);
//       arma::mat bijk(nbf*nbf,ndeg+1);
// 
//       for (size_t i = 0; i < nbf; i++) {
// 	for (size_t j = 0; j < nbf; j++) {
// 	  //std::cout << "bijk: i=" << i << " j=" << j << "\n";
// 	  // std::cout << "bijk(" << i << "+" << nnodes << "*" << j << ") =" << psi_prefactor(i, j, rnodes)*psi_monomial_coeffs(i, j, rnodes) << "\n";
// 	  //arma::uvec bf_iv = arma::find(en == i, 1);
// 	  //arma::uvec bf_jv = arma::find(en == j, 1);
// 	  //std::cout << "bijk: bf_iv=" << bf_iv << " bf_jv=" << bf_jv << "\n";
// 	  size_t bf_i = en[i];
// 	  size_t bf_j = en[j];
// 	  //std::cout << "bijk: bf_i=" << bf_i << " bf_j=" << bf_j << " idx=" << i + nbf*j << "\n";
// 	  bijk.row(i + nbf*j) = arma::trans(psi_prefactor(bf_i, bf_j, rnodes)*psi_monomial_coeffs(bf_i, bf_j, rnodes));
// 	}
//       }
// 
//       //arma::mat cijnl(nprim*nprim,Nmax+1,arma::fill::zeros);
//       arma::mat cijnl(nbf*nbf,Nmax+1,arma::fill::zeros);
// 
//       for (size_t n = 0; n <= Nmax; n++) {
// 	for (size_t i = 0; i < nbf; i++) {
// 	  for (size_t j = 0; j < nbf; j++) {
// 	    for (size_t k = 0; k <= ndeg; k++) {
// 	      // std::cout << "cijnl(" << i << "+" << nprim << "*" << j << "," << n << ") +=" << bijk(i+nnodes*j,k) * pow(rs,k+0.5) * (iknl.get_Iknl(k,n,L,rmax/rs) - iknl.get_Iknl(k,n,L,rmin/rs)) << "\n";
// 	      cijnl(i+nbf*j,n) += (bijk(i+nbf*j,k) * pow(rs,k+0.5) * (iknl.get_Iknl(k,n,L,rmax/rs) - iknl.get_Iknl(k,n,L,rmin/rs)))/sqrt(Nnl(n,L));
// 	    }
// 	  }
// 	}
//       }
// 
//       return cijnl;
// 
    }

    arma::mat twoe_integral_wrk(double rmin, double rmax, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & pb, IknlTable & iknl, int L, double rs) {
      //double rmin(fem.element_begin(iel));
      //double rmax(fem.element_end(iel));
      double rmid = 0.5*(rmax + rmin);
      double rlen = 0.5*(rmax - rmin);

      //std::shared_ptr<const polynomial_basis::PolynomialBasis> pb(fem.get_basis(iel));

      arma::vec nodes = pb->get_nodes();
      arma::vec rnodes(rmid*arma::ones<arma::vec>(nodes.n_elem)+rlen*nodes);

      //std::cout << "nodes =\n" << nodes << "\n";
      //std::cout << "rnodes =\n" << rnodes << "\n";

      int nnodes = rnodes.n_elem;
      size_t nprim = pb->get_nprim();
      size_t nbf = pb->get_nbf();
      size_t ndeg = 2*nprim - 2; // maximum polynomial degree of product of basis funs (= Kmax)
      int Nmax = iknl.get_Nmax(); // maximum radial order of coulomb resolution: n = (0,...,Nmax)
      arma::uvec en(pb->get_enabled());
      
      //std::cout << "cr::twoe_integral nnodes=" << nnodes << "\n";
      //std::cout << "cr::twoe_integral nprim=" << nprim << "\n";
      //std::cout << "cr::twoe_integral nbf=" << nbf << "\n";
      //std::cout << "cr::twoe_integral ndeg=" << ndeg << "\n";
      //std::cout << "cr::twoe_integral enabled=" << en << "\n";

      arma::mat Nnl(iknl.get_Nnl());

      iknl.compute(rmin/rs);
      iknl.compute(rmax/rs);

      //arma::mat bijk(nnodes*nnodes,ndeg+1);
      arma::mat bijk(nbf*nbf,ndeg+1);

      for (size_t i = 0; i < nbf; i++) {
	for (size_t j = 0; j < nbf; j++) {
	  //std::cout << "bijk: i=" << i << " j=" << j << "\n";
	  // std::cout << "bijk(" << i << "+" << nnodes << "*" << j << ") =" << psi_prefactor(i, j, rnodes)*psi_monomial_coeffs(i, j, rnodes) << "\n";
	  //arma::uvec bf_iv = arma::find(en == i, 1);
	  //arma::uvec bf_jv = arma::find(en == j, 1);
	  //std::cout << "bijk: bf_iv=" << bf_iv << " bf_jv=" << bf_jv << "\n";
	  size_t bf_i = en[i];
	  size_t bf_j = en[j];
	  //std::cout << "bijk: bf_i=" << bf_i << " bf_j=" << bf_j << " idx=" << i + nbf*j << "\n";
	  //std::cout << "prefactor: " << psi_prefactor(bf_i, bf_j, rnodes) << "\n";
	  bijk.row(i + nbf*j) = arma::trans(psi_prefactor(bf_i, bf_j, rnodes)*psi_monomial_coeffs(bf_i, bf_j, rnodes));
	}
      }

      //arma::mat cijnl(nprim*nprim,Nmax+1,arma::fill::zeros);
      arma::mat cijnl(nbf*nbf,Nmax+1,arma::fill::zeros);

      for (size_t n = 0; n <= Nmax; n++) {
	for (size_t i = 0; i < nbf; i++) {
	  for (size_t j = 0; j < nbf; j++) {
	    for (size_t k = 0; k <= ndeg; k++) {
	      // std::cout << "cijnl(" << i << "+" << nprim << "*" << j << "," << n << ") +=" << bijk(i+nnodes*j,k) * pow(rs,k+0.5) * (iknl.get_Iknl(k,n,L,rmax/rs) - iknl.get_Iknl(k,n,L,rmin/rs)) << "\n";
	      // std::cout << "bijk = " << bijk(i+nbf*j,k) << "\n";
	      cijnl(i+nbf*j,n) += (bijk(i+nbf*j,k) * pow(rs,k+0.5) * (iknl.get_Iknl(k,n,L,rmax/rs) - iknl.get_Iknl(k,n,L,rmin/rs)))/sqrt(Nnl(n,L));
	    }
	  }
	}
      }

      return cijnl;

    }

    arma::vec IBF0l_quadrature(double rmin, double rmax, int l, const arma::vec & xq, const arma::vec & wq, helfem::cr::PhinlTable phinl, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, double rs) {
      double rmid = 0.5*(rmax + rmin);
      double rlen = 0.5*(rmax - rmin);
      int nprim = poly->get_nprim();
      int nq = wq.n_rows; // number of integral weights
      
      arma::vec rq = (rmid*arma::ones<arma::vec>(xq.n_rows) + rlen*xq)/rs;
      
      arma::vec phiq(xq.n_rows);
      
      //std::cout << "xq:\n" << xq << "\n";
      //std::cout << "wq:\n" << wq << "\n";
      //std::cout << "wq sum: " << arma::sum(wq) << "\n";
      //std::cout << "rq:\n" << rq << "\n";
      
      arma::mat bf(poly->eval_dnf(xq,0,rlen));
      
      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
      for(size_t fi=0;fi<bf.n_cols;fi++)
	for(size_t fj=0;fj<bf.n_cols;fj++)
	  bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);
      
      //std::cout << "bfprod:\n" << bfprod << "\n";
      
      arma::mat Nnl(phinl.get_Nnl()); double N0l(Nnl(0,l));

      for (int i = 0; i < nq; i++) {
	phiq[i] = phinl.phi0l(l,rq[i]);
      }
      
      //std::cout << "phiq:\n" << phiq << "\n";
      
      arma::vec ints(bfprod.n_cols);
      
      for (int i = 0; i < ints.n_rows; i++) {
	ints[i] = arma::sum(bfprod.col(i) % phiq % wq)*rlen/sqrt(rs)/sqrt(N0l);
      }
      
      return ints;
    }

    arma::mat IBFnl_quadrature(double rmin, double rmax, const arma::vec & xq, const arma::vec & wq, helfem::cr::PhinlTable phinl, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, double rs) {
      // this could be folded into twoe_integral_quadrature()
      double rmid = 0.5*(rmax + rmin);
      double rlen = 0.5*(rmax - rmin);
      int nprim = poly->get_nprim();
      int nq = wq.n_rows; // number of integral weights

      arma::vec rq = (rmid*arma::ones<arma::vec>(xq.n_rows) + rlen*xq)/rs;

      arma::mat bf(poly->eval_dnf(xq,0,rlen));

      arma::mat bfprod(bf.n_rows,bf.n_cols*bf.n_cols);
      for(size_t fi=0;fi<bf.n_cols;fi++)
	for(size_t fj=0;fj<bf.n_cols;fj++)
	  bfprod.col(fi*bf.n_cols+fj)=bf.col(fi)%bf.col(fj);

      for(size_t i=0;i<bfprod.n_cols;i++) { // multiply weights
        bfprod.col(i)%=wq;
      }

      //std::cout << "bfprod rows: " << bfprod.n_rows << " cols: " << bfprod.n_cols << "\n";

      arma::mat Nnl(phinl.get_Nnl());

      int nmax = Nnl.n_rows - 1;
      int lmax = Nnl.n_cols - 1;

      //std::cout << "Nnl rows: " << Nnl.n_rows << " cols: " << Nnl.n_cols << "\n";

      for (int j = 0; j < nq; j++) {
	phinl.compute(rq[j]);
      }


      //arma::cube phiq(nmax+1,lmax+1,nq);
      arma::mat phiq((nmax+1)*(lmax+1),nq);

      //std::cout << "phiq rows: " << phiq.n_rows << " cols: " << phiq.n_cols << "\n";

      arma::vec vNnl(arma::sqrt(arma::vectorise(Nnl)));

      //std::cout << "vNnl rows: " << vNnl.n_rows << " cols: " << vNnl.n_cols << "\n";

      for (int j = 0; j < nq; j++) {
	arma::mat phinl_rq_mat(phinl.get_Phinl(rq[j]));
	//std::cout << "phinl_rq_mat rows: " << phinl_rq_mat.n_rows << " cols: " << phinl_rq_mat.n_cols << "\n";
	arma::vec phinl_rq_vec(arma::vectorise(phinl_rq_mat)/vNnl);
	//std::cout << "phinl_rq_vec rows: " << phinl_rq_vec.n_rows << " cols: " << phinl_rq_vec.n_cols << "\n";
	phiq.col(j) = phinl_rq_vec;
      }

      arma::mat ints((nmax+1)*(lmax+1),bfprod.n_cols);

      //std::cout << "ints rows: " << ints.n_rows << " cols: " << ints.n_cols << "\n";

      //arma::mat ones_mat(nmax+1,lmax+1);
      //arma::vec ones_vec((nmax+1)*(lmax+1));

      // for (int i = 0; i < bfprod.n_cols; i++) {
      // 	arma::vec bf_times_weights(bfprod.col(i) % wq);
      // 	std::cout << "bf_times_weights rows: " << bf_times_weights.n_rows << " cols: " << bf_times_weights.n_cols << "\n";
      // 	arma::vec phiq_bf(phiq*bf_times_weights);
      // 	std::cout << "phiq_bf rows: " << phiq_bf.n_rows << " cols: " << phiq_bf.n_cols << "\n";
      // 	ints.col(i) = rlen/sqrt(rs)*(phiq_bf/sqrt(arma::vectorise(Nnl)));
      // 	//ints.col(i) = arma::sum(phiq_bf)*rlen/sqrt(rs)/sqrt(Nnl);
      // }

      ints = rlen/sqrt(rs) * (phiq * bfprod) ; // / arma::vectorise(Nnl);


      return arma::trans(ints);
    }



  }
}

