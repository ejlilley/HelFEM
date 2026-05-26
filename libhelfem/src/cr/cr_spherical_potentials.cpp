#include "cr/cr_spherical_potentials.h"
#include "cr/cr_utils.h"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_poly.h>

namespace helfem {
  namespace cr {

    bool operator<(const phinl_table_t & lh, const phinl_table_t & rh) {
      return lh.r < rh.r;
    }

    PhinlTable::PhinlTable() {
      CR=-1;
      Nmax=-1;
      Lmax=-1;
      alpha=-1.0;
    }

    PhinlTable::~PhinlTable() {
    }

    PhinlTable::PhinlTable(int CR_, int Nmax_, int Lmax_, double alpha_) : CR(CR_), Nmax(Nmax_), Lmax(Lmax_), alpha(alpha_) {
      //std::cout << "initialised PhinlTable with Nmax=" << Nmax << ", Lmax=" << Lmax << "\n";
      if (Nmax < 0 || Lmax < 0) {
	std::ostringstream oss;
	oss << "Invalid CR Nmax or Lmax\n";
	throw std::logic_error(oss.str());
      }
      if (alpha < 0.5) {
	std::ostringstream oss;
	oss << "Invalid CR alpha parameter\n";
	throw std::logic_error(oss.str());
      }
      if (CR > 1) {
	std::ostringstream oss;
	oss << "Invalid CR \"" << CR << "\" (not implemented)\n";
	throw std::logic_error(oss.str());
      }

    }

    double PhinlTable::phi0l(int l, double r) {
      if (CR == 0) {
	return pow(r,l)/pow(1 + pow(r,1/alpha),(1+2*l)*alpha);
      } else if (CR == 1) {
	return -4*M_PI/(1 + 2*l)*( pow(r,l)*(1+r)*exp(-r) + pow(r,-l-1)*gsl_sf_gamma(2*l+3)*gsl_sf_gamma_inc_P(2*l+3,r) );
      } else {
	return -1;
      }
    }


//     double PhinlTable::Nnl(int n, int l) {
//       if (CR == 0) {
// 	double mu = alpha*(1+2*l);
// 	return (2*n+2*mu+1)*mu*gsl_sf_gamma(mu+2)/(sqrt(M_PI)*pow(alpha,2*n+1)*pow(2,4*n+2*mu+4)*(mu+0.5)*gsl_sf_gamma(mu+1.5)) * gsl_sf_fact(n)*gsl_sf_poch(mu,n)*gsl_sf_poch(mu+2,n)*gsl_sf_poch(2*mu+1)/pow(gsl_sf_poch(mu+1.5,n),2);
// 	//return (2n+2μ+1)*μ*Gamma[μ+2]/(Sqrt[Pi]*α^(2n+1)*2^(4n+2μ+4)*(μ+1/2)*Gamma[μ+3/2]) * (n!)*Pochhammer[μ,n]*Pochhammer[μ+2,n]*Pochhammer[2μ+1,n]/Pochhammer[μ+3/2,n]^2
//       } else {
// 	return -1;
//       }
//     }

    arma::vec PhinlTable::pnl(int n, int l, double s) {
      if (CR != 1) {
	return {};
      }

      if (CR == 1) { // only implemented for Slater-type so far
	// p_{n,l}(s) = s*p_{n-1,l}(s) - beta_{n-1,l}*p_{n-2,l}(s)
	// then let π_{n,l}(s) = i^n p_{n,l}(is) (π_{n,l} is "real form" of p_{n,l})
	// => π_{n,l}(s) = -s*π_{n-1,l}(s) + beta_{n-1,l}*π_{n-2,l}(s)
	// (Taylor coeffs are proportional to π_{n,l}(k+1/2))
	if (n == 0) {
	  return { 1.0 };
	}

	if (n == 1) {
	  return { 1.0, -s };
	}

	arma::vec pvec(n+1);
	pvec[0] = 1.0;
	pvec[1] = -s; // sign-flip because real form

	for (int i = 2; i <= n; i++) {
	  double ld(l);
	  double id(i); // cast to double first!
	  double beta = ((-1 + id)*(1 + 2*ld + id)*(-3 + 2*ld + 2*id)*(3 + 2*ld + 2*id))/(4*(-1 + 4*(ld + id)*(ld + id)));
	  pvec[i] = -s*pvec[i-1] + beta*pvec[i-2];  // sign-flip because real form
	}
	return pvec;
      }

    }

    arma::mat PhinlTable::pnl_mat(int n, int l, const arma::vec & s) {
      int sn = s.n_elem;

      if (CR != 1) {
	return {};
      }

      if (CR == 1) {
	if (n == 0) {
	  return arma::trans(arma::ones(sn));
	}

	arma::mat pmat(n+1,sn);
	pmat.row(0) = arma::trans(arma::ones(sn));
	pmat.row(1) = -1*arma::trans(s); // sign-flip because real form

	if (n == 1) {
	  return pmat;
	}

	for (int i = 2; i <= n; i++) {
	  double ld(l);
	  double id(i); // cast to double first!
	  double beta = ((-1 + id)*(1 + 2*ld + id)*(-3 + 2*ld + 2*id)*(3 + 2*ld + 2*id))/(4*(-1 + 4*(ld + id)*(ld + id)));

	  pmat.row(i) = -arma::trans(s)%pmat.row(i-1) + beta*pmat.row(i-2);  // sign-flip because real form
	}
	return pmat;
      }

    }



    arma::mat PhinlTable::get_Nnl() {
      if (!PhinlTable::Nnl.size()) {
	PhinlTable::compute_Nnl();
      }
      return PhinlTable::Nnl;

    }

    void PhinlTable::compute_Nnl() {
      PhinlTable::Nnl.resize(Nmax+1,Lmax+1);

      //arma::vec ret(n0+1);

	for (int l = 0; l <= Lmax; l++) {
	  for (int n = 0; n <= Nmax; n++) {
	    if (CR == 0) {
	      double mu = alpha*(1+2*l);
	      PhinlTable::Nnl(n,l) = (2*n+2*mu+1)*mu*gsl_sf_gamma(mu+2)/(sqrt(M_PI)*pow(alpha,2*n+1)*pow(2,4*n+2*mu+4)*(mu+0.5)*gsl_sf_gamma(mu+1.5)) * gsl_sf_fact(n)*gsl_sf_poch(mu,n)*gsl_sf_poch(mu+2,n)*gsl_sf_poch(2*mu+1,n)/pow(gsl_sf_poch(mu+1.5,n),2);
	    } else if (CR == 1) {
	      PhinlTable::Nnl(n,l) = M_PI*pow(2,-1-2*l-2*n)*(2*n+2*l+5)/(1 + 2*l + 2*n) * gsl_sf_fact(n) * gsl_sf_gamma(n+2*l+3);
	    }
	  }
	}
	// Nnl = ret;
    }

    size_t PhinlTable::get_index(double r, bool check) const {
      phinl_table_t p;
      p.r=r;

      std::vector<phinl_table_t>::const_iterator low(std::lower_bound(stor.begin(),stor.end(),p));
      if(check && low == stor.end()) {
        std::ostringstream oss;
        oss << "Could not find r=" << r << " on the list!\n";
        throw std::logic_error(oss.str());
      }

      // Index is
      size_t idx(low-stor.begin());
      if(check && (stor[idx].r != r)) {
        std::ostringstream oss;
        oss << "Map error: tried to get r = " << r << " but got r = " << stor[idx].r << "!\n";
        throw std::logic_error(oss.str());
      }

      return idx;
    }


    void PhinlTable::compute(double r) {
      // This routine could also be trivially parallelised w.r.t. r
      phinl_table_t entry;

      //std::cout << "begin computing Phinl at r=" << r << "\n";
      //std::cout << "Phinl: Nmax=" << Nmax << ", Lmax=" << Lmax << "\n";
      
      entry.r=r;
      
      entry.Phinl.zeros(Nmax+1,Lmax+1);
      
      if (CR == 0) { // Zhao
	
	double xi = (pow(r, 1/alpha) - 1)/(pow(r, 1/alpha) + 1);
	
	for (int l = 0; l <= Lmax; l++) {

	  //std::cout << "Phinl begin l=" << l << "\n";
	  double mu = alpha*(1 + 2*l);

	  double zeroth_order = phi0l(l,r);
	  
	  double poly_array[Nmax+1] {};
	  
	  gsl_sf_gegenpoly_array(Nmax, alpha*(1+2*l)+0.5, xi, poly_array);

	  arma::vec arma_poly_array = arma::vec(poly_array, Nmax+1);
	  arma::vec prefactor_array(Nmax+1);
	  
	  for (int n = 0; n <= Nmax; n++) {
	    prefactor_array[n] = gsl_sf_fact(n) * gsl_sf_poch(mu,n) / (pow(2, 2*n) * pow(alpha, n) * gsl_sf_poch(mu + 0.5,n));
	  }
	  
	  entry.Phinl.col(l) = zeroth_order * (prefactor_array % arma_poly_array);
	  
	}
	
      } else if (CR == 1) { // Slater
	for (int l = 0; l <= Lmax; l++) {

	  double zeroth_order = phi0l(l,r);

	  // int n = Nmax; // incorrectly using nmax to determine which type of phinl evaluation to use

	  //double nd(n);
	  double ld(l);

	  int taylor_order(Nmax+l+21);

	  int Nmaxmax(100); // the (higher than Nmax!) maximum order of polynomial we need

	  arma::vec laguerre_polys(genlaguerre_n(Nmaxmax, 2*ld+4, 2*r)); // all the Laguerre polynomials that we want to use with both the forwards and backwards recursion scheme
	  
	  //arma::vec forwards_norm(Nmaxmax+1); // pre-factor that ensures phinl is real-valued and monically-normalised
	  arma::vec forwards_norm(Nmax+1); // pre-factor that ensures phinl is real-valued and monically-normalised
	  
	  for (int i = 0; i <= Nmax; i++) {
	    double id(i);
	    forwards_norm[i] = pow(2,i)*gsl_sf_poch(ld+0.5,i)*gsl_sf_poch(l+2,i)/gsl_sf_poch(i+2*l+3,i);
	  }

	  //std::cout << "Slater forwards_norm=\n" << forwards_norm;
	  
	  arma::vec forwards_poly_prefactors(Nmaxmax+1);
	  
	  for (int i = 0; i <= Nmaxmax; i++) {
	    double id(i);
	    forwards_poly_prefactors[i] = minusonepow(i)*8*M_PI*gsl_sf_fact(id)*(2*id + 2*ld + 5)/((2*ld+1)*(ld+2)*(2*ld+3)*gsl_sf_poch(2*l+5,i));
	  }
	  
	  //std::cout << "Slater forwards_poly_prefactors=\n" << forwards_poly_prefactors;

	  double forwards_poly_r_factor(pow(r,l+2)*exp(-r));

	  arma::vec s(taylor_order+1);
	  for (int i = 0; i <= taylor_order; i++) {
	    s[i] = i + 0.5;
	  }

	  // matrix of Pnl evaluations
	  arma::mat pmat(PhinlTable::pnl_mat(Nmax,l,s));

	  arma::mat taylor_prefactors(Nmax+1,taylor_order+1);

	  for (int i = 0; i <= Nmax; i++) { // outer loop not really necessary here
	    for (int j = 0; j <= taylor_order; j++) {
	      if (j < l || j == l+1) { // certain Taylor coeffs are 0
		taylor_prefactors(i,j) = 0.0;
	      } else {
		double id(i);
		double jd(j);
		taylor_prefactors(i,j) = (4*minusonepow(j + l)*(-1 + jd - ld)*M_PI)/((1 + jd + ld)*gsl_sf_fact(j - l)); // the monic normalisation κnl is already taken care of by Pnl, so no n-dependence here
	      }
	    }
	  }

	  arma::mat taylor_coeffs(arma::trans(pmat % taylor_prefactors));

	  arma::vec taylor_eval(Nmax+1); // contains phinl evaluated via Taylor series

	  for (int i = 0; i <= Nmax; i++) {
	    double* pcol = taylor_coeffs.colptr(i);
	    // evaluate Taylor series using Horner form
	    taylor_eval[i] = gsl_poly_eval(pcol, taylor_order+1, r);
	  }
	  
	  arma::vec forwards_eval(Nmax+1); // contains phinl evaluated via forwards recurrence
	  
	  forwards_eval[0] = zeroth_order;
	  forwards_eval[1] = forwards_eval[0] + 8*M_PI/(2*ld+1)*pow(r,l)*exp(-r)*(1+r);
	  
	  for (int i = 2; i <= Nmax; i++) {
	    forwards_eval[i] = forwards_eval[i-2] + forwards_poly_r_factor*forwards_poly_prefactors[i-2]*laguerre_polys[i-2];
	  }

	  //forwards_eval = forwards_eval % forwards_norm.subvec(0,Nmax);
	  forwards_eval = forwards_eval % forwards_norm;
	  
	  arma::vec backwards_eval(Nmaxmax+1);
	  
	  backwards_eval[Nmaxmax] = -forwards_poly_prefactors[Nmaxmax]*forwards_poly_r_factor*laguerre_polys[Nmaxmax];
	  backwards_eval[Nmaxmax-1] = -forwards_poly_prefactors[Nmaxmax-1]*forwards_poly_r_factor*laguerre_polys[Nmaxmax-1];
	  
	  for (int i = 2; i <= Nmaxmax; i++) {
	    backwards_eval[Nmaxmax-i] = backwards_eval[Nmaxmax-i+2] - forwards_poly_r_factor*forwards_poly_prefactors[Nmaxmax-i]*laguerre_polys[Nmaxmax-i];
	  }
	  
	  //arma::vec backwards_eval_subvec(n+1);
	  //backwards_eval_subvec = backwards_eval.subvec(0,n) % forwards_norm.subvec(0,n);

	  //arma::vec backwards_eval_subvec(backwards_eval.subvec(0,Nmax) % forwards_norm.subvec(0,Nmax));
	  arma::vec backwards_eval_subvec(backwards_eval.subvec(0,Nmax) % forwards_norm); // contains phinl evaluated via backwards recurrence

	  // Now decide which combinations of (n,l,r) get which method of evaluation
	  // (technically we were wasteful in that we've already computed all 3 methods for each (n,l,r), but it was all vectorised enough that hopefully we still win on performance)
	  for (int n = 0; n <= Nmax; n++) {
	    double nd(n);
	    // The conditions in the next few if-statements are of
	    // course empirically-determined, and could perhaps be
	    // improved with further numerical experimentation.
	    if (r<(nd/4+1.1*ld)/20) { // use Taylor series
	      
	      //std::cout << "Slater Taylor eval with r=" << r << " n=" << n << " l=" << l << "\n";
	      //entry.Phinl.col(l) = taylor_eval;
	      entry.Phinl(n,l) = taylor_eval(n);
	      
	    } else if ((r<1+(0.3*nd+4*ld)/40)||((nd/4+1.5*ld)>15 && (nd/4+1.5*ld)<20)) { // forwards recursion
	      
	      //std::cout << "Slater forwards eval with r=" << r << " n=" << n << " l=" << l << "\n";
	      //entry.Phinl.col(l) = forwards_eval;
	      entry.Phinl(n,l) = forwards_eval(n);
	      
	    } else if ((r<-15+nd/2+4.5*ld)&&((nd/4+1.5*ld)>15)) { // backwards recursion
	      
	      //std::cout << "Slater backwards eval with r=" << r << " n=" << n << " l=" << l << "\n";
	      //entry.Phinl.col(l) = backwards_eval_subvec;
	      entry.Phinl(n,l) = backwards_eval_subvec(n);
	      
	    } else {
	      // default to forwards recurrence
	      entry.Phinl(n,l) = forwards_eval(n);
	      //std::cout << "Unhandled case of Slater phinl evaluation! n=" << n << " l=" << l << " r=" << r << "\n";
	    }
	  }
	}

      }
      
      if(!stor.size())
	stor.push_back(entry);
      else
	// Insert at lower bound
	stor.insert(stor.begin()+get_index(r,false),entry);
    }
    
    
    double PhinlTable::get_Phinl(int n, int l, double r) const {
      //std::cout << "get_Phinl_(" << n << "," << l << ")(" << r << ")\n";
      if(get_index(r)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Phinl(" << n << "," << l << "," << r << "): index " << get_index(r) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(r)].Phinl(n,l);
    }


    arma::vec PhinlTable::get_Phinl(int n, int l, const arma::vec & r) const {
      arma::vec phinl_vec(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++)
        phinl_vec(i)=get_Phinl(n,l,r(i));
      return phinl_vec;
    }

    arma::mat PhinlTable::get_Phinl(double r) const {
      if(get_index(r)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Phinl(" << r << "): index " << get_index(r) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(r)].Phinl;
    }


    bool operator<(const iknl_table_t & lh, const iknl_table_t & rh) {
      return lh.r < rh.r;
    }

    IknlTable::IknlTable() {
      CR=-1;
      Kmax=-1;
      Nmax=-1;
      Lmax=-1;
      alpha=-1.0;
    }

    IknlTable::~IknlTable() {
    }

    IknlTable::IknlTable(int CR_, int Kmax_, int Nmax_, int Lmax_, double alpha_) : CR(CR_), Kmax(Kmax_), Nmax(Nmax_), Lmax(Lmax_), alpha(alpha_), phinl(CR_,Nmax_,Lmax_,alpha_) {
      // (Rely on phinl initialisation to throw appropriate errors)
      //
      //if (Nmax < 0 || Lmax < 0 || Kmax < 0) {
      //	std::ostringstream oss;
      //	oss << "Invalid Nmax or Lmax or Kmax\n";
      //	throw std::logic_error(oss.str());
      //}
      //if (CR == 0 && alpha < 0.5) {
      //	std::ostringstream oss;
      //	oss << "Invalid alpha for Zhao CR\n";
      //	throw std::logic_error(oss.str());
      //}
      //if (CR > 0) {
      //	std::ostringstream oss;
      //	oss << "Other CRs not implemented yet!\n";
      //	throw std::logic_error(oss.str());
      //}
      //std::cout << "initialise internal phinl with Nmax=" << Nmax << ", Lmax=" << Lmax << "\n";
      //PhinlTable phinl(CR,Nmax,Lmax,alpha);
    }

    double IknlTable::Ik0l(int k, int l, double r) {
      if (CR == 0) {
	double chi = pow(r,(1/alpha))/(1+pow(r,(1/alpha)));
//	if (chi > 0.999) {
//	  double eps = 1 - chi;
//	  if (k==l) {
//	    return -gsl_sf_log(eps)*pow(chi, alpha*(1+2*k));
//	  } else if (l < k) {
//	    return pow(chi, alpha*(1+k+l))*pow(eps,alpha*(l-k))/(k-l);
//	  } else {
//	    return alpha*gsl_sf_beta(alpha*(1+k+l),alpha*(l-k));
//	  }
//	} else {
	double res = alpha*beta_inc(alpha*(1+k+l),alpha*(l-k),chi);
	std::cout << "Ik0l(" << k << "," << l << "," << r << ") = " << res << "\n";
	return res;
//	}
//	return alpha*gsl_sf_beta(alpha*(1+k+l),alpha*(l-k))*gsl_sf_beta_inc(alpha*(1+k+l),alpha*(l-k),chi);
      } else {
	return -1;
      }
    }

    arma::mat IknlTable::get_Nnl() {
      return phinl.get_Nnl();
    }


    size_t IknlTable::get_index(double r, bool check) const {
      iknl_table_t p;
      p.r=r;

      std::vector<iknl_table_t>::const_iterator low(std::lower_bound(stor.begin(),stor.end(),p));
      if(check && low == stor.end()) {
        std::ostringstream oss;
        oss << "Could not find r=" << r << " on the list!\n";
        throw std::logic_error(oss.str());
      }

      // Index is
      size_t idx(low-stor.begin());
      if(check && (stor[idx].r != r)) {
        std::ostringstream oss;
        oss << "Map error: tried to get r = " << r << " but got r = " << stor[idx].r << "!\n";
        throw std::logic_error(oss.str());
      }

      //std::cout << "found Iknl entry for r=" << r << " at idx=" << idx << "\n";

      return idx;
    }

    void IknlTable::compute(double r) {
        iknl_table_t entry;

	//std::cout << "computing Iknl at r=" << r << "\n";

	phinl.compute(r);

        entry.r=r;

	entry.Iknl.zeros(Kmax+1,Nmax+1,Lmax+1);

	if (CR == 0) { // Zhao

	  for (int k = 0; k <= Kmax; k++) {
	    
	    //std::cout << "Iknl begin k=" << k << "\n";

	    for (int l = 0; l <= Lmax; l++) {

	      //std::cout << "Iknl begin l=" << l << "\n";
	      
	      entry.Iknl(k,0,l) = Ik0l(k,l,r);
	      
	      for (int n = 1; n <= Nmax; n++) {

		//std::cout << "Iknl begin n=" << n << "\n";
		
		// special handling when n=1 to avoid division by zero & out of bounds access on entry.Iknl

		double b = (n==1 ? 0.0 : ((n-1)*(-1 + (n-1) + alpha + 2*l*alpha)*(1 + (n-1) + alpha + 2*l*alpha)* ((n-1) + 2*alpha + 4*l*alpha))/(4*pow(alpha,2)*(-1 + 2*(n-1) + (2 + 4*l)*alpha)* (1 + 2*(n-1) + (2 + 4*l)*alpha)));
		
		double p = phinl.get_Phinl(n-1,l,r);

		//std::cout << "beta_(" << n-1 << "," << l << ") = " << b << ", phi_(" << n-1 << "," << l << ")(" << r << ") = " << p << "\n";
		
		entry.Iknl(k,n,l) = (n==1 ? -pow(r,k+1)*p + (k+0.5)*entry.Iknl(k,0,l) : -pow(r,k+1)*p + (k+0.5)*entry.Iknl(k,n-1,l) + b*entry.Iknl(k,n-2,l));
		
	      }
	      
	    }

	  }

	}

        if(!stor.size())
          stor.push_back(entry);
        else
          // Insert at lower bound
          stor.insert(stor.begin()+get_index(r,false),entry);
    }

    double IknlTable::get_Iknl(int k, int n, int l, double r) const {
      if(get_index(r)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Iknl(" << k << "," << n << "," << l << "," << r << "): index " << get_index(r) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(r)].Iknl(k,n,l);
    }

    arma::vec IknlTable::get_Iknl(int k, int n, int l, const arma::vec & r) const {
      arma::vec iknl_vec(r.n_elem);
      for(size_t i=0;i<r.n_elem;i++)
        iknl_vec(i)=get_Iknl(k,n,l,r(i));
      return iknl_vec;
    }

    int IknlTable::get_Nmax() {
      return Nmax;
    }

  }

}




