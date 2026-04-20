#include "cr/cr_spherical_potentials.h"
#include "cr/cr_utils.h"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_sf_gegenbauer.h>

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
	oss << "Invalid Nmax or Lmax\n";
	throw std::logic_error(oss.str());
      }
      if (CR == 0 && alpha < 0.5) {
	std::ostringstream oss;
	oss << "Invalid alpha for Zhao CR\n";
	throw std::logic_error(oss.str());
      }
      if (CR > 0) {
	std::ostringstream oss;
	oss << "Other CRs not implemented yet!\n";
	throw std::logic_error(oss.str());
      }

    }

    double PhinlTable::phi0l(int l, double r) {
      if (CR == 0) {
	return pow(r,l)/pow(1 + pow(r,1/alpha),(1+2*l)*alpha);
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

    arma::mat PhinlTable::get_Nnl() {
      if (!PhinlTable::Nnl.size()) {
	PhinlTable::compute_Nnl();
      }
      return PhinlTable::Nnl;

    }

    void PhinlTable::compute_Nnl() {
      PhinlTable::Nnl.resize(Nmax+1,Lmax+1);

      //arma::vec ret(n0+1);

      if (CR == 0) {
	for (int l = 0; l <= Lmax; l++) {
	  for (int n = 0; n <= Nmax; n++) {
	    double mu = alpha*(1+2*l);
	    PhinlTable::Nnl(n,l) = (2*n+2*mu+1)*mu*gsl_sf_gamma(mu+2)/(sqrt(M_PI)*pow(alpha,2*n+1)*pow(2,4*n+2*mu+4)*(mu+0.5)*gsl_sf_gamma(mu+1.5)) * gsl_sf_fact(n)*gsl_sf_poch(mu,n)*gsl_sf_poch(mu+2,n)*gsl_sf_poch(2*mu+1,n)/pow(gsl_sf_poch(mu+1.5,n),2);
	  //return (2n+2μ+1)*μ*Gamma[μ+2]/(Sqrt[Pi]*α^(2n+1)*2^(4n+2μ+4)*(μ+1/2)*Gamma[μ+3/2]) * (n!)*Pochhammer[μ,n]*Pochhammer[μ+2,n]*Pochhammer[2μ+1,n]/Pochhammer[μ+3/2,n]^2
	  }
	}
	// Nnl = ret;
      }
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
	    //std::cout << "Phinl begin n=" << n << "\n";

	    //prefactor_array[n] = pow(2,n)*gsl_sf_fact(n)*gsl_sf_poch(l+0.5,n)*gsl_sf_poch(l+1.5,n)/(gsl_sf_poch(n+2*l+2,n)*gsl_sf_poch(1+2*alpha*(1+2*l),n));
	    
	    prefactor_array[n] = gsl_sf_fact(n) * gsl_sf_poch(mu,n) / (pow(2, 2*n) * pow(alpha, n) * gsl_sf_poch(mu + 0.5,n));

	    //std::cout << "computing phinl_(" << n << "," << l << ")(" << r << ") = " << zeroth_order << "*" << prefactor_array[n] << "*" << arma_poly_array[n] << " = " << zeroth_order*prefactor_array[n]*arma_poly_array[n] << "\n";
	  }
	  
	  entry.Phinl.col(l) = zeroth_order * (prefactor_array % arma_poly_array);
	  
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
      if (Nmax < 0 || Lmax < 0 || Kmax < 0) {
	std::ostringstream oss;
	oss << "Invalid Nmax or Lmax or Kmax\n";
	throw std::logic_error(oss.str());
      }
      if (CR == 0 && alpha < 0.5) {
	std::ostringstream oss;
	oss << "Invalid alpha for Zhao CR\n";
	throw std::logic_error(oss.str());
      }
      if (CR > 0) {
	std::ostringstream oss;
	oss << "Other CRs not implemented yet!\n";
	throw std::logic_error(oss.str());
      }
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
	return alpha*beta_inc(alpha*(1+k+l),alpha*(l-k),chi);
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




