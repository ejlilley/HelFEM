#include "cr/cr_spheroidal_potentials.h"
#include "cr/cr_utils.h"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_pow_int.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_poly.h>

namespace helfem {
  namespace cr {

    bool operator<(const phinlm_table_t & lh, const phinlm_table_t & rh) {
      return lh.mu < rh.mu;
    }

    PhinlmTable::PhinlmTable() {
      CR=-1;
      Nmax=-1;
      Lmax=-1;
      Mmax=-1;
      Rh=0.0;
    }

    PhinlmTable::~PhinlmTable() {
    }

    PhinlmTable::PhinlmTable(int CR_, int Nmax_, int Lmax_, int Mmax_, double Rh_) : CR(CR_), Nmax(Nmax_), Lmax(Lmax_), Mmax(Mmax_), Rh(Rh_) {

      if (Nmax < 0 || Lmax < 0 || Mmax < 0 || Rh <= 0.0) {
	std::ostringstream oss;
	oss << "Invalid CR Nmax or Lmax or Mmax or Rh\n";
	throw std::logic_error(oss.str());
      }
      //      if (CR < 0 || CR > 1) {
      if (CR != 0) {
	std::ostringstream oss;
	oss << "Invalid CR \"" << CR << "\" (not implemented)\n";
	throw std::logic_error(oss.str());
      }

    }

    double PhinlmTable::phi0lm(int l, int m, double mu) {
      if (CR == 0) {
	//return pow(r,l)/pow(1 + pow(r,1/alpha),(1+2*l)*alpha);
      } else if (CR == 1) {
	//return -4*M_PI/(1 + 2*l)*( pow(r,l)*(1+r)*exp(-r) + pow(r,-l-1)*gsl_sf_gamma(2*l+3)*gsl_sf_gamma_inc_P(2*l+3,r) );
      } else {
	return -1;
      }
    }


    arma::vec PhinlmTable::pnlm(int n, int l, int m, double s) {
      if (CR != 1) {
	return {};
      }

      //if (CR == 0) {
      //}

      //if (CR == 1) {
      //}

    }

    arma::mat PhinlmTable::pnlm_mat(int n, int l, int m, const arma::vec & s) {
      int sn = s.n_elem;

      if (CR != 1) {
	return {};
      }

      if (CR == 1) {
	if (n == 0) {
	  return arma::trans(arma::ones(sn));
	}
      }

    }


    arma::cube PhinlmTable::get_Nnlm() {
      if (!PhinlmTable::Nnlm.size()) {
	PhinlmTable::compute_Nnlm();
      }
      return PhinlmTable::Nnlm;

    }

    void PhinlmTable::compute_Nnlm() {
      PhinlmTable::Nnlm.resize(Nmax+1,Lmax+1,Mmax+1);

      if (CR == 0) { // does NOT include spherical harmonic normalisation factor Jlm
	for (int l = 0; l <= Lmax; l++) {
	  double ld(l);
	  for (int m = 0; m <= l && m <= Mmax; m++) {
	    double md(abs(m));

	    PhinlmTable::Nnlm(0,l,m) = (1+ld+md)*(2+ld+md)*gsl_sf_beta(ld+1.5,md+1)/(8*M_PI*pow(Rh,4));

	    for (int n = 1; n <= Nmax; n++) {
	      double nd(n);
	      PhinlmTable::Nnlm(n,l,m) = PhinlmTable::Nnlm(n-1,l,m)*(nd*(md + nd)*(1 + 2*ld + 2*nd)*(-1 + ld + md + 2*nd)*(ld + md + 2*nd)*(1 + ld + md + 2*nd)*(2 + ld + md + 2*nd)*(1 + 2*ld + 2*md + 2*nd))/(4*(-1 + 2*ld + 2*md + 4*nd)*pow(1 + 2*ld + 2*md + 4*nd,2)*(3 + 2*ld + 2*md + 4*nd));
	    }
	    
	  }
	}
      }
      
    }

    size_t PhinlmTable::get_index(double mu, bool check) const {
      phinlm_table_t p;
      p.mu=mu;

      std::vector<phinlm_table_t>::const_iterator low(std::lower_bound(stor.begin(),stor.end(),p));
      if(check && low == stor.end()) {
        std::ostringstream oss;
        oss << "Could not find mu=" << mu << " on the list!\n";
        throw std::logic_error(oss.str());
      }

      // Index is
      size_t idx(low-stor.begin());
      if(check && (stor[idx].mu != mu)) {
        std::ostringstream oss;
        oss << "Map error: tried to get mu = " << mu << " but got mu = " << stor[idx].mu << "!\n";
        throw std::logic_error(oss.str());
      }

      return idx;
    }


    void PhinlmTable::compute(double mu) {
      // This routine could also be trivially parallelised w.r.t. mu
      phinlm_table_t entry;

      entry.mu=mu;
      
      entry.Phinlm.zeros(Lmax+1,Mmax+1,Nmax+1);
      
      if (CR == 0) { // "Plummer/scale-free"
	
	double xi = (pow(sinh(mu),2) - 1)/(pow(sinh(mu),2) + 1);

	for (int l = 0; l <= Lmax; l++) {
	double ld(l);

	  for (int m = 0; m <= Mmax && m <= l; m++) {
	    double md(abs(m));

	    double zeroth_order(pow(sinh(mu),abs(m))/pow(cosh(mu),abs(m)+l+1));

	    arma::vec prefactors(Nmax+1);
	    prefactors[0] = 1.0;
	    
	    for (int n = 1; n <= Nmax; n++) {
	      double nd(n);
	      prefactors[n] = prefactors[n-1] * (-0.125) * (nd*(-1 + ld + md + 2*nd)*(ld + md + 2*nd)*(1 + 2*ld + 2*md + 2*nd))/((-0.5 + ld + md + 2*nd)*(0.5 + ld + md + 2*nd));
	    }

	    arma::vec polys(jacobi_n(Nmax,ld+0.5,md,xi));

	    entry.Phinlm.tube(l,m) = 1/Rh*zeroth_order*(prefactors % polys);

	  }
	  
	}
	
      } else if (CR == 1) { // "isochrone"
	for (int l = 0; l <= Lmax; l++) {

	  for (int m = 0; m <= Mmax && m <= l; m++) {
	    
	    for (int n = 0; n <= Nmax; n++) {
	      //
	    }

	  }

	}

      }
      
      if(!stor.size())
	stor.push_back(entry);
      else
	// Insert at lower bound
	stor.insert(stor.begin()+get_index(mu,false),entry);
    }
    
    
    double PhinlmTable::get_Phinlm(int n, int l, int m, double mu) const {
      if(get_index(mu)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Phinlm(" << n << "," << l << "," << m << "," << mu << "): index " << get_index(mu) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(mu)].Phinlm(l,m,n);
    }


    arma::vec PhinlmTable::get_Phinlm(int n, int l, int m, const arma::vec & mu) const {
      arma::vec phinlm_vec(mu.n_elem);
      for(size_t i=0;i<mu.n_elem;i++)
        phinlm_vec(i)=get_Phinlm(l,m,n,mu(i));
      return phinlm_vec;
    }

    arma::cube PhinlmTable::get_Phinlm(double mu) const {
      if(get_index(mu)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Phinlm(" << mu << "): index " << get_index(mu) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(mu)].Phinlm;
    }

  }

}

