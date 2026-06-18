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

      if (Nmax < 0 || Lmax < 0 || Mmax < 0 || Rh <= 0.0 || Mmax > Lmax) {
	std::ostringstream oss;
	oss << "Invalid CR Nmax(" << Nmax << ") or Lmax(" << Lmax << ") or Mmax(" << Mmax << ") or Rh(" << Rh << ")\n";
	throw std::logic_error(oss.str());
      }
      if (CR < 0 || CR > 2) {
	std::ostringstream oss;
	oss << "Invalid CR \"" << CR << "\" (not implemented)\n";
	throw std::logic_error(oss.str());
      }

    }

    // the index order is actually (l,m,n)
    arma::cube PhinlmTable::get_Nnlm() {
      if (!PhinlmTable::Nnlm.size()) {
	PhinlmTable::compute_Nnlm();
      }
      return PhinlmTable::Nnlm;

    }

    void PhinlmTable::compute_Nnlm() {
      PhinlmTable::Nnlm.resize(Lmax+1,Mmax+1,Nmax+1);

      if (CR == 0) { // does NOT include spherical harmonic normalisation factor Jlm
	for (int l = 0; l <= Lmax; l++) {
	  double ld(l);
	  for (int m = 0; m <= l && m <= Mmax; m++) {
	    double md(abs(m));

	    PhinlmTable::Nnlm(l,m,0) = (1+ld+md)*(2+ld+md)*gsl_sf_beta(ld+1.5,md+1)/(8*M_PI*pow(Rh,4));

	    for (int n = 1; n <= Nmax; n++) {
	      double nd(n);
	      PhinlmTable::Nnlm(l,m,n) = PhinlmTable::Nnlm(l,m,n-1)*(nd*(md + nd)*(1 + 2*ld + 2*nd)*(-1 + ld + md + 2*nd)*(ld + md + 2*nd)*(1 + ld + md + 2*nd)*(2 + ld + md + 2*nd)*(1 + 2*ld + 2*md + 2*nd))/(4*(-1 + 2*ld + 2*md + 4*nd)*pow(1 + 2*ld + 2*md + 4*nd,2)*(3 + 2*ld + 2*md + 4*nd));
	    }
	    
	  }
	}
      }

      if (CR == 1 || CR == 2) {
	PhinlmTable::Nnlm.ones();
	PhinlmTable::Nnlm *= 1/pow(Rh,4);

	// for (int l = 0; l <= Lmax; l++) {
	  // for (int m = 0; m <= l && m <= Mmax; m++) {
	    // PhinlmTable::Nnlm.tube(l,m) *= 4.0*M_PI*pow(Rh,5)*minusonepow(abs(m))*gsl_sf_fact(l - abs(m))/gsl_sf_fact(l + abs(m));
	  // }
	// }
	
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
      //std::cout << "computing Phinlm for mu=" << mu << "\n";

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
	
      } else if (CR == 1) { // "isochrone" (orthonormal)

	double zeta((pow(sinh(mu/2),2) - 1)/(pow(sinh(mu/2),2) + 1));

	for (int l = 0; l <= Lmax; l++) {
	  double ld(l);

	  for (int m = 0; m <= Mmax && m <= l; m++) {
	    double md(abs(m));

	    // double zeroth_order((pow(2,2 + 2*ld + md/2)*sqrt(M_PI))/sqrt((1 + ld)*(1 + ld + md)) * pow(sinh(mu),abs(m))/pow(1.0+cosh(mu),abs(m)+l+1));
	    double zeroth_order((pow(2,2.5 + 2*ld + md/2)*sqrt(M_PI)) * pow(sinh(mu),abs(m))/pow(1.0+cosh(mu),abs(m)+l+1));
	    
	    arma::vec prefactors(Nmax+1);

	    for (int n = 0; n <= Nmax; n++) {
	      double nd(n);
	      prefactors[n] = 1/sqrt((1 + ld + nd)*(1 + ld + md + nd));
	    }

	    arma::vec polys(jacobi_norm_n(Nmax,2*ld+1.0,md,zeta));

	    entry.Phinlm.tube(l,m) = 1/Rh*zeroth_order*(prefactors % polys);

	  }

	}

      } else if (CR == 2) { // same as 'CR==0' but orthonormal (Nnlm = 1 for all (n,l,m))

	double xi = (pow(sinh(mu),2) - 1)/(pow(sinh(mu),2) + 1);

	for (int l = 0; l <= Lmax; l++) {
	double ld(l);

	  for (int m = 0; m <= Mmax && m <= l; m++) {
	    double md(abs(m));

	    double zeroth_order(pow(2,-0.25)/sqrt(M_PI) * pow(2,0.5*(5 + l + abs(m)))*M_PI * pow(sinh(mu),abs(m))/pow(cosh(mu),abs(m)+l+1));

	    arma::vec prefactors(Nmax+1);
	    // prefactors[0] = 1.0;
	    
	    for (int n = 0; n <= Nmax; n++) {
	      double nd(n);
	      // prefactors[n] = prefactors[n-1] * (-0.125) * (nd*(-1 + ld + md + 2*nd)*(ld + md + 2*nd)*(1 + 2*ld + 2*md + 2*nd))/((-0.5 + ld + md + 2*nd)*(0.5 + ld + md + 2*nd));
	      prefactors[n] = minusonepow(n)*1/sqrt((1 + l + m + 2*n)*(2 + l + m + 2*n));
	    }

	    arma::vec polys(jacobi_norm_n(Nmax,ld+0.5,md,xi));

	    entry.Phinlm.tube(l,m) = 1/Rh*zeroth_order*(prefactors % polys);

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
      if (n > Nmax) {
        std::ostringstream oss;
	oss << "Error in get_Phinlm(" << n << "," << l << "," << m << "," << mu << "): n (=" << n << ") greater than Nmax (=" << Nmax << ")!\n";
      }
      if (l > Lmax) {
        std::ostringstream oss;
	oss << "Error in get_Phinlm(" << n << "," << l << "," << m << "," << mu << "): l (=" << l << ") greater than Lmax (=" << Lmax << ")!\n";
      }
      if ((abs(m) > Mmax) || (abs(m) > Lmax)) {
        std::ostringstream oss;
	oss << "Error in get_Phinlm(" << n << "," << l << "," << m << "," << mu << "): abs(m) (=" << abs(m) << ") greater than Mmax (=" << Mmax << ") or Lmax (=" << Lmax << ")!\n";
      }
      if(get_index(mu)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Phinlm(" << n << "," << l << "," << m << "," << mu << "): index " << get_index(mu) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(mu)].Phinlm(l,abs(m),n);
    }


    arma::vec PhinlmTable::get_Phinlm(int n, int l, int m, const arma::vec & mu) const {
      arma::vec phinlm_vec(mu.n_elem);
      //std::cout << "get_Phinlm(" << n << "," << l << "," << m << ") (mu vec)\n";
      for(size_t i=0;i<mu.n_elem;i++)
        phinlm_vec(i)=get_Phinlm(n,l,m,mu(i));
      return phinlm_vec;
    }

    // the index order is actually (l,m,n)
    arma::cube PhinlmTable::get_Phinlm(double mu) const {
      if(get_index(mu)>stor.size()) {
        std::ostringstream oss;
        oss << "Error in get_Phinlm(" << mu << "): index " << get_index(mu) << " greater than array size " << stor.size() << "!\n";
        throw std::logic_error(oss.str());
      }
      return stor[get_index(mu)].Phinlm;
    }

    int PhinlmTable::get_Nmax() {
      return Nmax;
    }

    int PhinlmTable::get_Lmax() {
      return Lmax;
    }

    int PhinlmTable::get_Mmax() {
      return Mmax;
    }

    double PhinlmTable::get_Rh() {
      return Rh;
    }

  }

}

