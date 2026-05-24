/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#include "TwoDBasis.h"
#include "basis.h"
#include "quadrature.h"
#include "chebyshev.h"
#include "../general/spherical_harmonics.h"
#include "../general/gaunt.h"
#include "utils.h"
#include "../general/scf_helpers.h"
#include <cassert>
#include <cfloat>
#include <helfem.h>
#include "../general/constants.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace helfem {
  namespace atomic {
    namespace basis {
      TwoDBasis::TwoDBasis() {
      }

      TwoDBasis::TwoDBasis(int Z_, modelpotential::nuclear_model_t model_, double Rrms_, const std::shared_ptr<const polynomial_basis::PolynomialBasis> & poly, bool zeroder_, int n_quad, const arma::vec & bval, int taylor_order, const arma::ivec & lval_, const arma::ivec & mval_, int Zl_, int Zr_, double Rhalf_) {
        // Nuclear charge
        Z=Z_;
        Zl=Zl_;
        Zr=Zr_;
        Rhalf=Rhalf_;
        model=model_;
        Rrms=Rrms_;

        // Construct radial basis
        bool zero_func_left=true;
        bool zero_deriv_left=false;
        bool zero_func_right=true;
        zeroder=zeroder_;
        polynomial_basis::FiniteElementBasis fem(poly, bval, zero_func_left, zero_deriv_left, zero_func_right, zeroder);
        radial=RadialBasis(fem, n_quad, taylor_order);

        // Construct angular basis
        lval=lval_;
        mval=mval_;

	Nmax = -1;
	rsfrac = 1.0;
      }

      TwoDBasis::~TwoDBasis() {
      }

      int TwoDBasis::get_nuclear_model() const {
        return model;
      }

      double TwoDBasis::get_nuclear_size() const {
        return Rrms;
      }

      int TwoDBasis::get_Z() const {
        return Z;
      }

      int TwoDBasis::get_Zl() const {
        return Zl;
      }

      int TwoDBasis::get_Zr() const {
        return Zr;
      }

      double TwoDBasis::get_Rhalf() const {
        return Rhalf;
      }

      arma::ivec TwoDBasis::get_lval() const {
        return lval;
      }

      arma::ivec TwoDBasis::get_mval() const {
        return mval;
      }

      int TwoDBasis::get_nquad() const {
        return radial.get_nquad();
      }

      arma::vec TwoDBasis::get_bval() const {
        return radial.get_bval();
      }

      int TwoDBasis::get_poly_id() const {
        return radial.get_poly_id();
      }

      int TwoDBasis::get_poly_nnodes() const {
        return radial.get_poly_nnodes();
      }

      int TwoDBasis::get_zeroder() const {
        return zeroder;
      }

      size_t TwoDBasis::Ndummy() const {
        return lval.n_elem*radial.Nbf();
      }

      size_t TwoDBasis::Nbf() const {
        return Ndummy();
      }

      size_t TwoDBasis::Nrad() const {
        return radial.Nbf();
      }

      size_t TwoDBasis::Nang() const {
        return lval.n_elem;
      }

      double TwoDBasis::get_small_r_taylor_cutoff() const {
        return radial.get_small_r_taylor_cutoff();
      }

      int TwoDBasis::get_taylor_order() const {
        return radial.get_taylor_order();
      }

      double TwoDBasis::get_taylor_diff() const {
        return radial.get_taylor_diff();
      }

      arma::uvec TwoDBasis::pure_indices() const {
        return arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
      }

      arma::ivec TwoDBasis::get_l() const {
        return lval;
      }

      arma::ivec TwoDBasis::get_m() const {
        return mval;
      }

      arma::uvec TwoDBasis::m_indices(int m) const {
        // Count how many functions
        size_t nm=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          if(mval(i)==m) {
            nm += radial.Nbf();
          }
        }

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=radial.Nbf();
          if(mval(i)==m) {
            idx.subvec(ioff,ioff+nsh-1)=arma::linspace<arma::uvec>(ibf,ibf+nsh-1,nsh);
            ioff+=nsh;
          }
          ibf+=nsh;
        }

        return idx;
      }

      arma::uvec TwoDBasis::lm_indices(int l, int m) const {
        // Count how many functions
        size_t nm=radial.Nbf();

        // Collect functions
        arma::uvec idx(nm);
        size_t ioff=0;
        size_t ibf=0;
        for(size_t i=0;i<mval.n_elem;i++) {
          // Number of functions on shell is
          size_t nsh=radial.Nbf();
          if(mval(i)==m && lval(i)==l) {
            idx.subvec(ioff,ioff+nsh-1)=arma::linspace<arma::uvec>(ibf,ibf+nsh-1,nsh);
            ioff+=nsh;
          }
          ibf+=nsh;
        }

        return idx;
      }

      std::vector<arma::uvec> TwoDBasis::get_sym_idx(int symm) const {
        std::vector<arma::uvec> idx;
        if(symm==0) {
          idx.resize(1);
          idx[0]=arma::linspace<arma::uvec>(0,Nbf()-1,Nbf());
        } else if(symm==1) {
          // Find unique m values
          arma::uvec muni(arma::find_unique(mval));
          arma::ivec mv(mval(muni));

          idx.resize(mv.n_elem);
          for(size_t i=0;i<mv.n_elem;i++)
            idx[i]=m_indices(mv(i));
        } else if(symm==2) {
          idx.resize(mval.n_elem);
          for(size_t i=0;i<mval.n_elem;i++) {
            idx[i]=lm_indices(lval(i),mval(i));
          }
        } else
          throw std::logic_error("Unknown symmetry\n");

        return idx;
      }

      arma::mat TwoDBasis::Shalf(bool chol, int sym) const {
        // Form overlap matrix
        arma::mat S(overlap());

        // Get the basis function norms
        arma::vec bfnormlz(arma::pow(arma::diagvec(S),-0.5));
        arma::vec bfinvnormlz(arma::pow(arma::diagvec(S),0.5));
	printf("Smallest normalization constant % e, largest % e\n",arma::min(bfnormlz),arma::max(bfnormlz));
        // Go to normalized basis
        S=arma::diagmat(bfnormlz)*S*arma::diagmat(bfnormlz);

        if(chol && sym==0) {
          // Half-inverse is
          return arma::diagmat(bfinvnormlz) * arma::chol(S);

        } else {
          arma::vec Sval;
          arma::mat Svec;
          if(sym) {
            // Symmetries
            std::vector<arma::uvec> midx(get_sym_idx(sym));
            scf::eig_sym_sub(Sval,Svec,S,midx);
          } else {
            if(!arma::eig_sym(Sval,Svec,S)) {
              S.save("S.dat",arma::raw_ascii);
              throw std::logic_error("Diagonalization of overlap matrix failed\n");
            }
          }
          printf("Smallest eigenvalue of overlap matrix is % e, condition number %e\n",Sval(0),Sval(Sval.n_elem-1)/Sval(0));

          arma::mat Shalf(Svec*arma::diagmat(arma::pow(Sval,0.5))*arma::trans(Svec));
          Shalf=arma::diagmat(bfinvnormlz)*Shalf;

          return Shalf;
        }
      }

      arma::mat TwoDBasis::Sinvh(bool chol, int sym) const {
        // Form overlap matrix
        arma::mat S(overlap());

        // Half-inverse is
        if(sym==0) {
          return scf::form_Sinvh(S,chol);
        } else {
          // Get basis function indices
          std::vector<arma::uvec> midx(get_sym_idx(sym));
          // Construct Sinvh in each subblock
          arma::mat Sinvh(Nbf(),Nbf(),arma::fill::zeros);
          size_t ioff=0;
          for(size_t i=0;i<midx.size();i++) {
            if(!midx[i].n_elem)
              continue;

            // Column indices
            arma::uvec cidx(arma::linspace<arma::uvec>(ioff,ioff+midx[i].n_elem-1,midx[i].n_elem));
            Sinvh(midx[i],cidx)=scf::form_Sinvh(S(midx[i],midx[i]),chol);
            // Increment offset
            ioff += midx[i].n_elem;
          }
          return Sinvh;
        }
      }

      void TwoDBasis::set_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Mrad) const {
        M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1)=Mrad;
      }

      void TwoDBasis::add_sub(arma::mat & M, size_t iang, size_t jang, const arma::mat & Mrad) const {
        M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1)+=Mrad;
      }

      arma::mat TwoDBasis::get_sub(const arma::mat & M, size_t iang, size_t jang) const {
        return M.submat(iang*radial.Nbf(),jang*radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*radial.Nbf()-1);
      }

      arma::mat TwoDBasis::radial_integral(int Rexp) const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
	  Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(Rexp,iel);
        }

        // Full overlap matrix
        arma::mat O(Ndummy(),Ndummy());
        O.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++)
          set_sub(O,iang,iang,Orad);

        return remove_boundaries(O);
      }

      arma::mat TwoDBasis::overlap() const {
        return radial_integral(0);
      }

      arma::mat TwoDBasis::overlap(const TwoDBasis & rh) const {
        // Full overlap matrix
        arma::mat S(Ndummy(),rh.Ndummy());
        S.zeros();
        // Form radial overlap
        arma::mat Srad(radial.overlap(rh.radial));

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++)
          for(size_t jang=0;jang<rh.lval.n_elem;jang++)
            if(lval(iang) == rh.lval(jang) && mval(iang) == rh.mval(jang))
              S.submat(iang*radial.Nbf(),jang*rh.radial.Nbf(),(iang+1)*radial.Nbf()-1,(jang+1)*rh.radial.Nbf()-1)=Srad;

        return S;
      }

      arma::mat TwoDBasis::kinetic() const {
        // Build radial kinetic energy matrix
        size_t Nrad(radial.Nbf());
        arma::mat Trad(Nrad,Nrad);
        Trad.zeros();
        arma::mat Trad_l(Nrad,Nrad);
        Trad_l.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Trad.submat(ifirst,ifirst,ilast,ilast)+=radial.kinetic(iel);
          Trad_l.submat(ifirst,ifirst,ilast,ilast)+=radial.kinetic_l(iel);
        }

        // Full kinetic energy matrix
        arma::mat T(Ndummy(),Ndummy());
        T.zeros();
        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          set_sub(T,iang,iang,Trad);
          if(lval(iang)>0) {
            // We also get the l(l+1) term
            add_sub(T,iang,iang,lval(iang)*(lval(iang)+1)*Trad_l);
          }
        }

        return remove_boundaries(T);
      }

      arma::mat TwoDBasis::nuclear() const {
        if(model != modelpotential::POINT_NUCLEUS) {
          modelpotential::ModelPotential *pot=modelpotential::get_nuclear_model(model,Z,Rrms);
          arma::mat Vnuc(model_potential(pot));
          delete pot;
          return Vnuc;
        } else {
          // Full nuclear attraction matrix
          arma::mat V(Ndummy(),Ndummy());
          V.zeros();

          if(Z!=0.0) {
            size_t Nrad(radial.Nbf());
            arma::mat Vrad(Nrad,Nrad);
            Vrad.zeros();
            // Loop over elements
            for(size_t iel=0;iel<radial.Nel();iel++) {
              // Where are we in the matrix?
              size_t ifirst, ilast;
              radial.get_idx(iel,ifirst,ilast);
              Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(-1,iel);
            }
            // Fill elements
            for(size_t iang=0;iang<lval.n_elem;iang++)
              set_sub(V,iang,iang,-Z*Vrad);
          }

          if(Zl != 0.0 || Zr != 0.0) {
            // Auxiliary matrices
            size_t Nrad(radial.Nbf());
            int Lmax(2*arma::max(lval));
            std::vector<arma::mat> Vaux(Lmax+1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int L=0;L<=Lmax;L++) {
              Vaux[L].zeros(Nrad,Nrad);
              for(size_t iel=0;iel<radial.Nel();iel++) {
                // Where are we in the matrix?
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);
                Vaux[L].submat(ifirst,ifirst,ilast,ilast)+=radial.nuclear_offcenter(iel,Rhalf,L);
              }
            }

            int gmax(std::max(arma::max(lval),arma::max(mval)));
            gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

            /// Loop over basis set
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
            for(size_t iang=0;iang<lval.n_elem;iang++) {
              for(size_t jang=0;jang<lval.n_elem;jang++) {
                int li(lval(iang));
                int mi(mval(iang));

                int lj(lval(jang));
                int mj(mval(jang));

                // Zero contribution
                if(mi!=mj)
                  continue;

                // Loop over L
                for(int L=std::abs(li-lj);L<=li+lj;L++) {
                  double cpl(gaunt.coeff(li,mi,L,0,lj,mj));
                  if(cpl==0.0)
                    continue;

                  add_sub(V,iang,jang,cpl*(std::pow(-1.0,L)*Zl + Zr)*Vaux[L]);
                }
              }
            }
          }

          return remove_boundaries(V);
        }
      }

      arma::mat TwoDBasis::model_potential(const modelpotential::ModelPotential * pot) const {
        // Full nuclear attraction matrix
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

	size_t Nrad(radial.Nbf());
	arma::mat Vrad(Nrad,Nrad);
	Vrad.zeros();
	// Loop over elements
	for(size_t iel=0;iel<radial.Nel();iel++) {
	  // Where are we in the matrix?
	  size_t ifirst, ilast;
	  radial.get_idx(iel,ifirst,ilast);
	  Vrad.submat(ifirst,ifirst,ilast,ilast)+=radial.model_potential(pot,iel);
	}
	// Fill elements
	for(size_t iang=0;iang<lval.n_elem;iang++)
	  set_sub(V,iang,iang,Vrad);

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::confinement(const int N, double r_0, const int iconf, const double V, const double shift_pot) const {
        // Full matrix
        arma::mat O(Ndummy(),Ndummy());
        O.zeros();

	if(iconf==0)
	  return remove_boundaries(O);

	// Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

	// Loop over elements
	for(size_t iel=0;iel<radial.Nel();iel++) {
	  // Where are we in the matrix?
	  size_t ifirst, ilast;
	  radial.get_idx(iel,ifirst,ilast);
	  Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.confinement_potential(iel,N,r_0,iconf,V,shift_pot);
	}

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++)
          set_sub(O,iang,iang,Orad);

        return remove_boundaries(O);
      }

      arma::mat TwoDBasis::dipole_z() const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(1,iel);
        }

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,1,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.cosine_coupling(lj,mj,li,mi));
            if(cpl!=0.0)
              set_sub(V,iang,jang,Orad*cpl);
          }
        }

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::quadrupole_zz() const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat Orad(Nrad,Nrad);
        Orad.zeros();

        // Full electric couplings
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          Orad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(2,iel);
        }

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.coeff(lj,mj,2,0,li,mi));
            if(cpl!=0.0) {
              const double c0(2.0/5.0*sqrt(5.0*M_PI));
              cpl*=c0;
              set_sub(V,iang,jang,Orad*cpl);
            }
          }
        }

        return remove_boundaries(V);
      }

      arma::mat TwoDBasis::Bz_field(double B) const {
        // Build radial elements
        size_t Nrad(radial.Nbf());
        arma::mat O0rad(Nrad,Nrad);
        O0rad.zeros();
        arma::mat O2rad(Nrad,Nrad);
        O2rad.zeros();

        // Loop over elements
        for(size_t iel=0;iel<radial.Nel();iel++) {
          // Where are we in the matrix?
          size_t ifirst, ilast;
          radial.get_idx(iel,ifirst,ilast);
          O0rad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(0,iel);
          O2rad.submat(ifirst,ifirst,ilast,ilast)+=radial.radial_integral(2,iel);
        }

        // Full coupling
        arma::mat V(Ndummy(),Ndummy());
        V.zeros();

        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,4,gmax);

        // Fill elements
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          int li(lval(iang));
          int mi(mval(iang));

          for(size_t jang=0;jang<lval.n_elem;jang++) {
            int lj(lval(jang));
            int mj(mval(jang));

            // Calculate coupling
            double cpl(gaunt.sine2_coupling(lj,mj,li,mi));
            if(cpl!=0.0) {
              set_sub(V,iang,jang,B*B/8.0*O2rad*cpl);
            }

            if(li==lj && mi==mj) {
              add_sub(V,iang,jang,-0.5*B*mj*O0rad);
            }
          }
        }

        return remove_boundaries(V);
      }

      size_t TwoDBasis::mem_1el() const {
        return Nbf()*Nbf()*sizeof(double);
      }

      size_t TwoDBasis::mem_1el_aux() const {
        size_t Nel(radial.Nel());
        size_t Nprim(radial.max_Nprim());
        size_t N_L(2*arma::max(lval)+1);

        return 2*N_L*Nel*Nprim*Nprim*sizeof(double);
      }

      size_t TwoDBasis::mem_2el_aux() const {
        // Auxiliary integrals required up to
        size_t N_L(2*arma::max(lval)+1);
        // Number of elements
        size_t Nel(radial.Nel());
        // Number of primitive functions per element
        size_t Nprim(radial.max_Nprim());

        // Memory use is thus
        //return 2*N_L*Nel*Nel*Nprim*Nprim*Nprim*Nprim*sizeof(double);
        // No off-diagonal storage
        return 2*N_L*Nel*Nprim*Nprim*Nprim*Nprim*sizeof(double);
      }


      void TwoDBasis::compute_tei(bool exchange) {
        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        disjoint_L.resize(Nel*N_L);
        disjoint_m1L.resize(Nel*N_L);
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_L[L*Nel+iel]=radial.radial_integral(L,iel);
            disjoint_m1L[L*Nel+iel]=radial.radial_integral(-L-1,iel);
          }

        // Form two-electron integrals
        prim_tei.resize(Nel*Nel*N_L);

	//size_t Nmax = TwoDBasis::get_Nmax();
	//std::cout << "Nmax=" << Nmax << "\n";
	//double rmax = radial.fem.element_end(Nel-1);
	//std::cout << "rmax=" << rmax << "\n";
	//double rs = rmax/5;
	//std::cout << "rs=" << rs << "\n";
	//size_t Nprim(radial.max_Nprim());
	//std::cout << "max Nprim=" << Nprim << "\n";
	//size_t Nrad(radial.Nbf());
	//std::cout << "Nrad=" << Nrad << "\n";
        //prim_tei_cr.resize(Nel*N_L);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++) {
          for(size_t iel=0;iel<Nel;iel++) {
            // In-element integral
            prim_tei[Nel*Nel*L + iel*Nel + iel]=radial.twoe_integral(L,iel);
	    //std::cout << "prim_tei[" << Nel*Nel*L + iel*Nel + iel << "] size is " << prim_tei[Nel*Nel*L + iel*Nel + iel].size() << "\n";

            //prim_tei_cr[Nel*L + iel] = radial.twoe_integral_cr(Nmax,L,iel,rs);
	    //std::cout << "prim_tei_cr[" << Nel*L + iel << "] size is " << prim_tei_cr[Nel*L + iel].size() << "\n";

            /*
              for(size_t jel=0;jel<Nel;jel++) {
	      // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
	      const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
	      const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

	      // Store integrals
	      prim_tei[Nel*Nel*L + iel*Nel + jel]=utils::product_tei(iint,jint);
              }
              }
            */
          }
        }

        /*
          The exchange matrix is given by
          K(jk) = (ij|kl) P(il)
          i.e. the complex conjugation hits i and l as
          in the density matrix.

          To get this in the proper order, we permute the integrals
          K(jk) = (jk;il) P(il)
          so we don't have to reform the permutations in the exchange routine.
        */
        if(exchange) {
          prim_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
          for(size_t L=0;L<N_L;L++)
            for(size_t iel=0;iel<Nel;iel++) {
              // Diagonal integrals
              size_t Ni(radial.Nprim(iel));
              prim_ktei[Nel*Nel*L + iel*Nel + iel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + iel],Ni,Ni,Ni,Ni);

              // Off-diagonal integrals (not used since faster to contract
              // the integrals in factorized form)
              /*
                for(size_t jel=0;jel<iel;jel++) {
                size_t Nj(radial.Nprim(jel));
                prim_ktei[Nel*Nel*L + iel*Nel + jel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + jel],Ni,Ni,Nj,Nj);
                }
                for(size_t jel=iel+1;jel<Nel;jel++) {
                size_t Nj(radial.Nprim(jel));
                prim_ktei[Nel*Nel*L + iel*Nel + jel]=utils::exchange_tei(prim_tei[Nel*Nel*L + iel*Nel + jel],Ni,Ni,Nj,Nj);
                }
              */
            }
        }
      }

      void TwoDBasis::compute_tei_cr(int CR) {
        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

	size_t Nmax = TwoDBasis::get_Nmax();
	double rmax = radial.fem.element_end(Nel-1);
	//double rs = rmax*TwoDBasis::get_rsfrac();  // should probably use Bohr radius instead of rmax
	double rs = BOHRINANGSTROM*TwoDBasis::get_rsfrac();

	std::cout << "CR: Nmax=" << Nmax << "    rs=" << rs << "\n";

	size_t Nprim(radial.max_Nprim());
	//std::cout << "CR: max Nprim=" << Nprim << "\n";

	size_t Nrad(radial.Nbf());
	//std::cout << "Nrad=" << Nrad << "\n";

        prim_tei_cr.resize(Nel*N_L);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif

	for(size_t L=0;L<N_L;L++) {
          for(size_t iel=0;iel<Nel;iel++) {
            // In-element integral
            //prim_tei[Nel*Nel*L + iel*Nel + iel]=radial.twoe_integral(L,iel);
	    //std::cout << "prim_tei[" << Nel*Nel*L + iel*Nel + iel << "] size is " << prim_tei[Nel*Nel*L + iel*Nel + iel].size() << "\n";

            prim_tei_cr[Nel*L + iel] = radial.twoe_integral_cr(CR,Nmax,L,iel,rs);
	    //std::cout << "prim_tei_cr[" << Nel*L + iel << "] size is " << prim_tei_cr[Nel*L + iel].size() << "\n";

          }
        }

      }

      void TwoDBasis::compute_yukawa(double lambda_) {
        lambda=lambda_;
        yukawa=true;

        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

        // Compute disjoint integrals
        disjoint_iL.resize(Nel*N_L);
        disjoint_kL.resize(Nel*N_L);
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            disjoint_iL[L*Nel+iel]=radial.bessel_il_integral(L,lambda,iel);
            disjoint_kL[L*Nel+iel]=radial.bessel_kl_integral(L,lambda,iel);
          }

        /*
          The exchange matrix is given by
          K(jk) = (ij|kl) P(il)
          i.e. the complex conjugation hits i and l as
          in the density matrix.

          To get this in the proper order, we permute the integrals
          K(jk) = (jk;il) P(il)
          so we don't have to reform the permutations in the exchange routine.
        */
        rs_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            // Diagonal integrals
            size_t Ni(radial.Nprim(iel));
            rs_ktei[Nel*Nel*L + iel*Nel + iel]=utils::exchange_tei(radial.yukawa_integral(L,lambda,iel),Ni,Ni,Ni,Ni);
          }
      }

      void TwoDBasis::compute_erfc(double mu) {
        lambda=mu;
        yukawa=false;

        // Number of distinct L values is
        size_t N_L(2*arma::max(lval)+1);
        size_t Nel(radial.Nel());

        // No disjoint integrals
        disjoint_iL.clear();
        disjoint_kL.clear();

        /*
          The exchange matrix is given by
          K(jk) = (ij|kl) P(il)
          i.e. the complex conjugation hits i and l as
          in the density matrix.

          To get this in the proper order, we permute the integrals
          K(jk) = (jk;il) P(il)
          so we don't have to reform the permutations in the exchange routine.
        */
        rs_ktei.resize(Nel*Nel*N_L);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for(size_t L=0;L<N_L;L++)
          for(size_t iel=0;iel<Nel;iel++) {
            for(size_t kel=0;kel<Nel;kel++) {
              // Diagonal integrals
              size_t Ni(radial.Nprim(iel));
              size_t Nk(radial.Nprim(kel));
              rs_ktei[Nel*Nel*L + iel*Nel + kel]=utils::exchange_tei(radial.erfc_integral(L,lambda,iel,kel),Ni,Ni,Nk,Nk);
            }
          }
      }

      arma::mat TwoDBasis::coulomb(const arma::mat & P0) const {
        if(!prim_tei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());
	// Max radial order of Coulomb resolution
	//int Nmax = TwoDBasis::get_Nmax();
        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // maximal M value
        int Mmax=arma::max(mval)-arma::min(mval);

        // Radial helper matrices
        std::vector< std::vector<arma::mat> > Paux(2*arma::max(lval)+1);
        for(int L=0;L<(int) Paux.size();L++) {
          Paux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Paux[L][M+Mmax].zeros(Nrad,Nrad);
          }
        }

	

        // std::vector< std::vector<arma::vec> > Paux_cr(2*arma::max(lval)+1);
        // for(int L=0;L<(int) Paux_cr.size();L++) {
        //   Paux_cr[L].resize(2*Mmax+1);
        //   for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
        //     Paux_cr[L][M+Mmax].zeros(Nmax+1);
        //   }
        // }


        // Form radial helpers: contract ket
        for(size_t kang=0;kang<lval.n_elem;kang++) {
          for(size_t lang=0;lang<lval.n_elem;lang++) {
            // l and m values
            int lk(lval(kang));
            int mk(mval(kang));
            int ll(lval(lang));
            int ml(mval(lang));
            // RH m value
            int M(mk-ml);
            // M values match. Loop over possible couplings
            int Lmin=std::max(std::abs(lk-ll),abs(M));
            int Lmax=lk+ll;
            for(int L=Lmin;L<=Lmax;L++) {
              // Calculate coupling coefficient
              double cpl(gaunt.coeff(lk,mk,L,M,ll,ml));
              // Increment
              Paux[L][M+Mmax]+=cpl*P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1);
            }
          }
        }

	// std::cout << "P rows: " << P.n_rows << "    ";
	// std::cout << "P cols: " << P.n_cols << "    ";
	// std::cout << "\n";
	// 
        // for(size_t kang=0;kang<lval.n_elem;kang++) {
        //   for(size_t lang=0;lang<lval.n_elem;lang++) {
        //     // l and m values
        //     int lk(lval(kang));
        //     int mk(mval(kang));
        //     int ll(lval(lang));
        //     int ml(mval(lang));
	//     std::cout << "kang=" << kang << " lk=" << lk << " mk=" << mk << "\n";
	//     std::cout << "lang=" << lang << " ll=" << ll << " ml=" << ml << "\n";
        //     // RH m value
        //     int M(mk-ml);
        //     // M values match. Loop over possible couplings
        //     int Lmin=std::max(std::abs(lk-ll),abs(M));
        //     int Lmax=lk+ll;
        //     for(int L=Lmin;L<=Lmax;L++) {
        //       // Calculate coupling coefficient
        //       double cpl(gaunt.coeff(lk,mk,L,M,ll,ml));
        //       // Increment
	// 
	//       arma::mat Psubkl(P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1));
	//       std::cout << "Psubkl rows: " << Psubkl.n_rows << "    ";
	//       std::cout << "Psubkl cols: " << Psubkl.n_cols << "    ";
	//       std::cout << "\n";
	// 
	//       
        //       // Paux_cr[L][M+Mmax]+=cpl*P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1);
	// 
	//       for(size_t iel=0;iel<Nel;iel++) {
	// 
	// 	size_t ifirst, ilast;
	// 	radial.get_idx(iel,ifirst,ilast);
	// 	size_t Ni(ilast-ifirst+1);
	// 
	// 	arma::mat Psubkli(Psubkl.submat(ifirst,ifirst,ilast,ilast));
	// 	Psubkli.reshape(Ni*Ni,1);
	// 	std::cout << "Psubkli rows: " << Psubkli.n_rows << "    ";
	// 	std::cout << "Psubkli cols: " << Psubkli.n_cols << "    ";
	// 	std::cout << "\n";
	// 
	// 	arma::mat prim_tei_cr_sub(prim_tei_cr[Nel*L + iel]);
	// 	std::cout << "prim_tei_cr_sub rows: " << prim_tei_cr_sub.n_rows << "    ";
	// 	std::cout << "prim_tei_cr_sub cols: " << prim_tei_cr_sub.n_cols << "    ";
	// 	std::cout << "\n";
	// 
	// 	Paux_cr[L][M+Mmax] += cpl*(arma::trans(prim_tei_cr_sub)*Psubkli);
	// 	std::cout << "Paux_cr[" << L << "][" << M+Mmax << "] rows: " << Paux_cr[L][M+Mmax].n_rows << "    ";
	// 	std::cout << "\n";
	// 
	//       }
        //     }
        //   }
        // }

        // Helper matrices
        std::vector< std::vector<arma::mat> > Jaux(2*arma::max(lval)+1);
        for(int L=0;L<(int) Jaux.size();L++) {
          Jaux[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Jaux[L][M+Mmax].zeros(Nrad,Nrad);
          }
        }

        // Contract integrals
        for(int L=0;L<(int) Paux.size();L++) {
          const double Lfac=4.0*M_PI/(2*L+1);

          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            for(size_t jel=0;jel<Nel;jel++) {
              size_t jfirst, jlast;
              radial.get_idx(jel,jfirst,jlast);
              size_t Nj(jlast-jfirst+1);

              // Get density submatrices
              arma::mat Psub(Paux[L][M+Mmax].submat(jfirst,jfirst,jlast,jlast));

              // Contract integrals
              double jsmall = Lfac*arma::trace(disjoint_L[L*Nel+jel]*Psub);
              double jbig = Lfac*arma::trace(disjoint_m1L[L*Nel+jel]*Psub);

              // Increment J: jel>iel
              for(size_t iel=0;iel<jel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                const arma::mat & iint=disjoint_L[L*Nel+iel];
                Jaux[L][M+Mmax].submat(ifirst,ifirst,ilast,ilast)+=jbig*iint;
              }

              // Increment J: jel<iel
              for(size_t iel=jel+1;iel<Nel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                const arma::mat & iint=disjoint_m1L[L*Nel+iel];
                Jaux[L][M+Mmax].submat(ifirst,ifirst,ilast,ilast)+=jsmall*iint;
              }

              // In-element contribution
              {
                size_t iel=jel;
                size_t ifirst=jfirst;
                size_t ilast=jlast;
                size_t Ni=Nj;

                // Contract integrals
                Psub.reshape(Nj*Nj,1);

                const size_t idx(Nel*Nel*L + iel*Nel + jel);
                arma::mat Jsub(Lfac*(prim_tei[idx]*Psub));
                Jsub.reshape(Ni,Ni);

                Jaux[L][M+Mmax].submat(ifirst,ifirst,ilast,ilast)+=Jsub;
              }

            }
          }
        }

        // Full Coulomb matrix
        arma::mat J(Ndummy(),Ndummy());
        J.zeros();
        // arma::mat J_cr(Ndummy(),Ndummy());
        // J_cr.zeros();
        for(size_t iang=0;iang<lval.n_elem;iang++) {
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            // l and m values
            int li(lval(iang));
            int mi(mval(iang));
            int lj(lval(jang));
            int mj(mval(jang));
            // LH m value
            int M(mj-mi);

            int Lmin=std::max(std::abs(lj-li),abs(M));
            int Lmax=lj+li;
            for(int L=Lmin;L<=Lmax;L++) {
              // Coupling
              double cpl(gaunt.coeff(lj,mj,L,M,li,mi));
              if(cpl!=0.0) {
		J.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1)+=cpl*Jaux[L][M+Mmax];

// 		//arma::mat Jauxsub(Jaux[L][M+Mmax]);
// 		std::cout << "Jaux[" << L << "][" << M+Mmax << "] rows: " << Jaux[L][M+Mmax].n_rows << "    ";
// 		std::cout << "Jaux[" << L << "][" << M+Mmax << "] cols: " << Jaux[L][M+Mmax].n_cols << "    ";
// 		std::cout << "\n";
// 
// 		
// 		std::cout << "Paux_cr[" << L << "][" << M+Mmax << "] rows: " << Paux_cr[L][M+Mmax].n_rows << "    ";
// 		std::cout << "\n";
// 
// 	      for(size_t iel=0;iel<Nel;iel++) {
// 
// 		size_t ifirst, ilast;
// 		radial.get_idx(iel,ifirst,ilast);
// 		size_t Ni(ilast-ifirst+1);
// 
// 
// 
// 		arma::mat prod(cpl*(prim_tei_cr[Nel*L + iel] * Paux_cr[L][M+Mmax]));
// 		prod.reshape(Ni,Ni);
// 		//std::cout << "prod rows: " << prod.n_rows << "    ";
// 		//std::cout << "prod cols: " << prod.n_cols << "    ";
// 		//std::cout << "\n";
// 
// 		J_cr.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1).submat(ifirst,ifirst,ilast,ilast) += prod;
// 
// 	      }
              }
            }
          }
        }

	// std::cout << "J rows: " << J.n_rows << "    ";
	// std::cout << "J cols: " << J.n_cols << "    ";
	// std::cout << "\n";
	// std::cout << "J:\n" << J << "    ";

        return remove_boundaries(J);
      }

      arma::mat TwoDBasis::coulomb_cr(const arma::mat & P0) const {
        if(!prim_tei_cr.size())
          throw std::logic_error("Primitive CR teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial functions
        size_t Nrad(radial.Nbf());
	// Max radial order of Coulomb resolution
	int Nmax = TwoDBasis::get_Nmax();
        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // maximal M value
        int Mmax=arma::max(mval)-arma::min(mval);

        std::vector< std::vector<arma::vec> > Paux_cr(2*arma::max(lval)+1);
        for(int L=0;L<(int) Paux_cr.size();L++) {
          Paux_cr[L].resize(2*Mmax+1);
          for(int M=-std::min(L,Mmax);M<=std::min(L,Mmax);M++) {
            Paux_cr[L][M+Mmax].zeros(Nmax+1);
          }
        }

	//std::cout << "P rows: " << P.n_rows << "    ";
	//std::cout << "P cols: " << P.n_cols << "    ";
	//std::cout << "\n";

        for(size_t kang=0;kang<lval.n_elem;kang++) {
          for(size_t lang=0;lang<lval.n_elem;lang++) {
            // l and m values
            int lk(lval(kang));
            int mk(mval(kang));
            int ll(lval(lang));
            int ml(mval(lang));
	    //std::cout << "kang=" << kang << " lk=" << lk << " mk=" << mk << "\n";
	    //std::cout << "lang=" << lang << " ll=" << ll << " ml=" << ml << "\n";
            // RH m value
            int M(mk-ml);
            // M values match. Loop over possible couplings
            int Lmin=std::max(std::abs(lk-ll),abs(M));
            int Lmax=lk+ll;
            for(int L=Lmin;L<=Lmax;L++) {
              // Calculate coupling coefficient
              double cpl(gaunt.coeff(lk,mk,L,M,ll,ml));

	      if (cpl==0.0)
		continue;

              // Increment

	      arma::mat Psubkl(P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1));

	      //std::cout << "Psubkl rows: " << Psubkl.n_rows << "    ";
	      //std::cout << "Psubkl cols: " << Psubkl.n_cols << "    ";
	      //std::cout << "\n";

	      
              // Paux_cr[L][M+Mmax]+=cpl*P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1);

	      for(size_t iel=0;iel<Nel;iel++) {

		size_t ifirst, ilast;
		radial.get_idx(iel,ifirst,ilast);
		size_t Ni(ilast-ifirst+1);

		arma::mat Psubkli(Psubkl.submat(ifirst,ifirst,ilast,ilast));
		//arma::mat Psubkli(P.submat(kang*Nrad,lang*Nrad,(kang+1)*Nrad-1,(lang+1)*Nrad-1).submat(ifirst,ifirst,ilast,ilast));
		Psubkli.reshape(Ni*Ni,1);
		//std::cout << "Psubkli rows: " << Psubkli.n_rows << "    ";
		//std::cout << "Psubkli cols: " << Psubkli.n_cols << "    ";
		//std::cout << "\n";

		arma::mat prim_tei_cr_sub(prim_tei_cr[Nel*L + iel]);
		//std::cout << "prim_tei_cr_sub rows: " << prim_tei_cr_sub.n_rows << "    ";
		//std::cout << "prim_tei_cr_sub cols: " << prim_tei_cr_sub.n_cols << "    ";
		//std::cout << "\n";

		Paux_cr[L][M+Mmax] += cpl*(arma::trans(prim_tei_cr_sub)*Psubkli);
		//std::cout << "Paux_cr[" << L << "][" << M+Mmax << "] rows: " << Paux_cr[L][M+Mmax].n_rows << "    ";
		//std::cout << "\n";

	      }
            }
          }
        }

	arma::mat J_cr(Ndummy(),Ndummy());
        J_cr.zeros();

	for(size_t iang=0;iang<lval.n_elem;iang++) {
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            // l and m values
            int li(lval(iang));
            int mi(mval(iang));
            int lj(lval(jang));
            int mj(mval(jang));
            // LH m value
            int M(mj-mi);

            int Lmin=std::max(std::abs(lj-li),abs(M));
            int Lmax=lj+li;
            for(int L=Lmin;L<=Lmax;L++) {
              // Coupling
              double cpl(gaunt.coeff(lj,mj,L,M,li,mi));
              if(cpl!=0.0) {
                //J.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1)+=cpl*Jaux[L][M+Mmax];
		//arma::mat Jauxsub(Jaux[L][M+Mmax]);
		//std::cout << "Jaux[" << L << "][" << M+Mmax << "] rows: " << Jaux[L][M+Mmax].n_rows << "    ";
		//std::cout << "Jaux[" << L << "][" << M+Mmax << "] cols: " << Jaux[L][M+Mmax].n_cols << "    ";
		//std::cout << "\n";
		
		//std::cout << "Paux_cr[" << L << "][" << M+Mmax << "] rows: " << Paux_cr[L][M+Mmax].n_rows << "    ";
		//std::cout << "\n";

		for(size_t iel=0;iel<Nel;iel++) {
		  
		  size_t ifirst, ilast;
		  radial.get_idx(iel,ifirst,ilast);
		  size_t Ni(ilast-ifirst+1);
		  
		  arma::mat prod(cpl*(prim_tei_cr[Nel*L + iel] * Paux_cr[L][M+Mmax]));
		  prod.reshape(Ni,Ni);
		  //std::cout << "prod rows: " << prod.n_rows << "    ";
		  //std::cout << "prod cols: " << prod.n_cols << "    ";
		  //std::cout << "\n";
		  
		  J_cr.submat(iang*Nrad,jang*Nrad,(iang+1)*Nrad-1,(jang+1)*Nrad-1).submat(ifirst,ifirst,ilast,ilast) += prod;
		  
		}
              }
            }
          }
        }

	// std::cout << "J_cr rows: " << J_cr.n_rows << "    ";
	// std::cout << "J_cr cols: " << J_cr.n_cols << "    ";
	// std::cout << "\n";
	// std::cout << "J_cr:\n" << J_cr << "    ";

	return remove_boundaries(J_cr);
      }


      arma::mat TwoDBasis::exchange(const arma::mat & P0) const {
        if(!prim_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Full exchange matrix
        arma::mat K(Ndummy(),Ndummy());
        K.zeros();

        // Helper memory
#ifdef _OPENMP
        const int nth(omp_get_max_threads());
#else
        const int nth(1);
#endif
        std::vector<arma::vec> mem_Ksub(nth);
        std::vector<arma::vec> mem_Psub(nth);
        std::vector<arma::vec> mem_T(nth);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
          const int ith(omp_get_thread_num());
#else
          const int ith(0);
#endif
          // These are only small submatrices!
          mem_Psub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_Ksub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_T[ith].zeros(radial.max_Nprim()*radial.max_Nprim());

          // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            for(size_t kang=0;kang<lval.n_elem;kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              size_t N_L(2*arma::max(lval)+1);
              std::vector<arma::mat> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.n_elem;iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.n_elem;lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(arma::norm(P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1),"fro"));
                  //printf("(%i %i) (%i %i) density block norm %e\n",li,mi,ll,ml,bdens);
                  if(bdens<10*DBL_EPSILON)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    if(cpl==0.0)
                      continue;

                    // L factor
                    double Lfac=4.0*M_PI/(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);
                    couple[L]=true;
                  }
                }
              }

              // Loop over elements: output
              for(size_t iel=0;iel<Nel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                // Input
                for(size_t jel=0;jel<Nel;jel++) {
                  size_t jfirst, jlast;
                  radial.get_idx(jel,jfirst,jlast);

                  // Number of functions in the two elements
                  size_t Ni(ilast-ifirst+1);
                  size_t Nj(jlast-jfirst+1);

                  if(iel == jel) {
                    /*
                      The exchange matrix is given by
                      K(jk) = (ij|kl) P(il)
                      i.e. the complex conjugation hits i and l as
                      in the density matrix.

                      To get this in the proper order, we permute the integrals
                      K(jk) = (jk;il) P(il)
                    */

                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
                    Ksub.zeros();

                    for(size_t L=0;L<N_L;L++) {
                      if(!couple[L])
                        continue;
                      Ksub+=prim_ktei[Nel*Nel*L + iel*Nel + jel]*arma::vectorise(Rmat[L].submat(ifirst,jfirst,ilast,jlast));
                    }
                    Ksub.reshape(Ni,Nj);

                    // Increment global exchange matrix
                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;

                    //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
                    //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
                    //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));

                  } else {
                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                    Ksub.zeros();

                    for(size_t L=0;L<N_L;L++) {
                      if(!couple[L])
                        continue;

                      // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
                      const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
                      const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];

                      // Get density submatrix (Niel x Njel)
                      arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
                      Psub=Rmat[L].submat(ifirst,jfirst,ilast,jlast);

                      // Calculate helper
                      arma::mat T(mem_T[ith].memptr(),Ni,Nj,false,true);
                      // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                      T=Psub*arma::trans(jint);

                      // Increment
                      Ksub+=iint*T;
                    }

                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;
                  }
                }
              }
            }
          }
        }

        return remove_boundaries(K);
      }

      arma::mat TwoDBasis::exchange_cr(const arma::mat & P0) const {
        if(!prim_tei_cr.size())
          throw std::logic_error("Primitive CR teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());
	//std::cout << "CR exchange: Nrad = " << Nrad << "\n";
	// Max radial order of Coulomb resolution
	int Nmax = TwoDBasis::get_Nmax();

        // Full exchange matrix
        //arma::mat K(Ndummy(),Ndummy());
        //K.zeros();

        arma::mat K_cr(Ndummy(),Ndummy());
        K_cr.zeros();

	//        arma::mat K_cr2(Ndummy(),Ndummy());
	//        K_cr2.zeros();


        // Helper memory
#ifdef _OPENMP
        const int nth(omp_get_max_threads());
#else
	const int nth(1);
#endif
	std::vector<arma::vec> mem_Ksub(nth);
	std::vector<arma::vec> mem_Psub(nth);
	std::vector<arma::vec> mem_S(nth);
//	std::vector<arma::vec> mem_T(nth);
// 
#ifdef _OPENMP
#pragma omp parallel
#endif
	{
#ifdef _OPENMP
          const int ith(omp_get_thread_num());
#else
	const int ith(0);
#endif
//           // These are only small submatrices!
	mem_Psub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
	mem_Ksub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
	mem_S[ith].zeros(radial.max_Nprim()*radial.max_Nprim()*(Nmax+1));
//	mem_T[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
// 
          // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            for(size_t kang=0;kang<lval.n_elem;kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              size_t N_L(2*arma::max(lval)+1);
              std::vector<arma::mat> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.n_elem;iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.n_elem;lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(arma::norm(P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1),"fro"));
                  //printf("(%i %i) (%i %i) density block norm %e\n",li,mi,ll,ml,bdens);
                  if(bdens<10*DBL_EPSILON)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    if(cpl==0.0)
                      continue;

                    // L factor
                    double Lfac=4.0*M_PI/(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);
                    couple[L]=true;
                  }
                }
              }




              // Loop over elements: output
              //for(size_t iel=0;iel<Nel;iel++) {
              //  size_t ifirst, ilast;
              //  radial.get_idx(iel,ifirst,ilast);
	      //
              //  // Input
              //  for(size_t jel=0;jel<Nel;jel++) {
              //    size_t jfirst, jlast;
              //    radial.get_idx(jel,jfirst,jlast);
	      //
              //    // Number of functions in the two elements
              //    size_t Ni(ilast-ifirst+1);
              //    size_t Nj(jlast-jfirst+1);
	      //
              //    if(iel == jel) {
              //      /*
              //        The exchange matrix is given by
              //        K(jk) = (ij|kl) P(il)
              //        i.e. the complex conjugation hits i and l as
              //        in the density matrix.
	      //
              //        To get this in the proper order, we permute the integrals
              //        K(jk) = (jk;il) P(il)
              //      */
	      //
              //      // Exchange submatrix
              //      arma::mat Ksub(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
              //      Ksub.zeros();
	      //
              //      for(size_t L=0;L<N_L;L++) {
              //        if(!couple[L])
              //          continue;
              //        Ksub+=prim_ktei[Nel*Nel*L + iel*Nel + jel]*arma::vectorise(Rmat[L].submat(ifirst,jfirst,ilast,jlast));
              //      }
              //      Ksub.reshape(Ni,Nj);
	      //
              //      // Increment global exchange matrix
              //      K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;
	      //
              //      //arma::vec Ptgt(arma::vectorise(P.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)));
              //      //printf("(%i %i) (%i %i) (%i %i) (%i %i) [%i %i]\n",li,mi,lj,mj,lk,mk,ll,ml,L,M);
              //      //printf("Element %i - %i contribution to exchange energy % .10e\n",(int) iel,(int) jel,-0.5*arma::dot(Ksub,Ptgt));
	      //
              //    } else {
              //      // Exchange submatrix
              //      arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
              //      Ksub.zeros();
	      //
              //      for(size_t L=0;L<N_L;L++) {
              //        if(!couple[L])
              //          continue;
	      //
              //        // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
              //        const arma::mat & iint=(iel>jel) ? disjoint_m1L[L*Nel+iel] : disjoint_L[L*Nel+iel];
              //        const arma::mat & jint=(iel>jel) ? disjoint_L[L*Nel+jel] : disjoint_m1L[L*Nel+jel];
	      //
              //        // Get density submatrix (Niel x Njel)
              //        arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
              //        Psub=Rmat[L].submat(ifirst,jfirst,ilast,jlast);
	      //
              //        // Calculate helper
              //        arma::mat T(mem_T[ith].memptr(),Ni,Nj,false,true);
              //        // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
              //        T=Psub*arma::trans(jint);
	      //
              //        // Increment
              //        Ksub+=iint*T;
              //      }
	      //
              //      K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;
              //    }
              //  }
              //}


	      //std::vector< std::vector<arma::mat> > Tmat(Nel*Nmax*N_L);

	      //Tmat.resize(Nel*Nmax*N_L);

	      for(size_t L=0;L<N_L;L++) {
		//std::cout << "CR exchange: L=" << L << "\n";
		
		if(!couple[L])
		  continue;
		
		const double Lfac=4.0*M_PI/(2*L+1);
		
		for(int i=0;i<Nel;i++) {
		  //std::cout << "CR exchange: element i=" << i << "\n";
		  
		  size_t ifirst, ilast;
		  radial.get_idx(i,ifirst,ilast);
		  size_t Ni(ilast-ifirst+1);
		  
		  for(int l=0;l<Nel;l++) {
		    //std::cout << "CR exchange: element l=" << l << "\n";
		    size_t lfirst, llast;
		    radial.get_idx(l,lfirst,llast);
		    size_t Nl(llast-lfirst+1);
		    
		    //arma::mat Psub(Rmat[L].submat(ifirst,lfirst,ilast,llast)/Lfac);
		    arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nl,false,true);
		    Psub=Rmat[L].submat(ifirst,lfirst,ilast,llast)/Lfac;



		    //arma::mat prim_tei_cr_mat_i2(prim_tei_cr[Nel*L + i]);
		    //arma::cube prim_tei_cr_cube_i(prim_tei_cr_mat_i2.memptr(),Ni,Ni,Nmax+1,false,true);
		    arma::cube prim_tei_cr_cube_i((double*) prim_tei_cr[Nel*L + i].memptr(),Ni,Ni,Nmax+1,false,true); // probably bad idea


		    //arma::mat prim_tei_cr_mat_l2(prim_tei_cr[Nel*L + l]);
		    //arma::cube prim_tei_cr_cube_l(prim_tei_cr_mat_l2.memptr(),Nl,Nl,Nmax+1,false,true);
		    arma::cube prim_tei_cr_cube_l((double*) prim_tei_cr[Nel*L + l].memptr(),Nl,Nl,Nmax+1,false,true);

		    //K_cr2.submat(jang*Nrad+ifirst,kang*Nrad+lfirst,jang*Nrad+ilast,kang*Nrad+llast) -= prim_tei_cr_cube_i.each_slice()*Psub*prim_tei_cr_cube_l.each_slice();

		    //arma::cube Ssub(prim_tei_cr_cube_i.each_slice()*Psub);

		    //arma::cube S(mem_S[ith].memptr(),Ni,Nl,Nmax+1,false,true);
		    //S = prim_tei_cr_cube_i.each_slice()*Psub;

		    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nl,false,true);
		    Ksub.zeros();
		    
		    for(size_t N=0;N<=Nmax;N++) { // ideally this would also use broadcasting
		      //S.slice(N) *= prim_tei_cr_cube_l.slice(N);
		      //Ksub += S.slice(N)*prim_tei_cr_cube_l.slice(N);
		      Ksub += prim_tei_cr_cube_i.slice(N)*Psub*prim_tei_cr_cube_l.slice(N);
		    }

		    //K_cr.submat(jang*Nrad+ifirst,kang*Nrad+lfirst,jang*Nrad+ilast,kang*Nrad+llast) -= accum;
		    K_cr.submat(jang*Nrad+ifirst,kang*Nrad+lfirst,jang*Nrad+ilast,kang*Nrad+llast) -= Ksub;

		    // slower way:
		    //for(size_t N=0;N<=Nmax;N++) {
		    //  
		    //  arma::vec prim_tei_cr_vec_i(prim_tei_cr[Nel*L + i].col(N));
		    //  arma::mat prim_tei_cr_mat_i(prim_tei_cr_vec_i.memptr(),Ni,Ni,false,true);
		    //  
		    //  arma::vec prim_tei_cr_vec_l(prim_tei_cr[Nel*L + l].col(N));
		    //  arma::mat prim_tei_cr_mat_l(prim_tei_cr_vec_l.memptr(),Nl,Nl,false,true);
		    //
		    //  K_cr.submat(jang*Nrad+ifirst,kang*Nrad+lfirst,jang*Nrad+ilast,kang*Nrad+llast) -= prim_tei_cr_mat_i*Psub*prim_tei_cr_mat_l;
		    //}

		  }
		}
	      }
	    }
          }
	}



	  //arma::mat ret(remove_boundaries(K_cr / (4 * M_PI)));
	  arma::mat ret(remove_boundaries(K_cr));
	  //arma::mat ret2(remove_boundaries(K_cr2));

	  //std::cout << "CR exchange: K_cr = \n" << ret << "\n";
	  //std::cout << "CR exchange: K_cr2 = \n" << ret2 << "\n";

	  //arma::mat origret(remove_boundaries(K));

	  //std::cout << "CR exchange: (non-CR) K_cr = \n" << origret << "\n";

	  //std::cout << "CR exchange: (CR/non-CR) K_cr/K = \n" << ret/origret << "\n";

	  return ret;
      }


      arma::mat TwoDBasis::rs_exchange(const arma::mat & P0) const {
        if(!rs_ktei.size())
          throw std::logic_error("Primitive teis have not been computed!\n");

        // Extend to boundaries
        arma::mat P(expand_boundaries(P0));

        // Gaunt coefficient table
        int gmax(std::max(arma::max(lval),arma::max(mval)));
        gaunt::Gaunt gaunt(gmax,2*gmax,gmax);

        // Number of radial elements
        size_t Nel(radial.Nel());
        // Number of radial basis functions
        size_t Nrad(radial.Nbf());

        // Full exchange matrix
        arma::mat K(Ndummy(),Ndummy());
        K.zeros();

        // Helper memory
#ifdef _OPENMP
        const int nth(omp_get_max_threads());
#else
        const int nth(1);
#endif
        std::vector<arma::vec> mem_Ksub(nth);
        std::vector<arma::vec> mem_Psub(nth);
        std::vector<arma::vec> mem_T(nth);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
          const int ith(omp_get_thread_num());
#else
          const int ith(0);
#endif
          // These are only small submatrices!
          mem_Psub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_Ksub[ith].zeros(radial.max_Nprim()*radial.max_Nprim());
          mem_T[ith].zeros(radial.max_Nprim()*radial.max_Nprim());

          // Increment
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
          for(size_t jang=0;jang<lval.n_elem;jang++) {
            for(size_t kang=0;kang<lval.n_elem;kang++) {
              int lj(lval(jang));
              int mj(mval(jang));

              int lk(lval(kang));
              int mk(mval(kang));

              // Form radial helpers
              size_t N_L(2*arma::max(lval)+1);
              std::vector<arma::mat> Rmat(N_L);
              for(size_t i=0;i<N_L;i++) {
                Rmat[i].zeros(Nrad,Nrad);
              }
              // Is there a coupling to the channel?
              std::vector<bool> couple(N_L,false);

              // Perform angular sums
              for(size_t iang=0;iang<lval.n_elem;iang++) {
                int li(lval(iang));
                int mi(mval(iang));

                for(size_t lang=0;lang<lval.n_elem;lang++) {
                  int ll(lval(lang));
                  int ml(mval(lang));

                  // LH m value
                  int M(mj-mi);
                  // RH m value
                  int Mp(mk-ml);
                  if(M!=Mp)
                    continue;

                  // Do we have any density in this block?
                  double bdens(arma::norm(P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1),"fro"));
                  //printf("(%i %i) (%i %i) density block norm %e\n",li,mi,ll,ml,bdens);
                  if(bdens<10*DBL_EPSILON)
                    continue;

                  // M values match. Loop over possible couplings
                  int Lmin=std::max(std::max(std::abs(li-lj),std::abs(lk-ll)),abs(M));
                  int Lmax=std::min(li+lj,lk+ll);

                  for(int L=Lmin;L<=Lmax;L++) {
                    // Calculate total coupling coefficient
                    double cpl(gaunt.coeff(lj,mj,L,M,li,mi)*gaunt.coeff(lk,mk,L,M,ll,ml));
                    if(cpl==0.0)
                      continue;

                    // L factor
                    double Lfac = yukawa ? 4.0*M_PI*lambda :  4.0*M_PI*lambda/(2*L+1);
                    Rmat[L]+=(Lfac*cpl)*P.submat(iang*Nrad,lang*Nrad,(iang+1)*Nrad-1,(lang+1)*Nrad-1);
                    couple[L]=true;
                  }
                }
              }

              // Loop over elements: output
              for(size_t iel=0;iel<Nel;iel++) {
                size_t ifirst, ilast;
                radial.get_idx(iel,ifirst,ilast);

                // Input
                for(size_t jel=0;jel<Nel;jel++) {
                  size_t jfirst, jlast;
                  radial.get_idx(jel,jfirst,jlast);

                  // Number of functions in the two elements
                  size_t Ni(ilast-ifirst+1);
                  size_t Nj(jlast-jfirst+1);

                  // error function does not factorize
                  if(!yukawa || iel == jel) {
                    /*
                      The exchange matrix is given by
                      K(jk) = (ij|kl) P(il)
                      i.e. the complex conjugation hits i and l as
                      in the density matrix.

                      To get this in the proper order, we permute the integrals
                      K(jk) = (jk;il) P(il)
                    */

                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni*Nj,1,false,true);
                    Ksub.zeros();

                    for(size_t L=0;L<N_L;L++) {
                      if(!couple[L])
                        continue;
                      Ksub+=rs_ktei[Nel*Nel*L + iel*Nel + jel]*arma::vectorise(Rmat[L].submat(ifirst,jfirst,ilast,jlast));
                    }
                    Ksub.reshape(Ni,Nj);

                    // Increment global exchange matrix
                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;

                  } else {
                    // Exchange submatrix
                    arma::mat Ksub(mem_Ksub[ith].memptr(),Ni,Nj,false,true);
                    Ksub.zeros();

                    for(size_t L=0;L<N_L;L++) {
                      if(!couple[L])
                        continue;

                      // Disjoint integrals. When r(iel)>r(jel), iel gets -1-L, jel gets L.
                      const arma::mat & iint=(iel>jel) ? disjoint_kL[L*Nel+iel] : disjoint_iL[L*Nel+iel];
                      const arma::mat & jint=(iel>jel) ? disjoint_iL[L*Nel+jel] : disjoint_kL[L*Nel+jel];

                      // Get density submatrix (Niel x Njel)
                      arma::mat Psub(mem_Psub[ith].memptr(),Ni,Nj,false,true);
                      Psub=Rmat[L].submat(ifirst,jfirst,ilast,jlast);

                      // Calculate helper
                      arma::mat T(mem_T[ith].memptr(),Ni,Nj,false,true);
                      // (Niel x Njel) = (Niel x Njel) x (Njel x Njel)
                      T=Psub*arma::trans(jint);

                      // Increment
                      Ksub+=iint*T;
                    }

                    K.submat(jang*Nrad+ifirst,kang*Nrad+jfirst,jang*Nrad+ilast,kang*Nrad+jlast)-=Ksub;
                  }
                }
              }
            }
          }
        }

        return remove_boundaries(K);
      }

      arma::mat TwoDBasis::remove_boundaries(const arma::mat & Fnob) const {
        if(Fnob.n_rows != Ndummy() || Fnob.n_cols != Ndummy()) {
          std::ostringstream oss;
          oss << "Matrix does not have expected size! Got " << Fnob.n_rows << " x " << Fnob.n_cols << ", expected " << Ndummy() << " x " << Ndummy() << "!\n";
          throw std::logic_error(oss.str());
        }

        // Get indices
        arma::uvec idx(pure_indices());

        // Matrix with the boundary conditions removed
        arma::mat Fpure(Fnob(idx,idx));

        //Fnob.print("Input: w/o built-in boundaries");
        //Fpure.print("Output: w built-in boundaries");

        return Fpure;
      }

      arma::mat TwoDBasis::expand_boundaries(const arma::mat & Ppure) const {
        if(Ppure.n_rows != Nbf() || Ppure.n_cols != Nbf()) {
          std::ostringstream oss;
          oss << "Matrix does not have expected size! Got " << Ppure.n_rows << " x " << Ppure.n_cols << ", expected " << Nbf() << " x " << Nbf() << "!\n";
          throw std::logic_error(oss.str());
        }

        // Get indices
        arma::uvec idx(pure_indices());

        // Matrix with the boundary conditions removed
        arma::mat Pnob(Ndummy(),Ndummy());
        Pnob.zeros();
        Pnob(idx,idx)=Ppure;

        //Ppure.print("Input: w built-in boundaries");
        //Pnob.print("Output: w/o built-in boundaries");

        return Pnob;
      }

      std::vector<arma::mat> TwoDBasis::get_prim_tei() const {
        return prim_tei;
      }

      arma::cx_mat TwoDBasis::eval_bf(size_t iel, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat rad(radial.get_bf(iel));

        // Form supermatrix
        arma::cx_mat bf(rad.n_rows,lval.n_elem*rad.n_cols);
        for(size_t i=0;i<lval.n_elem;i++)
          bf.cols(i*rad.n_cols,(i+1)*rad.n_cols-1)=sph(i)*rad;

        return bf;
      }

      void TwoDBasis::eval_df(size_t iel, double cth, double phi, arma::cx_mat & dr, arma::cx_mat & dth, arma::cx_mat & dphi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::mat frad(radial.get_bf(iel));
        arma::mat drad(radial.get_df(iel));

        // Form supermatrices
        dr.zeros(frad.n_rows,lval.n_elem*frad.n_cols);
        dth.zeros(frad.n_rows,lval.n_elem*frad.n_cols);
        dphi.zeros(frad.n_rows,lval.n_elem*frad.n_cols);

        // Radial one is easy
        for(size_t i=0;i<lval.n_elem;i++)
          dr.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=sph(i)*drad;
        // and so is phi
        for(size_t i=0;i<lval.n_elem;i++)
          dphi.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=std::complex<double>(0.0,mval(i))*sph(i)*frad;
        // but theta is nastier
        for(size_t i=0;i<lval.n_elem;i++) {
          // cot th = 1/tan th = cos th / sin th
          double cotth=cth/sqrt(1.0-cth*cth);

          int l(lval(i));
          int m(mval(i));

          // Angular factor
          std::complex<double> angfac(m*cotth*sph(i));
          if(mval(i)<lval(i))
            angfac+=sqrt((l-m)*(l+m+1))*std::exp(std::complex<double>(0,-phi))*::spherical_harmonics(lval(i),mval(i)+1,cth,phi);

          dth.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=angfac*frad;
        }
      }

      arma::cx_mat TwoDBasis::eval_lf(size_t iel, double cth, double phi) const {
        // Evaluate spherical harmonics
        arma::cx_vec sph(lval.n_elem);
        for(size_t i=0;i<lval.n_elem;i++)
          sph(i)=::spherical_harmonics(lval(i),mval(i),cth,phi);

        // Evaluate radial functions
        arma::vec r(radial.get_r(iel));
        arma::mat frad(radial.get_bf(iel));
        arma::mat drad(radial.get_df(iel));
        arma::mat lrad(radial.get_lf(iel));

        // Form supermatrix
        arma::cx_mat lf(frad.n_rows,lval.n_elem*frad.n_cols,arma::fill::zeros);
        // Loop over basis function indices
        for(size_t iang=0;iang<lval.n_elem;iang++)
          for(size_t irad=0;irad<frad.n_cols;irad++)
            // Loop over grid-point indices
            for(size_t igrid=0;igrid<frad.n_rows;igrid++)
              lf(igrid,iang*frad.n_cols+irad) = (lrad(igrid,irad) + 2*drad(igrid,irad)/r(igrid) - lval(iang)*(lval(iang)+1)*frad(igrid,irad)/(r(igrid)*r(igrid)))*sph(iang);

        //for(size_t i=0;i<lval.n_elem;i++)
        //lf.cols(i*frad.n_cols,(i+1)*frad.n_cols-1)=(arma::square(r)%lrad + 2*r%drad - lval(i)*(lval(i)+1)*frad) / arma::square(r) * sph(i);

        return lf;
      }

      arma::uvec TwoDBasis::bf_list(size_t iel) const {
        // Radial functions in element
        size_t ifirst, ilast;
        radial.get_idx(iel,ifirst,ilast);
        // Number of radial functions in element
        size_t Nr(ilast-ifirst+1);

        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // List of functions in the element
        arma::uvec idx(Nr*lval.n_elem);
        for(size_t iam=0;iam<lval.n_elem;iam++)
          for(size_t j=0;j<Nr;j++)
            idx(iam*Nr+j)=Nrad*iam+ifirst+j;

        //printf("Basis function in element %i\n",(int) iel);
        //idx.print();

        return idx;
      }

      size_t TwoDBasis::get_rad_Nel() const {
        return radial.Nel();
      }

      arma::vec TwoDBasis::get_wrad(size_t iel) const {
        return radial.get_wrad(iel);
      }

      arma::vec TwoDBasis::get_r(size_t iel) const {
        return radial.get_r(iel);
      }

      arma::vec TwoDBasis::nuclear_density(const arma::mat & P0) const {
        // Radial functions in first element
        size_t ifirst, ilast;
        radial.get_idx(0,ifirst,ilast);
        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // Expand density matrix to boundary conditions
        arma::mat P(expand_boundaries(P0));

        // Loop over angular momentum
        double nucden=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nucden)
#endif
        for(size_t iam=0;iam<lval.n_elem;iam++) {
          // Integration over angles yields extra factor 4 pi that must be removed
          nucden+=radial.nuclear_density(P.submat(Nrad*iam,Nrad*iam,Nrad*(iam+1)-1,Nrad*(iam+1)-1))/(4.0*M_PI);
        }

        arma::vec den(1);
        den(0)=nucden;

        return den;
      }

      arma::vec TwoDBasis::nuclear_density_gradient(const arma::mat & P0) const {
        // Radial functions in first element
        size_t ifirst, ilast;
        radial.get_idx(0,ifirst,ilast);
        // Total number of radial functions
        size_t Nrad(radial.Nbf());

        // Expand density matrix to boundary conditions
        arma::mat P(expand_boundaries(P0));

        // Loop over angular momentum
        double nucden=0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:nucden)
#endif
        for(size_t iam=0;iam<lval.n_elem;iam++) {
          // Integration over angles yields extra factor 4 pi that must be removed
          nucden+=radial.nuclear_density_gradient(P.submat(Nrad*iam,Nrad*iam,Nrad*(iam+1)-1,Nrad*(iam+1)-1))/(4.0*M_PI);
        }

        arma::vec den(1);
        den(0)=nucden;

        return den;
      }

      int TwoDBasis::get_Nmax() const {
	if (Nmax < 0) {
	  throw std::logic_error("CR Nmax not set!\n");
	} else {
	  return Nmax;
	}
      }

      void TwoDBasis::set_Nmax(int n) {
	Nmax = n;
	return;
      }
      
      double TwoDBasis::get_rsfrac() const {
	return rsfrac;
      }

      void TwoDBasis::set_rsfrac(double rsf) {
	rsfrac = rsf;
	return;
      }

    }
  }
}
