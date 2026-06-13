#pragma once

#include <vector>
#include <armadillo>

namespace helfem {
  namespace cr {
    typedef struct {
      /// Value of argument
      double mu;
      /// Phinl values
      arma::cube Phinlm;
    } phinlm_table_t;

    bool operator<(const phinlm_table_t & lh, const phinlm_table_t & rh);

    class PhinlmTable {
    private:
      /// Storage array
      std::vector<phinlm_table_t> stor;
      /// Maximum L value used in the actual computation
      //int Lpad;

      /// which Coulomb resolution we are using (0: Zhao, 1: ...)
      int CR;
      /// Maximum N value
      int Nmax;
      /// Maximum L value
      int Lmax;
      /// Maximum |M| value
      int Mmax;

      double Rh;

      /// Find index in array
      size_t get_index(double mu, bool check=true) const;

      arma::cube Nnlm;

      void compute_Nnlm();

    public:
      /// Dummy constructor
      PhinlmTable();
      /// Constructor
      PhinlmTable(int CR, int Nmax, int Lmax, int Mmax, double Rh);
      /// Destructor
      ~PhinlmTable();
      /// Add value to table
      void compute(double mu);

      /// Zeroth order of phinl
      double phi0lm(int l, int m, double mu);

      /// Return (vector of first n) real form of index-raising
      /// polynomials pnlm(s) (used for computing Taylor expansions)
      arma::vec pnlm(int n, int l, int m, double s);
      // Same but return one column for each argument
      arma::mat pnlm_mat(int n, int l, int m, const arma::vec & s);

      arma::cube get_Nnlm();

      int get_Nmax();
      int get_Lmax();
      int get_Mmax();
      double get_Rh();

      /// Get value(s) from table
      double get_Phinlm(int n, int l, int m, double mu) const;
      arma::vec get_Phinlm(int n, int l, int m, const arma::vec & mu) const;
      arma::cube get_Phinlm(double mu) const;

    };

  }
}
