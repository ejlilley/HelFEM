#include <vector>
#include <armadillo>

namespace helfem {
  namespace cr {
    typedef struct {
      /// Value of argument
      double r;
      /// Phinl values
      arma::mat Phinl;
    } phinl_table_t;

    bool operator<(const phinl_table_t & lh, const phinl_table_t & rh);

    class PhinlTable {
    private:
      /// Storage array
      std::vector<phinl_table_t> stor;
      /// Maximum L value used in the actual computation
      //int Lpad;

      /// which Coulomb resolution we are using (0: Zhao, 1: ...)
      int CR;
      /// Maximum N value
      int Nmax;
      /// Maximum L value
      int Lmax;
      /// alpha (affects power-law behaviour of Zhao CR)
      double alpha;

      /// Find index in array
      size_t get_index(double r, bool check=true) const;

      arma::mat Nnl;

      void compute_Nnl();

    public:
      /// Dummy constructor
      PhinlTable();
      /// Constructor
      PhinlTable(int CR, int Nmax, int Lmax, double alpha);
      /// Destructor
      ~PhinlTable();
      /// Add value to table
      void compute(double r);

      double phi0l(int l, double r);

      arma::mat get_Nnl();

      /// Get value from table
      double get_Phinl(int n, int l, double r) const;

      /// Get value from table
      arma::vec get_Phinl(int n, int l, const arma::vec & r) const;
    };

    typedef struct {
      double r;
      arma::cube Iknl;
    } iknl_table_t;

    bool operator<(const iknl_table_t & lh, const iknl_table_t & rh);

    class IknlTable {
    private:
      /// Storage array
      std::vector<iknl_table_t> stor;

      /// which Coulomb resolution we are using (0: Zhao, 1: ...)
      int CR;
      /// Maximum K value
      int Kmax;
      /// Maximum N value
      int Nmax;
      /// Maximum L value
      int Lmax;
      /// alpha (affects power-law behaviour of Zhao CR)
      double alpha;

      /// Find index in array
      size_t get_index(double r, bool check=true) const;

      // Iknl gets its own internal phinl object
      PhinlTable phinl;

    public:
      /// Dummy constructor
      IknlTable();
      /// Constructor
      IknlTable(int CR, int Kmax, int Nmax, int Lmax, double alpha);
      /// Destructor
      ~IknlTable();
      /// Add value to table
      void compute(double r);

      double Ik0l(int k, int l, double r);

      arma::mat get_Nnl();

      /// Get value from table
      double get_Iknl(int k, int n, int l, double r) const;

      /// Get value from table
      arma::vec get_Iknl(int k, int n, int l, const arma::vec & r) const;

      int get_Nmax();

    };
  }
}
