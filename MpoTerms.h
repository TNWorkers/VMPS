#ifndef DMRG_HAMILTONIAN_TERMS
#define DMRG_HAMILTONIAN_TERMS
#define EIGEN_DONT_VECTORIZE
#ifndef DEBUG_VERBOSITY
#define DEBUG_VERBOSITY 1
#endif

/// \cond
#include <vector>
#include <string>
#include "boost/multi_array.hpp"
/// \endcond

#include "numeric_limits.h" // from TOOLS
#include "Stopwatch.h" // from TOOLS
#include "tensors/SiteOperator.h"
#include "symmetry/qarray.h"
#include "tensors/Qbasis.h"
#include "tensors/Biped.h"
#include "MemCalc.h" // from TOOLS

template<typename Symmetry, typename Scalar> class MpoTerms
{
    typedef SiteOperator<Symmetry,Scalar> OperatorType;
    typedef typename Symmetry::qType qType;
    typedef Eigen::SparseMatrix<Scalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> MatrixType;
    
private:
    
    /**
     *  Operators that have been pushed into this instance of MpoTerms.
     *  Index structure: [Lattice site] Map: {qIn, qOut} -> [row][column]
     */
    std::vector<std::map<std::array<qType, 2>,std::vector<std::vector<std::map<qType,OperatorType>>>>> O;

    
    /**
     *  Dimension of the auxiliar MPO basis
     *  Index i extends from 0 to N_sites, auxdim[i] connects lattice site i-1 and lattice site i
     *  Index structure: [i] Map: {q} -> value
     */
    std::vector<std::map<qType, std::size_t>> auxdim;
    
    /**
     *  Total quantum number of the MPO. All interactions pushed have to lead to this quantum number.
     */
    qType qTot;
    
    /**
     *  Vacuum quantum number of the MPO.
     */
    qType qVac = Symmetry::qvacuum();
    
    /**
     *  Index of right unity operator (operator thas is right to any local operators) in the quantum number block qTot. Equals 1, if qTot=qVac, and 0 else.
     */
    std::size_t pos_qTot;
    
    /**
     *  Index of left unity operator (operator that is left to any local operators) in the quantum number block Symmetry::qvacuum(). Equals 0.
     */
    std::size_t pos_qVac;
    
    /**
     *  Stores whether to use open boundary conditions or not (for VUMPS)
     */
    BC boundary_condition;
	
    /**
     *  All informations stored about the MpoTerms
     *  Index structure [Lattice Site i][Index of information string]
     */
    std::vector<std::vector<std::string>> info;
    
    /**
     *  Local hilbert space dimensions. Initialized with 0 and set whenever an operator is added at a given site
     *  Index structure: [Lattice Site]
     */
	std::vector<int> hilbert_dimension;
    
    /**
     *  Stores whether the MpoTerms have already been compressed. Afterwards, operators cannot be pushed anymore.
     */
    bool FINALIZED = false;
    
    /**
    *  Set true if nothing shall be logged
    */
    //bool SILENT = false;
    DMRG::VERBOSITY::OPTION VERB = DMRG::VERBOSITY::OPTION::SILENT;
    std::string before_verb_set;
    
    /**
     *  Local operator bases.
     *  Index structure: [Lattice Site][Index of quantum number]
     */
    std::vector<std::vector<qType>> qOp;
    
    /**
     *  MPO bases. After calculation this encodes the same information as auxdim, but in a convenient way.
     *  Index structure: [Lattice site]
     */
    std::vector<Qbasis<Symmetry>> qAux;
    
    /**
     *  Bases of local Hilbert spaces.
     *  Index structure: [Lattice Site][Index of quantum number]
     */
    std::vector<std::vector<qType>> qPhys;
    
    /**
     *  Operator bases of the higher powers
     *  Index structure: [Power-2][Lattice Site][Index of quantum number]
     */
    std::vector<std::vector<std::vector<qType>>> qOp_powers;
    
    /**
     *  MPO bases of the higher powers
     *  Index structure: [Power-2][Lattice site]
     */
    std::vector<std::vector<Qbasis<Symmetry>>> qAux_powers;
    
    /**
     *  Stores the highest power of the MPO that has been calculated.
     */
    std::size_t current_power = 1;
    
    /**
     *  Is qOp up to date?
     */
    bool GOT_QOP = false;
    
    /**
     *  Is qAux up to date?
     */
    bool GOT_QAUX = false;
    
    /**
     *  Is W up to date?
     */
    bool GOT_W = false;
    
    /**
     *  Has qPhys been set at a certrain lattice site?
     *  Index structure: [Lattice Site]
     */
    std::vector<bool> GOT_QPHYS;
    
    /**
     *  Increments the MPO auxiliar basis dimension by one. Also manages allocation of O.
     *  If BC::INFINITE, loc >= N_sites is allowed. In this case, the method increments qAux[0] and qAux[N_sites] for loc = N_sites
     *  @param  loc  Lattice site, for BC::OPEN: 0 < loc < N_sites
     *  @param  q    Quantum number
     */
    void increment_auxdim(const std::size_t loc, const qType& q);
    
    /**
     *  Increments the MPO auxiliar basis dimension connecting legt edge and first lattice site by one, also manages allocation of O. Just for open boundary condition, when this cannot be done by increment_auxdim.
     *  @param  qIn Quantum number
     */
    void increment_first_auxdim_OBC(const qType& qIn);
    
    /**
     *  Increments the MPO auxiliar basis dimension connectinglast lattice site and right edge by one, also manages allocation of O. Just for open boundary condition, when this cannot be done by increment_auxdim.
     *  @param  qOut    Quantum number
     */
    void increment_last_auxdim_OBC(const qType& qOut);
    
    /**
     *  Decrements the MPO auxiliar basis dimension by one. Also manages reshaping of O.
     *  If BC::INFINITE, loc >= N_sites is allowed. In this case, the method decrements qAux[0] and qAux[N_sites] for loc = N_sites
     *  @param  loc  Lattice site, for BC::OPEN: 0 < loc < N_sites
     *  @param  q    Quantum number
     */
    void decrement_auxdim(const std::size_t loc, const qType& q);
    
    /**
     *  Decrements the MPO auxiliar basis dimension connecting legt edge and first lattice site by one, also manages reshaping of O. Just for open boundary condition, when this cannot be done by decrement_auxdim.
     *  @param  qIn Quantum number
     */
    void decrement_first_auxdim_OBC(const qType& qIn);
    
    /**
     *  Decrements the MPO auxiliar basis dimension connectinglast lattice site and right edge by one, also manages reshaping of O. Just for open boundary condition, when this cannot be done by decrement_auxdim.
     *  @param  qOut    Quantum number
     */
    void decrement_last_auxdim_OBC(const qType& qOut);
    
    /**
     *  Adds an operator in O
     *  @param  loc         Lattice site
     *  @param  op          Operator
     *  @param  qIn         Incoming quantum number block for row
     *  @param  qOut        Outgoing quantum number block for column
     *  @param  IndexIn     Row index
     *  @param  IndexOut    Column index
     */
    void add(const std::size_t loc, const OperatorType& op, const qType& qIn, const qType& qOut, const std::size_t IndexIn, const std::size_t IndexOut);
    
    /**
     *  Calculates the dimension of the MPO auxiliar basis.
     *  If BC::INFINITE, loc >= N_sites is allowed. In this case qAux[0] = qAux[N_sites]
     *  @param  loc The auxiliar basis index, i.e. the basis connecting lattice site loc-1 and loc
     *  @param  q   Quantum number
     *  @return  Dimension of auxiliar basis
     */
    std::size_t get_auxdim(const std::size_t loc, const qType& q) const;
    
    /**
     *  Checks whether the dimension of an operator matches the local Hilbert space dimension. Sets the latter if it has not been set yet.
     *  @param  loc  Lattice site
     *  @param  dim  Assumed dimension of local Hilbert space
     */
    void assert_hilbert(const std::size_t loc, const std::size_t dim);
    
    /**
     *  Checks for linearly dependent rows (including zero rows) within a certain block in O and manages their deletion.
     *  @param  loc  Lattice site
     *  @param  qIn  Quantum number of the block to check
     */
    bool eliminate_linearlyDependent_rows(const std::size_t loc, const qType& qIn, const double tolerance);
    
    /**
     *  Checks for linearly dependent columns (including zero columns) within a certain block in O and manages their deletion.
     *  @param  loc  Lattice site
     *  @param  qOut Quantum number of the block to check
     *  @param  tolerance    Threshold up to which value linear dependencies shall be removed
     */
    bool eliminate_linearlyDependent_cols(const std::size_t loc, const qType& qOut, const double tolerance);
    
    /**
     *  Deletes a certain row in O. Does not resize O, but shifts the empty row to the bottom.
     *  @param  loc              Lattice site
     *  @param  qIn              Quantum number of the block
     *  @param  row_to_delete    Index of the row
     *  @param  SAMESITE    If true, then the row with the same quantum number is skipped. Useful for VUMPS compression, when there is only one lattice site.
     *  @return  Deleted row as a map {q} -> [col]
     */
    std::map<qType, std::vector<std::map<qType,OperatorType>>> delete_row(const std::size_t loc, const qType& qIn, const std::size_t row_to_delete, const bool SAMESITE);
    
    /**
     *  Deletes a certain column in O. Does not resize O, but shifts the empty column to the right.
     *  @param  loc              Lattice site
     *  @param  qOut             Quantum number of the block
     *  @param  col_to_delete    Index of the column
     *  @param  SAMESITE    If true, then the column with the same quantum number is skipped. Useful for VUMPS compression, when there is only one lattice site.
     *  @return  Deleted column as a map {q} -> [row]
     */
    std::map<qType, std::vector<std::map<qType,OperatorType>>> delete_col(const std::size_t loc, const qType& qOut, const std::size_t col_to_delete, const bool SAMESITE);
    
    /**
     *  Adds a multiple of another row to a certain row
     *  @param  loc     Lattice site
     *  @param  qIn     Quantum number of the block
     *  @param  row     Index of the row
     *  @param  ops     Map {q} -> [col] of the row that shall be added
     *  @param  factor  Optional factor to scale the added row
     */
    void add_to_row(const std::size_t loc, const qType& qIn, const std::size_t row, const std::map<qType,std::vector<std::map<qType,OperatorType>>>& ops, const Scalar factor);
    
    /**
     *  Adds a multiple of another column to a certain column
     *  @param  loc     Lattice site
     *  @param  qOut    Quantum number of the block
     *  @param  row     Index of the column
     *  @param  ops     Map {q} -> [row] of the column that shall be added
     *  @param  factor  Optional factor to scale the added column
     */
    void add_to_col(const std::size_t loc, const qType& qOut, const std::size_t col, const std::map<qType,std::vector<std::map<qType,OperatorType>>>& ops, const Scalar factor);
    
    /**
     *  Calculates all local operator bases by checking which quantum numbers appear in the respective operator set. QOT_QOP is set true.
     */
    void calc_qOp();
    
    /**
     *  Calculates the MPO bond bases by analyzing auxdim. GOT_QAUX is set true.
     */
    void calc_qAux();
    
    /**
     *  Calculates the W matrix. Needs qOp calculated before. GOT_W is set true.
     */
    void calc_W();
    
    /**
     *  Sets GOT_QAUX, GOT_QOP and GOT_W to false, sets current_power to 1. Called whenever something is changed.
     */
    inline void got_update();
    
     /**
      *  Clears all relevant members and sets the right size for them.
       */
     void initialize();
    
    /**
     *  Compresses the MpoTerms. Needs FINALIZED Terms.
     *  @param tolerance    Threshold up to which value linear dependencies shall be removed
     */
    void compress(const double tolerance);

    /**
     *  Distributes the total norm of the MPO equally between all lattice sites
     */
    void renormalize();

    /**
     *  Empties all operator labels in O
     */
    void clear_opLabels();
    
    /**
     *  Calculates the average MPO bond dimension and the maximum MPO bond dimension
     *  @return  [Average,Maximum]
     */
    inline std::tuple<double,std::size_t> auxdim_infos() const;
    
    /**
     * Swaps rows and columns after calculating the product of two MPOs for VUMPS. Ensures that the identity operator that connects the vacua at each lattice site appear at position [0|0] and the identity connecting the target quantum number (for VUMPS vacuum) at position [1|1].
     *  @param  O_out   Operators as created by the prod-method.
     *  @param  row_qVac    List of the rows of the vacuum identity operators for each lattice site
     *  @param  col_qVac    List of the columns of the vacuum identity operators for each lattice site
     *  @param  row_qVac    List of the rows of the target quantum number identity operators for each lattice site
     *  @param  row_qVac    List of the columns of the target quantum number identity operators for each lattice site
     */
    static void prod_swap_IBC(std::vector<std::map<std::array<qType,2>,std::vector<std::vector<std::map<qType,OperatorType>>>>>& O_out, std::vector<std::size_t>& row_qVac, std::vector<std::size_t>& col_qVac, std::vector<std::size_t>& row_qTot, std::vector<std::size_t>& col_qTot);
    
    /**
     *  Deletes zero columns at the last lattice site after calculating the product of two MPOs when the combination of twice the target quantum number is not unique.
     *  @param  O_last  O matrix of the product MPO at last lattice site. Will be truncated.
     *  @param  qAux_last   Auxiliar basis of the product MPO at last lattice site. Will be truncated..
     *  @param  qAux_prev   Auxiliar basis of the product MPO at lattice site left to the last one. Will not be modified.
     *  @param  qTot    Target quantum number of the product MPO.
     *  @param  col_qTot    Column of the vacuum identity operators for last lattice site
     */
    static void prod_delZeroCols_OBC(std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>& O_last, Qbasis<Symmetry>& qAux_last, Qbasis<Symmetry>& qAux_prev, const qType& qTot, const std::size_t col_qTot);

protected:
    
    /**
     *  W matrix that stores the MPO in a convenient way.
     *  Index structure: [Lattice site][Upper Hilbert space basis entry][Lower Hilbert space basis entry][Quantum number of operators] Biped: {qIn,qOut} -> Matrix [row][column]
     */
    std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>> W;
    
    /**
     *  W matrix of the higher powers
     *  Index structure: [Power-2][Lattice site][Upper Hilbert space basis entry][Lower Hilbert space basis entry][Quantum number of operators] Biped: {qIn,qOut} -> Matrix [row][column]
     */
    std::vector<std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>>> W_powers;
    
    /**
     *  Number of lattice sites (for VUMPS: in the unit cell)
     */
    std::size_t N_sites;
    
    /**
     *  Allows to reconstruct an instance of MpoTerms with preset data.
     *  @param  O_in            Operators, are copied into O
     *  @param  qAux_in         Qbasis from which auxdim is calculated
     *  @param  FINALIZED_IN    Shall this instance be FINALIZED?
     *  @param  boundary_condition_in   Boundary condition
     *  @param  qTot_in    Total quantum number of the MPO
     */
    void reconstruct(const std::vector<std::map<std::array<qType,2>, std::vector<std::vector<std::map<qType,OperatorType>>>>>& O_in, const std::vector<Qbasis<Symmetry>>& qAux_in, const std::vector<std::vector<qType>>& qPhys_in, const bool FINALIZED_IN, const BC boundary_condition_in, const qType& qTot_in = Symmetry::qvacuum());
    
public:

    /**
     *  Constructor for an instance of MpoTerms with fixed lattice size (for VUMPS: of the unit cell)
     *  @param  L   Lattice size
     *  @param  boundary_condition_in   Boundary condition
     *  @param  qTot_in Total quantum number of the MPO
     */
    MpoTerms(const std::size_t L=0, const BC boundary_condition_in=BC::OPEN, const qType& qTot_in=Symmetry::qvacuum());
    
    /**
     *  Same as initialize, but allows to set a new combination of lattice size, total MPO quantum number and boundary condition
     *  @param  L   Lattice size
     *  @param  boundary_condition_in   Boundary condition
     *  @param  qTot_in Total quantum number of the MPO
     */
    void initialize(const std::size_t L, const BC boundary_condition_in, const qType& qTot_in);
	
    /**
     *  Pushes an interaction into this instance of MpoTerms.
     *  @param  loc     Lattice site where the interaction starts
     *  @param  opList  Vector of operators that make up the interaction. opList[i] acts on lattice site loc+i
     *  @param  qList   Vector of quantum numbers. qList[i] is left from lattice site loc+i, qList[i+1] is right from it
     *  @param  lambda  Scalar factor for the interaction
     */
    void push(const std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, const Scalar lambda = 1.0);
    
    /**
     *  Pushes an interaction into this instance of MpoTerms, but with reversed order of lattice sites. Needed, when get_reversed_data() shall be called afterwards.
     *  @param  loc     Lattice site where the interaction starts
     *  @param  opList  Vector of operators that make up the interaction. opList[i] acts on lattice site loc+i
     *  @param  qList   Vector of quantum numbers. qList[i] is left from lattice site loc+i, qList[i+1] is right from it
     *  @param  lambda  Scalar factor for the interaction
    */
    void push_reverse(const std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, const Scalar lambda = 1.0);
    
    /**
     *  Generates the quantum number list and then calls push(std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, Scalar lambda = 1.0).
     *  If more than one quantum number branch leads to the vacuum, for each quantum number branch an interaction is pushed.
     *  @param  loc     Lattice site where the interaction starts
     *  @param  opList  Vector of operators that make up the interaction. opList[i] acts on lattice site loc+i
     *  @param  lambda  Scalar factor for the interaction
     */
    void push(const std::size_t loc, const std::vector<OperatorType>& opList, const Scalar lambda = 1.0, const bool reverse=false);
    
    /**
     *  Prints information about the current state of operators and MPO auxiliar basis.
     */
    void show();
    
    /**
     *  @param loc      Lattice site
     *  @param label    Information
     */
    void save_label(const std::size_t loc, const std::string& info_label);
    
    /**
     *  @param label_in    Name to be given to this instance of MpoTerms
     */
    void set_name(const std::string& label_in) {label = label_in;}
    
    /**
     *  @return Name of this instance of MpoTerms
     */
    const std::string get_name() const {return label;}
    
    /**
     *  @return A vector of formatted strings that contain information about the MpoTerms. Zeroth entry = name.
     */
    std::vector<std::string> get_info() const;

    /**
     *  Scales all interactions by a given factor and adds a local term as offset (afterwards).
     *  @param factor   The factor to scale the interactions with
     *  @param offset   The factor all local identity operators are multiplied by
     */
    void scale(const double factor, const double offset=0.);
    
    /**
     *  @return Cast instance of MpoTerms with another scalar type
     */
    template<typename OtherScalar> MpoTerms<Symmetry, OtherScalar> cast();
    
    /**
     *  Sets the basis of a local Hilbert space. Also checks consistency with Hilbert space dimension set before.
     *  @param  loc                Lattice site
     *  @param  qPhys_in    Vector of quantum numbers that appear in the local Hilbert space
     */
    void set_qPhys(const std::size_t loc, const std::vector<qType>& qPhys_in){assert_hilbert(loc, qPhys_in.size()); GOT_QPHYS[loc] = true; qPhys[loc] = qPhys_in;}
    
    /**
     *  Finalizes the MpoTerms by cutting the row of right unity (qTot, pos_qTot) at the first and the column of left unity (Symmetry::qvacuum(), pos_qVac) at the last lattice site. Also calculates the operator bases.
     *  Afterwards, operators cannot be pushed anymore.
     *  FINALIZED is set true.
     *  @param  COMPRESS    Shall the Terms be compressed?
     *  @param  power   All W-matrices, qAux-bases and qOp-bases up to this power of the MPO are being calculated.
     *  @param  tolerance    Tolerance for compression
     */
    void finalize(const bool COMPRESS=true, const std::size_t power=1, const double tolerance=::mynumeric_limits<double>::epsilon());
        
    /**
     *  Calculates auxiliar bases, operator bases and the W matrix (if they are not up to date). Possibility to calculate higher powers of the MPO, filling the auxiliar bases, the operator bases and the W matrix for these.
     *  @param  power   All W-matrices, qAux-bases and qOp-bases up to this power of the MPO are being calculated.
     */
    void calc(const std::size_t power);
    
    /**
     *  @return True, if this instance of MpoTerms has already been finalized.
     */
    bool is_finalized() const {return FINALIZED;}
    
    /**
     *  @return  All operators that make up the W matrix
     */
    const std::vector<std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>>& get_O() const {return O;}
    
    /**
     *  @return  W matrix
     */
    const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>>& get_W() const {assert(GOT_W and "W has not been calculated!"); return W;}

    /**
     *  @return  Auxiliar bases between all lattice sites
     */
    const std::vector<Qbasis<Symmetry>>& get_qAux() const {assert(GOT_QAUX and "qAux has not been calculated!"); return qAux;}
    
    /**
     *  @return  Bases of local operators
     */
    const std::vector<std::vector<qType>>& get_qOp() const {assert(GOT_QOP and "qOp has not been calculated!"); return qOp;}
    
    /**
     *  @return  Bases of local physical Hilbert spaces
     */
    const std::vector<std::vector<qType>>& get_qPhys() const {assert(check_qPhys() and "qPhys has not been set!"); return qPhys;}
   
    /**
     *  @return  W matrix of a given power of the MPO
     *  @param  power   Chosen power of the MPO
     */
    const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>>& get_W_power(std::size_t power) const;
    
    /**
     *  @return  Auxiliar bases between all lattice sites for a given power of the MPO
     *  @param  power   Chosen power of the MPO
     */
    const std::vector<Qbasis<Symmetry>>& get_qAux_power(std::size_t power) const;
    
    /**
     *  @return  Bases of local operators for a given power of the MPO
     *  @param  power   Chosen power of the MPO
     */
    const std::vector<std::vector<qType>>& get_qOp_power(std::size_t power) const;
    
    /**
     *  @return Dimension of local Hilbert space. 0, if dimension has not been set.
     */
    std::size_t get_hilbert_dimension(const std::size_t loc) const {return hilbert_dimension[loc];}
    
    /**
     *  @return Boundary condition. False, if infinite boundary conditions (for VUMPS).
     */
    BC get_boundary_condition() const {return boundary_condition;}
    
    /**
     *  @return Vacuum MPO quantum number
     */
    const qType& get_qVac() const {return qVac;}
    
    /**
     *  @return Total MPO quantum number
     */
    const qType& get_qTot() const {return qTot;}
    
    /**
     *  @return Position of the total quantum number in its quantum number block.
     */
    const std::size_t get_pos_qTot() const {return pos_qTot;}

    /**
     *  @return Have all physical bases of local Hilbert spaces been set?
     */
    bool check_qPhys() const;
    
    /**
     *  @return Has a given power of the MPO been calculated?
     *  @param  power   Chosen power of the MPO
     */
    bool check_power(std::size_t power) const {return (power <= current_power);}

    /**
     *  @return The highest power of the MPO which is currently computed.
     */
    std::size_t maxPower() const {return current_power;}
	
    /**
     *  @return Size of the lattice (for VUMPS: unit cell)
     */
    std::size_t size() const {return N_sites;}
    
    /**
     *  A given name for the MpoTerms, such as Heisenberg
     */
    std::string label = "MPO";
    
    /**
     *  Transforms the auxiliar bases, the operator bases and the physical bases as follows:
     *  qPhys -> (qPhys - qShift) * factor
     *  qAux -> qAux * factor
     *  qOp -> qOp * factor
     *  qTot -> qTot * factor
     *  Attention: qAux, qOp and qTot are not shifted, this happens just for the physical bases
     *  @param  qShift  Quantum number by which the MPO shall be shifted
     *  @param  PRINT   Shall information on the transformation be printed?
     *  @param  factor  Additional factor to avoid fractions (Size of unit cell)
     */
    void transform_base(const qType& qShift, const bool PRINT=false, const int factor=-1);
    
    /**
     *  Precalcs data for two-site calculations
     */
    std::vector<std::vector<TwoSiteData<Symmetry,Scalar>>> calc_TwoSiteData() const;
    
    /**
     *  @return List of quantum numbers and degeneracy indices of the auxiliar basis connecting both edges of the unit cell. Ordered in such a way that a lower triangular matrix can be achieved.
     */
    std::vector<std::pair<qType,std::size_t>> base_order_IBC() const;
    
    /**
     *  Calculates the approximate amount of memory needed for this MPO
     *  @param  memunit Unit for the result
     *  @return  Memory used
     */
    double memory(MEMUNIT memunit=kB) const;
    
    /**
     *  Calculates the sparsity of the MPO or a certain power
     *  @param  power   For which power of the MPO?
     *  @param  PER_MATRIX  Sparsity related to number of matrices (?)
     *  @result Sparsity
     */
    double sparsity(const std::size_t power=1, const bool PER_MATRIX=false) const;

    /**
     * Calculates the product of two MPOs.
     *  @param  top MPO that shall be applied second
     *  @param  bottom  MPO that shall be applied first to the MPS
     *  @param  qTot    Target quantum number for the product
     *  @param  tolerance    Tolerance for compression
     *  @return Product of both MPOs.
     */
    static MpoTerms<Symmetry,Scalar> prod(const MpoTerms<Symmetry,Scalar>& top, const MpoTerms<Symmetry,Scalar>& bottom, const qType& qTot, const double tolerance=::mynumeric_limits<double>::epsilon());
    
    /**
     * Calculates the sum of two MPOs. Does not work properly for VUMPS yet.
     *  @param  top  MPO for addition
     *  @param  bottom  MPO for addition
     *  @param  tolerance    Tolerance for compression
     *  @return Sum of both MPOs.
     */
    static MpoTerms<Symmetry,Scalar> sum(const MpoTerms<Symmetry,Scalar>& top, const MpoTerms<Symmetry,Scalar>& bottom, const double tolerance=::mynumeric_limits<double>::epsilon());
    
    /**
     * Calculates relevant data for reversed order of lattice sites. Does not change the actual order for the respective instance.
     *  @return Tuple of W, qAux, qOp, qPhys
     */
    std::tuple<std::vector<std::vector<std::vector<std::vector<Biped<Symmetry,Eigen::SparseMatrix<Scalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>>>>>>, std::vector<Qbasis<Symmetry>>, std::vector<std::vector<typename Symmetry::qType>>, std::vector<std::vector<typename Symmetry::qType>>> get_reversed_data() const;
    
    /**
     *  Creates an identity MPO with bond dimension 1
     */
    void set_Identity();
    
    /**
     *  Creates a zero MPO with bond dimension 1
     */
    void set_Zero();
    
    //bool is_silent() const {return SILENT;}
    //void set_silent() {SILENT = true;}
    
    void set_verbosity(const DMRG::VERBOSITY::OPTION VERB_in);
    DMRG::VERBOSITY::OPTION get_verbosity() const {return VERB;}
    
    
    const std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>& W_at(const std::size_t loc) const {return W[loc];}
    const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>>& W_full() const {return W;}
    
    const std::vector<std::vector<qType>>& locBasis() const {return qPhys;}
    const std::vector<qType>& locBasis(const std::size_t loc) const {return qPhys[loc];}
    
    const std::vector<Qbasis<Symmetry>>& auxBasis() const {return qAux;}
    const Qbasis<Symmetry>& auxBasis(const std::size_t loc) const {return qAux[loc];}
    const Qbasis<Symmetry>& inBasis(const std::size_t loc) const {return qAux[loc];}
    const Qbasis<Symmetry>& outBasis(const std::size_t loc) const {return qAux[loc+1];}

    const std::vector<std::vector<qType>>& opBasis() const {return qOp;}
    const std::vector<qType>& opBasis(const std::size_t loc) const {return qOp[loc];}
    
    const qType& Qtarget() const {return qTot;}
    
    void setLocBasis(const std::vector<std::vector<qType>>& q) {for(std::size_t loc=0; loc<q.size(); ++loc) set_qPhys(loc, q[loc]);}
    void setLocBasis(const std::vector<qType>& q, std::size_t loc) {set_qPhys(loc, q);}
    
    const std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>& Wsq_at(const std::size_t loc) const {if(VERB != DMRG::VERBOSITY::OPTION::SILENT) lout << "Warning: method Wsq_at(loc) is deprecated" << std::endl; return W_powers[0][loc];}
    const std::vector<qType>& opBasisSq(const std::size_t loc) const {if(VERB != DMRG::VERBOSITY::OPTION::SILENT) lout << "Warning: method opBasisSq(loc) is deprecated" << std::endl; return qOp_powers[0][loc];}
    const std::vector<std::vector<qType>>& opBasisSq() const {if(VERB != DMRG::VERBOSITY::OPTION::SILENT) lout << "Warning: method opBasisSq() is deprecated" << std::endl; return qOp_powers[0];}
    const bool check_SQUARE() const {if(VERB != DMRG::VERBOSITY::OPTION::SILENT) lout << "Warning: method check_SQUARE() is deprecated" << std::endl; return (current_power>=2);}
    double sparsity(bool USE_SQUARE, bool PER_MATRIX) const {if(VERB != DMRG::VERBOSITY::OPTION::SILENT) lout << "Warning: method sparsity(bool,bool) is deprecated" << std::endl; return sparsity(2,PER_MATRIX);}


    void setQtarget(const qType& q) {assert(false and "setQtarget should not be called after the MPO has been initialized.");}
};

template<typename Symmetry> using MpoTermsXd  = MpoTerms<Symmetry,double>;
template<typename Symmetry> using MpoTermsXcd = MpoTerms<Symmetry,std::complex<double>>;

template<typename Symmetry, typename Scalar> MpoTerms<Symmetry,Scalar>::
MpoTerms(const std::size_t L, const BC boundary_condition_in, const qType& qTot_in)
: N_sites(L), boundary_condition(boundary_condition_in), qTot(qTot_in)
{
    if(N_sites>0)
    {
        initialize();
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
initialize()
{
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "Initializing an MPO with L=" << N_sites << ", Boundary condition=" << boundary_condition << " and qTot={" << Sym::format<Symmetry>(qTot) << "}" << std::endl;
    }
    else
    {
        std::stringstream ss;
        ss << before_verb_set << "Initializing an MPO with L=" << N_sites << ", Boundary condition=" << boundary_condition << " and qTot={" << Sym::format<Symmetry>(qTot) << "}" << std::endl;
        before_verb_set = ss.str();
    }
    assert(boundary_condition == BC::OPEN or qTot == qVac);
    current_power = 1;
    if(qTot == qVac)
    {
        pos_qVac = 0;
        pos_qTot = 1;
    }
    else
    {
        pos_qVac = 0;
        pos_qTot = 0;
    }
    hilbert_dimension.clear();
    hilbert_dimension.resize(N_sites, 0);
    O.clear();
    O.resize(N_sites);
    qAux.clear();
    auxdim.clear();
    auxdim.resize(N_sites+1);
    qPhys.clear();
    qPhys.resize(N_sites);
    GOT_QPHYS.clear();
    GOT_QPHYS.resize(N_sites, false);
    info.clear();
    info.resize(N_sites);

    if(boundary_condition == BC::OPEN)
    {
        increment_first_auxdim_OBC(qVac);
        increment_first_auxdim_OBC(qTot);
        increment_last_auxdim_OBC(qVac);
        increment_last_auxdim_OBC(qTot);
    }
    else
    {
        increment_auxdim(0,qVac);
        increment_auxdim(0,qTot);
    }
    for(std::size_t loc=1; loc<N_sites; ++loc)
    {
        increment_auxdim(loc,qVac);
        increment_auxdim(loc,qTot);
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
initialize(const std::size_t L, const BC boundary_condition_in, const qType& qTot_in)
{
    N_sites = L;
    boundary_condition = boundary_condition_in;
    qTot = qTot_in;
    initialize();
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
reconstruct(const std::vector<std::map<std::array<qType,2>,std::vector<std::vector<std::map<qType,OperatorType>>>>>& O_in, const std::vector<Qbasis<Symmetry>>& qAux_in, const std::vector<std::vector<qType>>& qPhys_in, const bool FINALIZED_IN, const BC boundary_condition_in, const qType& qTot_in)
{
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "Reconstructing an MPO with L=" << O_in.size() << ", Boundary condition=" << boundary_condition_in << " and qTot={" << Sym::format<Symmetry>(qTot_in) << "}" << std::endl;
    }
    else
    {
        std::stringstream ss;
        ss << before_verb_set << "Reconstructing an MPO with L=" << O_in.size() << ", Boundary condition=" << boundary_condition_in << " and qTot={" << Sym::format<Symmetry>(qTot_in) << "}" << std::endl;
        before_verb_set = ss.str();
    }
    N_sites = O_in.size();
    boundary_condition = boundary_condition_in;
    qTot = qTot_in;
    qPhys.clear();
    qPhys = qPhys_in;
    FINALIZED = FINALIZED_IN;
    
    assert(boundary_condition == BC::OPEN or qTot == qVac);
    if(qTot == qVac)
    {
        pos_qVac = 0;
        pos_qTot = 1;
    }
    else
    {
        pos_qVac = 0;
        pos_qTot = 0;
    }
    
    hilbert_dimension.clear();
    hilbert_dimension.resize(N_sites,0);
    auxdim.clear();
    qAux.clear();
    auxdim.resize(N_sites+1);
    GOT_QPHYS.clear();
    GOT_QPHYS.resize(N_sites,false);
    info.clear();
    info.resize(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        hilbert_dimension[loc] = qPhys[loc].size();
        GOT_QPHYS[loc] = true;
    }
    for(std::size_t loc=0; loc<N_sites+1; ++loc)
    {
        std::vector<qType> qs = qAux_in[loc].qs();
        auxdim[loc].clear();
        for(const auto& q : qs)
        {
            std::size_t deg = qAux_in[loc].inner_dim(q);
            auxdim[loc].insert({q,deg});
        }
    }
    O.clear();
    O = O_in;
    calc(1);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
set_verbosity(const DMRG::VERBOSITY::OPTION VERB_in)
{
    #if DEBUG_VERBOSITY > 0
    if(VERB == DMRG::VERBOSITY::OPTION::SILENT and VERB_in != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << before_verb_set;
    }
    #endif
    VERB = VERB_in;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push_reverse(const std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, const Scalar lambda)
{
    std::size_t range = opList.size();
    assert(qList.size() == range+1 and "Amount of quantum numbers does not match amount of operators!");
    std::vector<OperatorType> opList_reverse(range);
    std::vector<qType> qList_reverse(range+1);
    std::size_t loc_reverse = N_sites - loc;
    for(std::size_t n=0; n<range; ++n)
    {
        opList_reverse[n] = opList[range-n];
        qList_reverse[n] = qList[range+1-n];
    }
    qList_reverse[range] = qList[0];
    push(loc_reverse,opList_reverse,qList_reverse,lambda);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push(const std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, const Scalar lambda)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    assert((loc+opList.size() <= N_sites or boundary_condition == BC::INFINITE) and "For finite lattices operators must not exceed lattice size");
    assert(!FINALIZED and "Terms have already been finalized");
    std::size_t range = opList.size();
    assert(qList.size() == range+1 and "Amount of quantum numbers does not match amount of operators!");
    assert(qList[0] == qVac and qList[range] == qTot and "Quantum number list does not match the total MPO quantum number!");
    if(std::abs(lambda) < ::mynumeric_limits<double>::epsilon())
    {
        return;
    }
    for(std::size_t i=0; i<range; ++i)
    {
        assert_hilbert((loc+i)%N_sites, opList[i].data.rows());
        if(opList[i].data.norm() < ::mynumeric_limits<double>::epsilon())
        {
            return;
        }
    }
    if(range == 1)
    {
        #if DEBUG_VERBOSITY > 0
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Local interaction at site " << loc;
            #ifdef OPLABELS
            lout << ": " << lambda << " * " << opList[0].label;
            #endif
            lout << std::endl;
        }
        #endif
        assert(opList[0].Q == qTot and "Local operator does not match the total MPO quantum number!");
        add(loc, lambda*opList[0], qVac, qTot, pos_qVac, pos_qTot);
    }
    else
    {
        #if DEBUG_VERBOSITY > 0
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << range-1 << ".-neighbour interaction between the sites " << loc << " and " << (loc+range-1)%N_sites;
            #ifdef OPLABELS
            lout << ": " << lambda << " * " << opList[0].label << " ";
            for(std::size_t n=1; n<range; ++n)
            {
                lout << " * " << opList[n].label;
            }
            #endif
            lout << std::endl;
        }
        #endif
        std::size_t row = pos_qVac;
        std::size_t col = get_auxdim(loc+1, qList[1]);
        increment_auxdim((loc+1)%N_sites, qList[1]);
        add(loc, lambda*opList[0], qVac, qList[1], row, col);
        for(int i=1; i<range-1; ++i)
        {
            row = col;
            col = get_auxdim(loc+1+i, qList[i+1]);
            increment_auxdim((loc+1+i)%N_sites, qList[i+1]);
            add((loc+i)%N_sites, opList[i], qList[i], qList[i+1], row, col);
        }
        row = col;
        col = pos_qTot;
        add((loc+range-1)%N_sites, opList[range-1], qList[range-1], qTot, row, col);
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push(const std::size_t loc, const std::vector<OperatorType>& opList, const Scalar lambda, const bool reverse)
{
    struct qBranch
    {
        qType current;
        std::vector<qType> history;
        qBranch() {}
        qBranch(qType current_in, std::vector<qType> history_in)
        : current{current_in}, history{history_in}{}
        std::string info()
        {
            std::stringstream sout;
            for(int i=0; i<history.size(); ++i)
            {
                sout << "{" << Sym::format<Symmetry>(history[i]) << "} -> ";
            }
            sout << "{" << Sym::format<Symmetry>(current) << "}";
            return sout.str();
        }
    };

    std::size_t n = opList.size();
    std::vector<std::vector<qBranch>> Qtree(n);
    qBranch temp{opList[0].Q, {}};
    Qtree[0].push_back(temp);
    for(std::size_t m=1; m<n; ++m)
    {
        assert_hilbert((loc+m)%N_sites, opList[m].data.rows());
        for(int i=0; i<Qtree[m-1].size(); ++i)
        {
            std::vector<qType> qs = Symmetry::reduceSilent(opList[m].Q, Qtree[m-1][i].current);
            for(int j=0; j<qs.size(); ++j)
            {
                qBranch temp{qs[j], Qtree[m-1][i].history};
                temp.history.push_back(Qtree[m-1][i].current);
                Qtree[m].push_back(temp);
            }
        }
    }
    
    std::vector<qType> qList(n+1);
    qList[0] = qVac;
    int count = 0;
    for(int i=0; i<Qtree[n-1].size(); ++i)
    {
        if(Qtree[n-1][i].current == qTot)
        {
            ++count;
            for(int j=0; j<n-1; ++j)
            {
                qList[j+1] = Qtree[n-1][i].history[j];
            }
            qList[n] = qTot;
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "This branch of quantum numbers leads to the total MPO quantum number: {" << Sym::format<Symmetry>(qList[0]) << "} -> ";
                for(int j=0; j<n-1; ++j)
                {
                    lout << "{" << Sym::format<Symmetry>(qList[j+1]) << "} -> ";
                }
                lout << "{" << Sym::format<Symmetry>(qList[n]) << "}" << std::endl;
            }
            #endif
        }
    }
    assert(count == 1 and "Either no or more than one quantum number branch leads to the total MPO quantum number");
    if(reverse)
    {
        push_reverse(loc, opList, qList, lambda);
    }
    else
    {
        push(loc, opList, qList, lambda);
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
show()
{
    lout << "####################################################################################################" << std::endl;
    lout << "Name: " << label << std::endl;
    lout << "System with " << N_sites << " lattice sites and " << boundary_condition << " boundary condition, target quantum number {" << Sym::format<Symmetry>(qTot) << "}";
    if(!FINALIZED)
    {
        lout << ", not finalized yet";
    }
    lout << std::endl;
    lout << "Approximate memory usage: " << round(memory(kB),1) << " kB";
    if(GOT_W and GOT_QOP)
    {
        lout << ", sparsity: " << round(sparsity()*100,1) << "%";
    }
    lout << std::endl;
    auto [average_auxdim, maximum_auxdim] = auxdim_infos();
    lout << "Calculated bases and data:\n";
    if(GOT_QAUX) lout << "• MPO auxiliar bases (Average bond dimension: " << average_auxdim << ", maximum bond dimension: " << maximum_auxdim << ")" << std::endl;
    if(check_qPhys()) lout << "• Physical bases of local Hilbert spaces" << std::endl;
    if(GOT_QOP) lout << "• Local operator bases" << std::endl;
    if(GOT_W) lout << "• W matrix" << std::endl;
    if(current_power > 1) lout << "• All of the previous for all powers of the MPO up to " << current_power << std::endl;
    #if DEBUG_VERBOSITY > 0
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        lout << "Lattice site: " << loc << std::endl;
        lout << "\tPhysical basis of local Hilbert space (" << hilbert_dimension[loc] << " dim):\t";
        if(GOT_QPHYS[loc])
        {
            for(std::size_t n=0; n<qPhys[loc].size(); ++n)
            {
                lout << "\t{" << Sym::format<Symmetry>(qPhys[loc][n]) << "}";
            }
        }
        lout << "\n\tIncoming quantum numbers:\t";
        for(const auto& [qIn, deg] : auxdim[loc])
        {
            lout << "\t({" << Sym::format<Symmetry>(qIn) << "} [#=" << deg << "])";
        }
        lout << "\n\tOutgoing quantum numbers:\t";
        for(const auto& [qOut, deg] : auxdim[loc+1])
        {
            lout << "\t({" << Sym::format<Symmetry>(qOut) << "} [#=" << deg << "])";
        }
        lout << std::endl;
        #if DEBUG_VERBOSITY > 2
        lout << "\tOperators:" << std::endl;
        for(const auto& [qs, ops] : O[loc])
        {
            std::size_t rows = get_auxdim(loc, std::get<0>(qs));
            std::size_t cols = get_auxdim(loc+1, std::get<1>(qs));
            lout << "\t\t{" << Sym::format<Symmetry>(std::get<0>(qs)) << "}->{" << Sym::format<Symmetry>(std::get<1>(qs)) << "}:" << std::endl;
            for(std::size_t row=0; row<rows; ++row)
            {
                for(std::size_t col=0; col<cols; ++col)
                {
                    if(ops[row][col].size() > 0)
                    {
                        lout << "\t\t\tPosition [" << row << "|" << col << "]:";
                        for(const auto& [Q,op] : ops[row][col])
                        {
                            #ifdef OPLABELS
                            lout << "\t" << op.label << " ({" << Sym::format<Symmetry>(Q) << "}) ";
                            #else
                            lout << "\t ({" << Sym::format<Symmetry>(Q) << "}) ";
                            #endif
                        }
                        lout << std::endl;
                    }
                }
            }
        }
        #endif
    }
    #endif
    lout << "####################################################################################################" << std::endl;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
calc(const std::size_t power)
{
    if(!GOT_QAUX)
    {
        calc_qAux();
    }
    if(!GOT_QOP)
    {
        calc_qOp();
    }
    if(!GOT_W)
    {
        calc_W();
    }
    W_powers.clear();
    qAux_powers.clear();
    qOp_powers.clear();
    W_powers.resize(power-1);
    qAux_powers.resize(power-1);
    qOp_powers.resize(power-1);
    MpoTerms<Symmetry,Scalar> current = *this;
    for(std::size_t n=2; n<=power; ++n)
    {
        std::vector<qType> qTots = Symmetry::reduceSilent(current.get_qTot(),qTot);
        assert(qTots.size() == 1 and "Target quantum number has to be unique for computing higher powers of the MPO.");
        current = MpoTerms<Symmetry,Scalar>::prod(current,*this,qTots[0]);
        current.calc(1);
        W_powers[n-2] = current.get_W();
        qAux_powers[n-2] = current.get_qAux();
        qOp_powers[n-2] = current.get_qOp();
        #if DEBUG_VERBOSITY > 0
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << n << ". power of the MPO:" << std::endl;
            current.show();
        }
        #endif
    }
    current_power = power;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
calc_W()
{
    assert(GOT_QOP and "qOp is needed for calculation of W matrix!");
    W.resize(N_sites);

    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t hdim = hilbert_dimension[loc];
        std::size_t odim = qOp[loc].size();
        W[loc].resize(hdim);
        for(std::size_t m=0; m<hdim; ++m)
        {
            W[loc][m].resize(hdim);
            for(std::size_t n=0; n<hdim; ++n)
            {
                W[loc][m][n].resize(odim);
                for(std::size_t t=0; t<odim; ++t)
                {
                    Biped<Symmetry, MatrixType> bip;
                    for(const auto& [qIn, rows] : auxdim[loc])
                    {
                        for(const auto& [qOut, cols] : auxdim[loc+1])
                        {
                            auto it = O[loc].find({qIn, qOut});
                            assert(it != O[loc].end());
                            bool found_match = false;
                            MatrixType mat(rows, cols);
                            mat.setZero();
                            for(std::size_t row=0; row<rows; ++row)
                            {
                                for(std::size_t col=0; col<cols; ++col)
                                {
                                    for(const auto& [Q,op] : (it->second)[row][col])
                                    {
                                        if(Q == qOp[loc][t])
                                        {
                                            mat.coeffRef(row, col) = op.data.coeff(m,n);
                                            if(std::abs(mat.coeffRef(row, col)) > ::mynumeric_limits<double>::epsilon())
                                            {
                                                found_match = true;
                                            }
                                        }
                                    }
                                }
                            }
                            if(found_match and mat.norm() > ::mynumeric_limits<double>::epsilon())
                            {
                                bip.push_back(qIn, qOut, mat);
                            }
                        }
                    }
                    W[loc][m][n][t] = bip;
                }
            }
        }
    }
    GOT_W = true;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
calc_qOp()
{
    qOp.resize(N_sites);
    
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::set<typename Symmetry::qType> qOp_set;
        for(const auto& [qs, ops] : O[loc])
        {
            std::size_t rows = get_auxdim(loc, std::get<0>(qs));
            std::size_t cols = get_auxdim(loc+1, std::get<1>(qs));
            for(std::size_t row=0; row<rows; ++row)
            {
                for(std::size_t col=0; col<cols; ++col)
                {
                    for(const auto& [Q,op] : ops[row][col])
                    {
                        qOp_set.insert(Q);
                    }
                }
            }
        }
        qOp[loc].resize(qOp_set.size());
        copy(qOp_set.begin(), qOp_set.end(), qOp[loc].begin());
    }
    GOT_QOP = true;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
calc_qAux()
{
    qAux.resize(N_sites+1);
    for(std::size_t loc=0; loc<N_sites+1; ++loc)
    {
        qAux[loc].clear();
        for(const auto& [q,deg] : auxdim[loc])
        {
            qAux[loc].push_back(q,deg);
        }
    }
    GOT_QAUX = true;

}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
finalize(const bool COMPRESS, const std::size_t power, const double tolerance)
{
    assert(!FINALIZED);
    FINALIZED = true;
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        if(hilbert_dimension[loc] == 0) hilbert_dimension[loc] = 1;
        OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc], hilbert_dimension[loc]).sparseView(),qVac);
        #ifdef OPLABELS
        Id.label = "id";
        #endif
        add(loc, Id, qVac, qVac, pos_qVac, pos_qVac);
        add(loc, Id, qTot, qTot, pos_qTot, pos_qTot);
    }
    if(boundary_condition == BC::OPEN)
    {
        delete_row(0, qTot, pos_qTot,false);
        bool SAMESITE = false;
        if(N_sites == 1)
        {
            SAMESITE = true;
        }
        delete_col(N_sites-1, qVac, pos_qVac,SAMESITE);
        
        decrement_first_auxdim_OBC(qTot);
        decrement_last_auxdim_OBC(qVac);
    }
    #ifndef OPLABELS
    clear_opLabels();
    #endif
    if(COMPRESS)
    {
        compress(tolerance);
    }
    calc(power);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
compress(const double tolerance)
{
    assert(FINALIZED and "Terms need to be finalized before compression.");
    #if DEBUG_VERBOSITY > 0
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "Starting compression of the MPO" << std::endl;
    }
    #endif
    Stopwatch<> watch;
    auto [average_auxdim_initial,maximum_auxdim_initial] = auxdim_infos();
    std::vector<bool> updated_bond(N_sites,true);
    std::size_t minimum_bond = 0;
    if(boundary_condition == BC::OPEN)
    {
        minimum_bond = 1;
        updated_bond[0] = false;
    }
    auto any_update = [minimum_bond](const std::vector<bool>& bonds)->bool
    {
        for(std::size_t loc=minimum_bond; loc<bonds.size(); ++loc)
        {
            if(bonds[loc])
            {
                return true;
            }
        }
        return false;
    };
    std::size_t counter = 0;
    while(any_update(updated_bond))
    {
        bool change = false;
        for(std::size_t loc=minimum_bond; loc<N_sites; ++loc)
        {
            if(updated_bond[(loc+N_sites-1)%N_sites] or updated_bond[loc])
            {
                for(auto& [q,deg] : auxdim[loc])
                {
                    if(eliminate_linearlyDependent_cols((loc+N_sites-1)%N_sites,q,tolerance))
                    {
                        counter++;
                        updated_bond[loc] = true;
                        change = true;
                        #if DEBUG_VERBOSITY > 2
                        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                        {
                            lout << "Current MPO:" << std::endl;
                            show();
                        }
                        #endif
                        break;
                    }
                }
            }
            if(!change and (updated_bond[loc] or updated_bond[(loc+1)%N_sites]))
            {
                for(auto& [q,deg] : auxdim[loc])
                {
                    if(eliminate_linearlyDependent_rows(loc,q,tolerance))
                    {
                        counter++;
                        updated_bond[loc] = true;
                        change = true;
                        #if DEBUG_VERBOSITY > 2
                        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                        {
                            lout << "Current MPO:" << std::endl;
                            show();
                        }
                        #endif
                        break;
                    }
                }
            }
            
            if(change)
            {
                break;
            }
            else if(updated_bond[loc])
            {
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "Bond connecting lattice sites " << (loc+N_sites-1)%N_sites << " and " << loc << ": No linear dependence detected" << std::endl;
                }
                #endif
                updated_bond[loc] = false;
            }
        }
    }
    auto [average_auxdim_final,maximum_auxdim_final] = auxdim_infos();
    int compr_rate = (int)std::round(100*(1-average_auxdim_final/average_auxdim_initial));
    calc(1);
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "Compression finished  |  " << watch.info("Time") << "  |  Steps: " << counter << "  |  Rate: " << compr_rate << "% (" << average_auxdim_initial << "->" << average_auxdim_final << ")" << std::endl;
        #if DEBUG_VERBOSITY > 1
        lout << "Compressed MPO:" << std::endl;
        show();
        #endif
    }
    else
    {
        std::stringstream ss;
        ss << before_verb_set << "Compression finished  |  " << watch.info("Time") << "  |  Steps: " << counter << "  |  Rate: " << compr_rate << "% (" << average_auxdim_initial << "->" << average_auxdim_final << ")" << std::endl;
        before_verb_set = ss.str();
    }
}

template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
eliminate_linearlyDependent_rows(const std::size_t loc, const qType& qIn, const double tolerance)
{
    bool SAMESITE = false;
    if(N_sites == 1 and boundary_condition == BC::INFINITE)
    {
        SAMESITE = true;
    }
    assert(hilbert_dimension[loc] > 0);
    std::size_t cols_eff = 0;
    std::size_t hd = hilbert_dimension[loc];
    
    std::map<qType,std::vector<std::vector<qType>>> opQs;
    for(const auto& [qs,ops] : O[loc])
    {
        if(std::get<0>(qs) != qIn)
        {
            continue;
        }
        std::vector<std::vector<qType>> opQs_fixed_qOut(ops[0].size());
        for(std::size_t col=0; col<ops[0].size(); ++col)
        {
            std::unordered_set<qType> unique_control;
            for(std::size_t row=0; row<ops.size(); ++row)
            {

                for(const auto& [Q,op] : ops[row][col])
                {
                    unique_control.insert(Q);
                }
            }
            cols_eff += (unique_control.size() * hd * hd);
            opQs_fixed_qOut[col].resize(unique_control.size());
            std::copy(unique_control.begin(), unique_control.end(), opQs_fixed_qOut[col].begin());
        }
        opQs.insert({std::get<1>(qs), opQs_fixed_qOut});
    }

    std::size_t rows = get_auxdim(loc, qIn);
    std::size_t rows_eff = rows;
    std::size_t skipcount = 0;
    if(boundary_condition == BC::INFINITE and qIn == qVac)
    {
        rows_eff = rows_eff-2;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows_eff, cols_eff);
    for(std::size_t row=0; row<rows; ++row)
    {
        std::size_t count = 0;
        if((boundary_condition == BC::INFINITE and qIn == qVac and (row == pos_qVac or row == pos_qTot)))
        {
            skipcount++;
            continue;
        }
        bool zero_row = true;
        for(const auto& [qOut, deg] : auxdim[loc+1])
        {
            auto it_ops = O[loc].find({qIn,qOut});
            assert(it_ops != O[loc].end());
            auto it_qs = opQs.find(qOut);
            assert(it_qs != opQs.end());
            for(std::size_t col=0; col<deg; ++col)
            {
                std::map<qType,OperatorType>& ops = (it_ops->second)[row][col];
                for(std::size_t n=0; n<(it_qs->second)[col].size(); ++n)
                {
                    qType& opQ = (it_qs->second)[col][n];
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> opMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(hd,hd);
                    auto it_op = ops.find(opQ);
                    if(it_op != ops.end())
                    {
                        opMat = (it_op->second).data;
                        if(opMat.norm() > tolerance)
                        {
                            zero_row = false;
                        }
                    }
                    for(std::size_t i=0; i<hd; ++i)
                    {
                        for(std::size_t j=0; j<hd; ++j)
                        {
                            mat(row-skipcount, count) = opMat(i,j);
                            count++;
                        }
                    }
                }
            }
        }
        if(zero_row)
        {
            if(rows > 1)
            {
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "O[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->...): Row " << row << " is a zero row, but not the last row in these blocks. Thus it is deleted." << std::endl;
                }
                #endif
                delete_row(loc, qIn, row, false);
                delete_col((loc+N_sites-1)%N_sites, qIn, row, SAMESITE);
            }
            #if DEBUG_VERBOSITY > 1
            else
            {
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "O[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->...): Row " << row << " is a zero row and the last row in this block. Thus the quantum number is removed from the auxiliar basis to delete the row and the corresponding column at the previous lattice site." << std::endl;
                }
            }
            #endif
            decrement_auxdim(loc,qIn);
            return true;
        }
    }
    if(rows > 3 or (rows > 1 and (boundary_condition == BC::OPEN or qIn != qVac)))
    {
        std::size_t rowskipcount = 0;
        for(std::size_t row_to_delete=0; row_to_delete<rows; ++row_to_delete)
        {
            if((boundary_condition == BC::INFINITE and qIn == qVac and (row_to_delete == pos_qVac or row_to_delete == pos_qTot)))
            {
                rowskipcount++;
                continue;
            }
            std::size_t row_to_delete_eff = row_to_delete - rowskipcount;
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> tempmat(cols_eff, rows_eff-1);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tempvec(cols_eff);
            tempvec = mat.block(row_to_delete_eff, 0, 1, cols_eff).transpose();
            tempmat.block(0,0, cols_eff, row_to_delete_eff) = mat.block(0,0,row_to_delete_eff,cols_eff).transpose();
            tempmat.block(0,row_to_delete_eff, cols_eff, rows_eff-row_to_delete_eff-1) = mat.block(row_to_delete_eff+1,0,rows_eff-row_to_delete_eff-1,cols_eff).transpose();

            
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec = tempmat.colPivHouseholderQr().solve(tempvec);
            
            if((tempmat*vec - tempvec).norm() < tolerance)
            {
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "O[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->...): Row " << row_to_delete << " can be written as linear combination of other rows" << std::endl;
                }
                #endif
                delete_row(loc, qIn, row_to_delete, false);
                std::map<qType, std::vector<std::map<qType,OperatorType>>> deleted_col = delete_col((loc+N_sites-1)%N_sites, qIn, row_to_delete, SAMESITE);
                decrement_auxdim(loc,qIn);
                std::size_t cols = get_auxdim(loc,qIn);
                std::size_t colskipcount = 0;
                for(std::size_t col=0; col<cols; ++col)
                {
                    if((boundary_condition == BC::INFINITE and qIn == qVac and (col == pos_qVac or col == pos_qTot)))
                    {
                        colskipcount++;
                        continue;
                    }
                    if(std::abs(vec(col-colskipcount)) > tolerance)
                    {
                        add_to_col((loc+N_sites-1)%N_sites, qIn, col, deleted_col, vec(col-colskipcount));
                    }
                }
                return true;
            }
        }
    }
    return false;
}

template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
eliminate_linearlyDependent_cols(const std::size_t loc, const qType& qOut, const double tolerance)
{
    bool SAMESITE = false;
    if(N_sites == 1 and boundary_condition == BC::INFINITE)
    {
        SAMESITE = true;
    }
    assert(hilbert_dimension[loc] > 0);
    std::size_t rows_eff = 0;
    std::size_t hd = hilbert_dimension[loc];
    
    std::map<qType,std::vector<std::vector<qType>>> opQs;
    for(const auto& [qs,ops] : O[loc])
    {
        if(std::get<1>(qs) != qOut)
        {
            continue;
        }
        std::vector<std::vector<qType>> opQs_fixed_qIn(ops.size());
        for(std::size_t row=0; row<ops.size(); ++row)
        {
            std::unordered_set<qType> unique_control;
            for(std::size_t col=0; col<ops[row].size(); ++col)
            {
                for(const auto& [Q,op] : ops[row][col])
                {
                    unique_control.insert(Q);
                }
            }
            rows_eff += (unique_control.size() * hd * hd);
            opQs_fixed_qIn[row].resize(unique_control.size());
            std::copy(unique_control.begin(), unique_control.end(), opQs_fixed_qIn[row].begin());
        }
        opQs.insert({std::get<0>(qs), opQs_fixed_qIn});
    }
    
    std::size_t cols = get_auxdim(loc+1, qOut);
    std::size_t cols_eff = cols;
    std::size_t skipcount = 0;
    if(boundary_condition == BC::INFINITE and qOut == qVac)
    {
        cols_eff = cols_eff-2;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows_eff, cols_eff);
    for(std::size_t col=0; col<cols; ++col)
    {
        std::size_t count = 0;
        if((boundary_condition == BC::INFINITE and qOut == qVac and (col == pos_qVac or col == pos_qTot)))
        {
            skipcount++;
            continue;
        }
        bool zero_col = true;
        for(const auto& [qIn, deg] : auxdim[loc])
        {
            auto it_ops = O[loc].find({qIn,qOut});
            assert(it_ops != O[loc].end());
            auto it_qs = opQs.find(qIn);
            assert(it_qs != opQs.end());
            for(std::size_t row=0; row<deg; ++row)
            {
                std::map<qType,OperatorType>& ops = (it_ops->second)[row][col];
                for(std::size_t n=0; n<(it_qs->second)[row].size(); ++n)
                {
                    qType& opQ = (it_qs->second)[row][n];
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> opMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(hd,hd);
                    auto it_op = ops.find(opQ);
                    if(it_op != ops.end())
                    {
                        opMat = (it_op->second).data;
                        if(opMat.norm() > tolerance)
                        {
                            zero_col = false;
                        }
                    }
                    for(std::size_t i=0; i<hd; ++i)
                    {
                        for(std::size_t j=0; j<hd; ++j)
                        {
                            mat(count, col-skipcount) = opMat(i,j);
                            count++;
                        }
                    }
                }
            }
        }
        if(zero_col)
        {
            if(cols > 1)
            {
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "O[" << loc << "](...->{" << Sym::format<Symmetry>(qOut) << "}): Column " << col << " is a zero column, but not the last column in these blocks. Thus it is deleted." << std::endl;
                }
                #endif
                delete_col(loc, qOut, col, false);
                delete_row((loc+1)%N_sites, qOut, col, SAMESITE);
            }
            #if DEBUG_VERBOSITY > 1
            else
            {
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "O[" << loc << "](...->{" << Sym::format<Symmetry>(qOut) << "}): Column " << col << " is a zero column and the last column in this block. Thus the quantum number is removed from the auxiliar basis to delete the column and the corresponding row at the next lattice site." << std::endl;
                }
            }
            #endif
            decrement_auxdim((loc+1)%N_sites,qOut);
            return true;
        }
    }
    if(cols > 3 or ((boundary_condition == BC::OPEN or qOut != qVac) and cols>1))
    {
        std::size_t colskipcount = 0;
        for(std::size_t col_to_delete=0; col_to_delete<cols; ++col_to_delete)
        {
            if((boundary_condition == BC::INFINITE and qOut == qVac and (col_to_delete == pos_qVac or col_to_delete == pos_qTot)))

            {
                colskipcount++;
                continue;
            }
            std::size_t col_to_delete_eff = col_to_delete - colskipcount;
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> tempmat(rows_eff, cols_eff-1);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tempvec(rows_eff);
            tempvec = mat.block(0, col_to_delete_eff, rows_eff, 1);
            tempmat.block(0,0, rows_eff, col_to_delete_eff) = mat.block(0, 0, rows_eff, col_to_delete_eff);
            tempmat.block(0, col_to_delete_eff, rows_eff, cols_eff-col_to_delete_eff-1) = mat.block(0, col_to_delete_eff+1, rows_eff, cols_eff-col_to_delete_eff-1);

            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec = tempmat.colPivHouseholderQr().solve(tempvec);
            
            if((tempmat*vec - tempvec).norm() < tolerance)
            {
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "O[" << loc << "](...->{" << Sym::format<Symmetry>(qOut) << "}): Column " << col_to_delete << " can be written as linear combination of other columns" << std::endl;
                }
                #endif
                delete_col(loc, qOut, col_to_delete,false);
                std::map<qType, std::vector<std::map<qType,OperatorType>>> deleted_row = delete_row((loc+1)%N_sites, qOut, col_to_delete, SAMESITE);
                decrement_auxdim((loc+1)%N_sites,qOut);
                std::size_t rows = get_auxdim(loc+1,qOut);
                std::size_t rowskipcount = 0;
                for(std::size_t row=0; row<rows; ++row)
                {
                    if((boundary_condition == BC::INFINITE and qOut == qVac and (row == pos_qVac or row == pos_qTot)))

                    {
                        rowskipcount++;
                        continue;
                    }
                    if(std::abs(vec(row-rowskipcount)) > tolerance)
                    {
                        add_to_row((loc+1)%N_sites, qOut, row, deleted_row, vec(row-rowskipcount));
                    }
                }
                return true;
            }
        }
    }
    return false;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
add_to_row(const std::size_t loc, const qType& qIn, const std::size_t row, const std::map<qType,std::vector<std::map<qType,OperatorType>>>& ops, const Scalar factor)
{
    #if DEBUG_VERBOSITY > 2
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tO[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->...): Add another row scaled by factor " << factor << " to row " << row << std::endl;
    }
    #endif
    std::size_t rows = get_auxdim(loc, qIn);
    assert(row < rows and "Trying to add to a nonexisting row");
    got_update();
    for(const auto& [qOut, deg] : auxdim[loc+1])
    {
        auto it = O[loc].find({qIn, qOut});
        auto it2 = ops.find(qOut);
        assert(it != O[loc].end());
        assert(it2 != ops.end());
        assert((it2->second).size() == deg and "Adding rows of different size");
        for(std::size_t col=0; col<deg; ++col)
        {
            std::map<qType,OperatorType>& existing_ops = (it->second)[row][col];
            const std::map<qType,OperatorType>& ops_new = (it2->second)[col];
            for(const auto& [Q_new,op_new] : ops_new)
            {
                auto it3 = existing_ops.find(Q_new);
                if(it3 != existing_ops.end())
                {
                    (it3->second) += factor*op_new;
                }
                else
                {
                    existing_ops.insert({Q_new,factor*op_new});
                }
            }
        }
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
add_to_col(const std::size_t loc, const qType& qOut, const std::size_t col, const std::map<qType, std::vector<std::map<qType,OperatorType>>>& ops, const Scalar factor)
{
    #if DEBUG_VERBOSITY > 2
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tO[" << loc << "](...->{" << Sym::format<Symmetry>(qOut) << "}): Add another column scaled by factor " << factor << " to column " << col << std::endl;
    }
    #endif
    std::size_t cols = get_auxdim(loc+1, qOut);
    assert(col < cols and "Trying to add to a nonexisting col");
    got_update();
    for(const auto& [qIn, deg] : auxdim[loc])
    {
        auto it = O[loc].find({qIn, qOut});
        auto it2 = ops.find(qIn);
        assert(it != O[loc].end());
        assert(it2 != ops.end());
        assert((it2->second).size() == deg and "Adding columns of different size");
        for(std::size_t row=0; row<deg; ++row)
        {
            std::map<qType,OperatorType>& existing_ops = (it->second)[row][col];
            const std::map<qType,OperatorType>& ops_new = (it2->second)[row];
            for(const auto& [Q_new,op_new] : ops_new)
            {
                auto it3 = existing_ops.find(Q_new);
                if(it3 != existing_ops.end())
                {
                    (it3->second) += factor*op_new;
                }
                else
                {
                    existing_ops.insert({Q_new,factor*op_new});
                }
            }
        }
    }
}

template<typename Symmetry, typename Scalar> std::map<typename Symmetry::qType,std::vector<std::map<typename Symmetry::qType,SiteOperator<Symmetry,Scalar>>>> MpoTerms<Symmetry,Scalar>::
delete_row(const std::size_t loc, const qType& qIn, const std::size_t row_to_delete, const bool SAMESITE)
{
    #if DEBUG_VERBOSITY > 2
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tO[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->...): Delete row " << row_to_delete << std::endl;
        if(SAMESITE)
        {
            lout << "\tPaying attention since a column has been deleted at the same site before" << std::endl;
        }
    }
    #endif
    std::map<qType, std::vector<std::map<qType,OperatorType>>> deleted_row;
    std::size_t rows = get_auxdim(loc, qIn);
    assert(row_to_delete < rows and "Trying to delete a nonexisting row");
    got_update();
    std::map<qType,OperatorType> empty_op_map;
    for(const auto& [qOut, deg] : auxdim[loc+1])
    {
        auto it = O[loc].find({qIn, qOut});
        std::vector<std::map<qType,OperatorType>> temp;
        assert(it != O[loc].end());
        std::size_t skip = 0;
        if(SAMESITE and qOut == qIn)
        {
            skip = 1;
        }
        for(std::size_t col=0; col<deg-skip; ++col)
        {
            temp.push_back((it->second)[row_to_delete][col]);
            for(std::size_t row=row_to_delete; row<rows-1; ++row)
            {
                (it->second)[row][col] = (it->second)[row+1][col];
            }
            (it->second)[rows-1][col] = empty_op_map;
        }
        deleted_row.insert({qOut, temp});
    }
    return deleted_row;
}

template<typename Symmetry, typename Scalar> std::map<typename Symmetry::qType,std::vector<std::map<typename Symmetry::qType,SiteOperator<Symmetry,Scalar>>>> MpoTerms<Symmetry,Scalar>::
delete_col(const std::size_t loc, const qType& qOut, const std::size_t col_to_delete, const bool SAMESITE)
{
    #if DEBUG_VERBOSITY > 2
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tO[" << loc << "](...->{" << Sym::format<Symmetry>(qOut) << "}): Delete column " << col_to_delete << std::endl;
        if(SAMESITE)
        {
            lout << "\tPaying attention since a row has been deleted at the same site before" << std::endl;
        }
    }
    #endif
    std::map<qType, std::vector<std::map<qType,OperatorType>>> deleted_col;
    std::size_t cols = get_auxdim(loc+1, qOut);
    assert(col_to_delete < cols and "Trying to delete a nonexisting column");
    got_update();
    std::map<qType,OperatorType> empty_op_map;
    for(const auto& [qIn, deg] : auxdim[loc])
    {
        auto it = O[loc].find({qIn, qOut});
        std::vector<std::map<qType,OperatorType>> temp;
        assert(it != O[loc].end());
        std::size_t skip = 0;
        if(SAMESITE and qIn == qOut)
        {
            skip = 1;
        }
        for(std::size_t row=0; row<deg-skip; ++row)
        {
            temp.push_back((it->second)[row][col_to_delete]);
            for(std::size_t col=col_to_delete; col<cols-1; ++col)
            {
                (it->second)[row][col] = (it->second)[row][col+1];
            }
            (it->second)[row][cols-1] = empty_op_map;
        }
        deleted_col.insert({qIn, temp});
    }
    return deleted_col;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
add(const std::size_t loc, const OperatorType& op, const qType& qIn, const qType& qOut, const std::size_t leftIndex, const std::size_t rightIndex)
{
    #if DEBUG_VERBOSITY > 2
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        #ifdef OPLABELS
        lout << "O[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "})[" << leftIndex << "|" << rightIndex << "]: Add " << op.label << " ({" << Sym::format<Symmetry>(op.Q) << "})" << std::endl;
        #else
        lout << "O[" << loc << "]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "})[" << leftIndex << "|" << rightIndex << "]: Add Operator ({" << Sym::format<Symmetry>(op.Q) << "})" << std::endl;
        #endif
    }
    #endif
    std::size_t rows = get_auxdim(loc, qIn);
    std::size_t cols = get_auxdim(loc+1, qOut);
    auto it = O[loc].find({qIn, qOut});
    assert(leftIndex <= rows and "Index out of bounds");
    assert(rightIndex <= cols and "Index out of bounds");
    assert(it != O[loc].end() and "Quantum numbers not available");
    got_update();
    std::map<qType,OperatorType>& existing_ops = (it->second)[leftIndex][rightIndex];
    auto it2 = existing_ops.find(op.Q);
    if(it2 != existing_ops.end())
    {
        (it2->second) += op;
    }
    else
    {
        existing_ops.insert({op.Q,op});
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
increment_auxdim(const std::size_t loc, const qType& q)
{
    #if DEBUG_VERBOSITY > 1
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tqAux[" << (loc+N_sites-1)%N_sites << "->" << loc << "]({" << Sym::format<Symmetry>(q) << "}): ";
    }
    #endif
    got_update();
    std::map<qType,OperatorType> empty_op_map;
    if(loc != 0)
    {
        assert(loc < N_sites);
        auto it = auxdim[loc].find(q);
        if(it == auxdim[loc].end())
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "New counter started" << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[loc-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Block created with one column of operators (number of rows: " << dimPrev << ")" << std::endl;
                }
                #endif
                std::vector<std::map<qType,OperatorType>> temp_col(1,empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(dimPrev,temp_col);
                O[loc-1].insert({{qPrev,q},temp_ops});
            }
            for(const auto& [qNext,dimNext] : auxdim[loc+1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc << "]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Block created with one row of operators (number of columns: " << dimNext << ")" << std::endl;
                }
                #endif
                std::vector<std::map<qType,OperatorType>> temp_col(dimNext,empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(1,temp_col);
                O[loc].insert({{q,qNext},temp_ops});
            }
            auxdim[loc].insert({q,1});
        }
        else
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "Counter incremented to " << (it->second)+1 << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[loc-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Create a new column of operators (number of rows: " << dimPrev << ")" << std::endl;
                }
                #endif
                auto it_prev = O[loc-1].find({qPrev,q});
                assert(it_prev != O[loc-1].end());
                for(std::size_t row=0; row<dimPrev; ++row)
                {
                    (it_prev->second)[row].push_back(empty_op_map);
                }
            }
            for(const auto& [qNext,dimNext] : auxdim[loc+1])
            {
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc << "]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Create a new row of operators (number of columns: " << dimNext << ")" << std::endl;
                }
                #endif
                auto it_next = O[loc].find({q,qNext});
                assert(it_next != O[loc].end());
                std::vector<std::map<qType,OperatorType>> temp_row(dimNext, empty_op_map);
                (it_next->second).push_back(temp_row);
            }
            (it->second)++;
        }
    }
    else
    {
        assert(boundary_condition == BC::INFINITE);
        auto it = auxdim[0].find(q);
        if(it == auxdim[0].end())
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "New counter started" << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[N_sites-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Block created with one column of operators (number of rows: " << dimPrev << ")" << std::endl;
                }
                #endif
                std::vector<std::map<qType,OperatorType>> temp_col(1,empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(dimPrev,temp_col);
                O[N_sites-1].insert({{qPrev,q},temp_ops});
            }
            for(const auto& [qNext,dimNext] : auxdim[1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[0]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Block created with one row of operators (number of columns: " << dimNext << ")" << std::endl;
                }
                #endif
                std::vector<std::map<qType,OperatorType>> temp_col(dimNext,empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(1,temp_col);
                O[0].insert({{q,qNext},temp_ops});
            }
            auxdim[0].insert({q,1});
            auxdim[N_sites].insert({q,1});
        }
        else
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "Counter incremented to " << (it->second)+1 << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[N_sites-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Create a new column of operators (number of rows: " << dimPrev << ")" << std::endl;
                }
                #endif
                auto it_prev = O[N_sites-1].find({qPrev,q});
                assert(it_prev != O[N_sites-1].end());
                for(std::size_t row=0; row<dimPrev; ++row)
                {
                    (it_prev->second)[row].push_back(empty_op_map);
                }
            }
            for(const auto& [qNext,dimNext] : auxdim[1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[0]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Create a new row of operators (number of columns: " << dimNext << ")" << std::endl;
                }
                #endif
                auto it_next = O[0].find({q,qNext});
                assert(it_next != O[0].end());
                std::vector<std::map<qType,OperatorType>> temp_row(dimNext, empty_op_map);
                (it_next->second).push_back(temp_row);
            }
            (it->second)++;
            auto it_edge = auxdim[N_sites].find(q);
            assert(it_edge != auxdim[N_sites].end());
            (it_edge->second)++;
        }
        if(N_sites == 1)
        {
            auto it = O[0].find({q,q});
            if(it == O[0].end())
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[0]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(q) << "}): Create a new block of operators with 1 row and 1 column" << std::endl;
                }
                #endif
                std::vector<std::map<qType,OperatorType>> temp_col(1,empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(1,temp_col);
                O[0].insert({{q,q},temp_ops});
            }
            else
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[0]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(q) << "}): Add the corner operator" << std::endl;
                }
                #endif
                (it->second)[(it->second).size()-1].push_back(empty_op_map);
            }
        }
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
increment_first_auxdim_OBC(const qType& qIn)
{
    assert(boundary_condition == BC::OPEN);
    #if DEBUG_VERBOSITY > 1
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tqAux[left edge->0]({" << Sym::format<Symmetry>(qIn) << "}): ";
    }
    #endif
    got_update();
    std::map<qType,OperatorType> empty_op_map;
    auto it = auxdim[0].find(qIn);
    if(it == auxdim[0].end())
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "New counter started" << std::endl;
        }
        #endif
        for(const auto& [qOut,deg] : auxdim[1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[0]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Block created with one row of operators (number of columns: " << deg << ")" << std::endl;
            }
            #endif
            std::vector<std::map<qType,OperatorType>> temp_col(deg,empty_op_map);
            std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(1,temp_col);
            O[0].insert({{qIn,qOut},temp_ops});
        }
        auxdim[0].insert({qIn,1});
    }
    else
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Counter incremented to " << (it->second)+1 << std::endl;
        }
        #endif
        for(const auto& [qOut,deg] : auxdim[1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[0]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Create a new row of operators (number of columns: " << deg << ")" << std::endl;
            }
            #endif
            auto it_ops = O[0].find({qIn,qOut});
            assert(it_ops != O[0].end());
            std::vector<std::map<qType,OperatorType>> temp_row(deg, empty_op_map);
            (it_ops->second).push_back(temp_row);
        }
        (it->second)++;
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
increment_last_auxdim_OBC(const qType& qOut)
{
    assert(boundary_condition == BC::OPEN);
    #if DEBUG_VERBOSITY > 1
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tqAux[" << N_sites-1 << "->right edge]({" << Sym::format<Symmetry>(qOut) << "}): ";
    }
    #endif
    got_update();
    std::map<qType,OperatorType> empty_op_map;
    auto it = auxdim[N_sites].find(qOut);
    if(it == auxdim[N_sites].end())
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "New counter started" << std::endl;
        }
        #endif
        for(const auto& [qIn,deg] : auxdim[N_sites-1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Block created with one column of operators (number of rows: " << deg << ")" << std::endl;
            }
            #endif
            std::vector<std::map<qType,OperatorType>> temp_col(1,empty_op_map);
            std::vector<std::vector<std::map<qType,OperatorType>>> temp_ops(deg,temp_col);
            O[N_sites-1].insert({{qIn,qOut},temp_ops});
        }
        auxdim[N_sites].insert({qOut,1});
    }
    else
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Counter incremented to " << (it->second)+1 << std::endl;
        }
        #endif
        for(const auto& [qIn,deg] : auxdim[N_sites-1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Create a new column of operators (number of rows: " << deg << ")" << std::endl;
            }
            #endif
            auto it_ops = O[N_sites-1].find({qIn,qOut});
            assert(it_ops != O[N_sites-1].end());
            for(std::size_t row=0; row<deg; ++row)
            {
                (it_ops->second)[row].push_back(empty_op_map);
            }
        }
        (it->second)++;
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
decrement_auxdim(const std::size_t loc, const qType& q)
{
    #if DEBUG_VERBOSITY > 1
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tqAux[" << (loc+N_sites-1)%N_sites << "->" << loc << "]({" << Sym::format<Symmetry>(q) << "}): ";
    }
    #endif
    got_update();
    auto it = auxdim[loc].find(q);
    assert(it != auxdim[loc].end());
    if(loc != 0)
    {
        assert(loc < N_sites);
        if(it->second == 1)
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "Quantum number deleted" << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[loc-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Block deleted" << std::endl;
                }
                #endif
                O[loc-1].erase({qPrev,q});
            }
            for(const auto& [qNext,dimNext] : auxdim[loc+1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc << "]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Block deleted" << std::endl;
                }
                #endif
                O[loc].erase({q,qNext});
            }
            auxdim[loc].erase(q);
        }
        else
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "Counter decremented to " << (it->second)-1 << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[loc-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Delete a column of operators (number of rows: " << dimPrev << ")" << std::endl;
                }
                #endif
                auto it_prev = O[loc-1].find({qPrev,q});
                assert(it_prev != O[loc-1].end());
                for(std::size_t row=0; row<dimPrev; ++row)
                {
                    (it_prev->second)[row].resize((it_prev->second)[row].size()-1);
                }
            }
            for(const auto& [qNext,dimNext] : auxdim[loc+1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << loc << "]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Delete a row of operators (number of columns: " << dimNext << ")" << std::endl;
                }
                #endif
                auto it_next = O[loc].find({q,qNext});
                assert(it_next != O[loc].end());
                (it_next->second).resize((it_next->second).size()-1);
            }
            (it->second)--;
        }
    }
    else
    {
        assert(boundary_condition == BC::INFINITE);
        if(it->second == 1)
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "Quantum number deleted" << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[N_sites-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Block deleted" << std::endl;
                }
                #endif
                O[N_sites-1].erase({qPrev,q});
            }
            for(const auto& [qNext,dimNext] : auxdim[1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tOperators[0]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Block deleted" << std::endl;
                }
                #endif
                O[0].erase({q,qNext});
            }
            auxdim[0].erase(q);
            auxdim[N_sites].erase(q);
        }
        else
        {
            #if DEBUG_VERBOSITY > 1
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "Counter decremented to " << (it->second)-1 << std::endl;
            }
            #endif
            for(const auto& [qPrev,dimPrev] : auxdim[N_sites-1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qPrev) << "}->{" << Sym::format<Symmetry>(q) << "}): Delete a column of operators (number of rows: " << dimPrev << ")" << std::endl;
                }
                #endif
                auto it_prev = O[N_sites-1].find({qPrev,q});
                assert(it_prev != O[N_sites-1].end());
                for(std::size_t row=0; row<dimPrev; ++row)
                {
                    (it_prev->second)[row].resize((it_prev->second)[row].size()-1);
                }
            }
            for(const auto& [qNext,dimNext] : auxdim[1])
            {
                #if DEBUG_VERBOSITY > 2
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\t\tO[0]({" << Sym::format<Symmetry>(q) << "}->{" << Sym::format<Symmetry>(qNext) << "}): Delete a row of operators (number of columns: " << dimNext << ")" << std::endl;
                }
                #endif
                auto it_next = O[0].find({q,qNext});
                assert(it_next != O[0].end());
                (it_next->second).resize((it_next->second).size()-1);
            }
            (it->second)--;
            auto it_edge = auxdim[N_sites].find(q);
            assert(it_edge != auxdim[N_sites].end());
            (it_edge->second)--;
            
        }
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
decrement_first_auxdim_OBC(const qType& qIn)
{
    assert(boundary_condition == BC::OPEN);
    #if DEBUG_VERBOSITY > 1
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tqAux[left edge->0]({" << Sym::format<Symmetry>(qIn) << "}): ";
    }
    #endif
    got_update();
    auto it = auxdim[0].find(qIn);
    assert(it != auxdim[0].end());
    if(it->second == 1)
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Quantum number deleted" << std::endl;
        }
        #endif
        for(const auto& [qOut,deg] : auxdim[1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[0]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Block deleted" << std::endl;
            }
            #endif
            O[0].erase({qIn,qOut});
        }
        auxdim[0].erase(qIn);
    }
    else
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Counter decremented to " << (it->second)-1 << std::endl;
        }
        #endif
        for(const auto& [qOut,deg] : auxdim[1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[0]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Delete a row of operators (number of columns: " << deg << ")" << std::endl;
            }
            #endif
            auto it_ops = O[0].find({qIn,qOut});
            assert(it_ops != O[0].end());
            (it_ops->second).resize((it_ops->second).size()-1);
        }
        (it->second)--;
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
decrement_last_auxdim_OBC(const qType& qOut)
{
    assert(boundary_condition == BC::OPEN);
    #if DEBUG_VERBOSITY > 1
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "\tqAux[" << N_sites-1 << "->right edge]({" << Sym::format<Symmetry>(qOut) << "}): ";
    }
    #endif
    got_update();
    auto it = auxdim[N_sites].find(qOut);
    assert(it != auxdim[N_sites].end());
    if(it->second == 1)
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Quantum number deleted" << std::endl;
        }
        #endif
        for(const auto& [qIn,deg] : auxdim[N_sites-1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Block deleted" << std::endl;
            }
            #endif
            O[N_sites-1].erase({qIn,qOut});
        }
        auxdim[N_sites].erase(qOut);
    }
    else
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Counter decremented to " << (it->second)-1 << std::endl;
        }
        #endif
        for(const auto& [qIn,deg] : auxdim[N_sites-1])
        {
            #if DEBUG_VERBOSITY > 2
            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
            {
                lout << "\t\tO[" << N_sites-1 << "]({" << Sym::format<Symmetry>(qIn) << "}->{" << Sym::format<Symmetry>(qOut) << "}): Delete a column of operators (number of rows: " << deg << ")" << std::endl;
            }
            #endif
            auto it_ops = O[N_sites-1].find({qIn,qOut});
            assert(it_ops != O[N_sites-1].end());
            for(std::size_t row=0; row<deg; ++row)
            {
                (it_ops->second)[row].resize((it_ops->second)[row].size()-1);
            }
        }
        (it->second)--;
    }
}



template<typename Symmetry, typename Scalar> std::size_t MpoTerms<Symmetry,Scalar>::
get_auxdim(const std::size_t loc, const qType& q) const
{
    std::size_t loc_eff;
    if(boundary_condition == BC::INFINITE)
    {
        loc_eff = loc%N_sites;
    }
    else
    {
        loc_eff = loc;
    }
    auto it = auxdim[loc_eff].find(q);
    if (it != auxdim[loc_eff].end())
    {
        return (it->second);
    }
    else
    {
        return 0;
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
assert_hilbert(const std::size_t loc, const std::size_t dim)
{
	if(hilbert_dimension[loc] == 0)
	{
		hilbert_dimension[loc] = dim;
	}
	else
	{
		assert(hilbert_dimension[loc] == dim and "Dimensions of operator and local Hilbert space do not match!");
	}
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
save_label(const std::size_t loc, const std::string& info_label)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    if(info_label != "")
    {
        info[loc].push_back(info_label);
    }
}

template<typename Symmetry, typename Scalar> std::vector<std::string> MpoTerms<Symmetry,Scalar>::
get_info() const
{
    std::vector<std::string> res(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::stringstream ss;
        if (info[loc].size()>0)
        {
            std::copy(info[loc].begin(), info[loc].end()-1, std::ostream_iterator<std::string>(ss,","));
            ss << info[loc].back();
        }
        else
        {
            ss << "no info";
        }
        
        res[loc] = ss.str();
        
        while (res[loc].find("perp") != std::string::npos) res[loc].replace(res[loc].find("perp"), 4, "⟂");
        while (res[loc].find("para") != std::string::npos) res[loc].replace(res[loc].find("para"), 4, "∥");
        while (res[loc].find("prime") != std::string::npos) res[loc].replace(res[loc].find("prime"), 5, "'");
        while (res[loc].find("Perp") != std::string::npos) res[loc].replace(res[loc].find("Perp"), 4, "⟂");
        while (res[loc].find("Para") != std::string::npos) res[loc].replace(res[loc].find("Para"), 4, "∥");
        while (res[loc].find("Prime") != std::string::npos) res[loc].replace(res[loc].find("Prime"), 5, "'");
        while (res[loc].find("mu") != std::string::npos) res[loc].replace(res[loc].find("mu"), 2, "µ");
        while (res[loc].find("Delta") != std::string::npos) res[loc].replace(res[loc].find("Delta"), 5, "Δ");
        while (res[loc].find("next") != std::string::npos) res[loc].replace(res[loc].find("next"), 4, "ₙₑₓₜ");
        while (res[loc].find("prev") != std::string::npos) res[loc].replace(res[loc].find("prev"), 4, "ₚᵣₑᵥ");
        while (res[loc].find("3site") != std::string::npos) res[loc].replace(res[loc].find("3site"), 5, "₃ₛᵢₜₑ");
        while (res[loc].find("sub") != std::string::npos) res[loc].replace(res[loc].find("sub"), 3, "ˢᵘᵇ");
        while (res[loc].find("rung") != std::string::npos) res[loc].replace(res[loc].find("rung"), 4, "ʳᵘⁿᵍ");
        while (res[loc].find("t0") != std::string::npos) res[loc].replace(res[loc].find("t0"), 2, "t₀");
        
        //⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁺ ⁻ ⁼ ⁽ ⁾ ₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₊ ₋ ₌ ₍ ₎
        //ᵃ ᵇ ᶜ ᵈ ᵉ ᶠ ᵍ ʰ ⁱ ʲ ᵏ ˡ ᵐ ⁿ ᵒ ᵖ ʳ ˢ ᵗ ᵘ ᵛ ʷ ˣ ʸ ᶻ
        //ᴬ ᴮ ᴰ ᴱ ᴳ ᴴ ᴵ ᴶ ᴷ ᴸ ᴹ ᴺ ᴼ ᴾ ᴿ ᵀ ᵁ ⱽ ᵂ
        //ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
        //ᵅ ᵝ ᵞ ᵟ ᵋ ᶿ ᶥ ᶲ ᵠ ᵡ ᵦ ᵧ ᵨ ᵩ ᵪ
    }
    return res;
}
template<typename Symmetry, typename Scalar> const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, Eigen::SparseMatrix<Scalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>>>>>>& MpoTerms<Symmetry,Scalar>::
get_W_power(std::size_t power) const
{
    assert(power > 0 and "Power has to be at least 1");
    assert(power <= current_power and "This power of the MPO has not been calculated!");
    if(power == 1)
    {
        return W;
    }
    else
    {
        return W_powers[power-2];
    }
}

template<typename Symmetry, typename Scalar> const std::vector<Qbasis<Symmetry>>& MpoTerms<Symmetry,Scalar>::
get_qAux_power(std::size_t power) const
{
    assert(power > 0 and "Power has to be at least 1");
    assert(power <= current_power and "This power of the MPO has not been calculated!");
    if(power == 1)
    {
        return qAux;
    }
    else
    {
        return qAux_powers[power-2];
    }
}


template<typename Symmetry, typename Scalar> const std::vector<std::vector<typename Symmetry::qType>>& MpoTerms<Symmetry,Scalar>::
get_qOp_power(std::size_t power) const
{
    assert(power > 0 and "Power has to be at least 1");
    assert(power <= current_power and "This power of the MPO has not been calculated!");
    if(power == 1)
    {
        return qOp;
    }
    else
    {
        return qOp_powers[power-2];
    }
}

template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
check_qPhys() const
{
    bool all = true;
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        if(!GOT_QPHYS[loc])
        {
            all = false;
        }
    }
    return all;
}


template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
scale(const double factor, const double offset)
{
    if (std::abs(factor-1.) > ::mynumeric_limits<double>::epsilon())
    {
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            for(const auto& [qIn, rows] : auxdim[loc])
            {
                for(const auto& [qOut, cols] : auxdim[loc+1])
                {
                    auto it = O[loc].find({qIn,qOut});
                    assert(it != O[loc].end());
                    for(std::size_t row=0; row<rows; ++row)
                    {
                        for(std::size_t col=0; col<cols; ++col)
                        {
                            std::map<qType,OperatorType>& existing_ops = (it->second)[row][col];
                            for(auto& [q,op] : existing_ops)
                            {
                                op *= factor;
                            }
                        }
                    }
                }
            }
        }
    }

    if (std::abs(offset) > ::mynumeric_limits<double>::epsilon())
    {
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            std::size_t hd = hilbert_dimension[loc];
            if(hd > 0)
            {
                SiteOperator<Symmetry,Scalar> Id;
                Id.data = Matrix<Scalar,Dynamic,Dynamic>::Identity(hd,hd).sparseView();
                #ifdef OPLABELS
                Id.label = "id";
                #endif
                push(loc, {Id}, offset);
            }
        }
    }
}

template<typename Symmetry, typename Scalar> template<typename OtherScalar> MpoTerms<Symmetry, OtherScalar> MpoTerms<Symmetry,Scalar>::
cast()
{
    MpoTerms<Symmetry, OtherScalar> other(N_sites, boundary_condition);
    other.set_name(label);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(const auto& [qIn, rows] : auxdim[loc])
        {
            for(const auto& [qOut, cols] : auxdim[loc+1])
            {
                auto it = O[loc].find({qIn,qOut});
                assert(it != O[loc].end());
                for(std::size_t row=0; row<rows; ++row)
                {
                    for(std::size_t col=0; col<cols; ++col)
                    {
                        std::map<qType,OperatorType>& existing_ops = (it->second)[row][col];
                        for(auto& [q,op] : existing_ops)
                        {
                            op.template cast<OtherScalar>();
                        }
                    }
                }
            }
        }
    }
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(std::size_t i=0; i<info[loc].size(); ++i)
        {
            other.save_label(loc, info[loc][i]);
        }
    }
    return other;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
transform_base(const qType& qShift, const bool PRINT, const int factor)
{
	int length = (factor==-1)? static_cast<int>(qPhys.size()):factor;
    ::transform_base<Symmetry>(qPhys, qShift, PRINT, false, length); // from symmery/functions.h, BACK=false
    
    std::vector<std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>> O_new(N_sites);
    std::vector<std::map<qType, std::size_t>> auxdim_new(N_sites+1);
    
    if(qShift != Symmetry::qvacuum())
    {
        std::vector<std::size_t> symmetries_to_transform;
        for(std::size_t n=0; n<Symmetry::Nq; ++n)
        {
            if(Symmetry::kind()[n] != Sym::KIND::S and Symmetry::kind()[n] != Sym::KIND::T)
            {
                symmetries_to_transform.push_back(n);
            }
        }
        
        for(std::size_t n=0; n<symmetries_to_transform.size(); ++n)
        {
            qTot[symmetries_to_transform[n]] *= length;
            qVac[symmetries_to_transform[n]] *= length;
        }

        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            for(const auto& [qs_key, ops] : O[loc])
            {

                qType qIn = std::get<0>(qs_key);
                qType qOut = std::get<1>(qs_key);
                for(std::size_t n=0; n<symmetries_to_transform.size(); ++n)
                {
                        qIn[symmetries_to_transform[n]] *= length;
                        qOut[symmetries_to_transform[n]] *= length;
                }
                std::map<qType,OperatorType> empty_op_map;
                std::vector<std::map<qType,OperatorType>> temp_row(ops[0].size(), empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> ops_new(ops.size(), temp_row);
                for(std::size_t row=0; row<ops.size(); ++row)
                {
                    for(std::size_t col=0; col<ops[row].size(); ++col)
                    {
                        for(const auto& [q,op] : ops[row][col])
                        {
                            qType q_new = q;
                            OperatorType op_new = op;
                            for(std::size_t n=0; n<symmetries_to_transform.size(); ++n)
                            {
                                q_new[symmetries_to_transform[n]] *= length;
                                op_new.Q[symmetries_to_transform[n]] *= length ;
                            }
                            ops_new[row][col].insert({q_new,op_new});
                        }
                    }
                }
                O_new[loc].insert({{qIn,qOut},ops_new});
            }
        }
                
        for(std::size_t loc=0; loc<N_sites+1; ++loc)
        {
            for(const auto& [q_key, deg] : auxdim[loc])
            {
                qType q = q_key;
                for(std::size_t n=0; n<symmetries_to_transform.size(); ++n)
                {
                    q[symmetries_to_transform[n]] *= length;
                }
                auxdim_new[loc].insert({q,deg});
            }
        }
        
        O = O_new;
        auxdim = auxdim_new;
        got_update();
        calc(1);
    }
}

template<typename Symmetry, typename Scalar> std::vector<std::vector<TwoSiteData<Symmetry,Scalar>>> MpoTerms<Symmetry,Scalar>::
calc_TwoSiteData() const
{
    std::vector<std::vector<TwoSiteData<Symmetry,Scalar>>> tsd(N_sites-1);
    
    for(std::size_t loc=0; loc<N_sites-1; ++loc)
    {
        auto tensor_basis = Symmetry::tensorProd(qPhys[loc], qPhys[loc+1]);
        for(std::size_t n_lefttop=0; n_lefttop<qPhys[loc].size(); ++n_lefttop)
        {
            for(std::size_t n_leftbottom=0; n_leftbottom<qPhys[loc].size(); ++n_leftbottom)
            {
                for(std::size_t t_left=0; t_left<qOp[loc].size(); ++t_left)
                {
                    if(std::array<qType,3> qCheck = {qPhys[loc][n_leftbottom], qOp[loc][t_left], qPhys[loc][n_lefttop]}; !Symmetry::validate(qCheck))
                    {
                        continue;
                    }
                    for(std::size_t n_righttop=0; n_righttop<qPhys[loc+1].size(); ++n_righttop)
                        {
                        for(std::size_t n_rightbottom=0; n_rightbottom<qPhys[loc+1].size(); ++n_rightbottom)
                        {
                            for(std::size_t t_right=0; t_right<qOp[loc+1].size(); ++t_right)
                            {
                                if(std::array<qType,3> qCheck = {qPhys[loc+1][n_rightbottom], qOp[loc+1][t_right], qPhys[loc+1][n_righttop]}; !Symmetry::validate(qCheck))
                                {
                                    continue;
                                }

                                auto qOp_merges = Symmetry::reduceSilent(qOp[loc][t_left], qOp[loc+1][t_right]);
                                
                                for(const auto& qOp_merge : qOp_merges)
                                {
                                    if(!qAux[loc+1].find(qOp_merge))
                                    {
                                        continue;
                                    }
                                    
                                    auto qPhys_tops = Symmetry::reduceSilent(qPhys[loc][n_lefttop], qPhys[loc+1][n_righttop]);
                                    auto qPhys_bottoms = Symmetry::reduceSilent(qPhys[loc][n_leftbottom], qPhys[loc+1][n_rightbottom]);
                                    for(const auto& qPhys_top : qPhys_tops)
                                    {
                                        for(const auto& qPhys_bottom : qPhys_bottoms)
                                        {
                                            auto qTensor_top = make_tuple(qPhys[loc][n_lefttop], n_lefttop, qPhys[loc+1][n_righttop], n_righttop, qPhys_top);
                                            std::size_t n_top = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qTensor_top));
                                            
                                            auto qTensor_bottom = make_tuple(qPhys[loc][n_leftbottom], n_leftbottom, qPhys[loc+1][n_rightbottom], n_rightbottom, qPhys_bottom);
                                            std::size_t n_bottom = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qTensor_bottom));

                                            Scalar factor_cgc = 1.;
                                            if(Symmetry::NON_ABELIAN)
                                            {
                                                factor_cgc = Symmetry::coeff_tensorProd(qPhys[loc][n_leftbottom], qPhys[loc+1][n_rightbottom], qPhys_bottom, qOp[loc][t_left], qOp[loc+1][t_right], qOp_merge, qPhys[loc][n_lefttop], qPhys[loc+1][n_righttop], qPhys_top);
                                            }
                                            if(std::abs(factor_cgc) < mynumeric_limits<double>::epsilon())
                                            {
                                                continue;
                                            }
                                            TwoSiteData<Symmetry,Scalar> entry({{n_lefttop,n_leftbottom,n_righttop,n_rightbottom,n_top,n_bottom}}, {{qPhys_top,qPhys_bottom}}, {{t_left,t_right}}, qOp_merge, factor_cgc);
                                            tsd[loc].push_back(entry);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return tsd;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
got_update()
{
    GOT_W = false;
    GOT_QOP = false;
    GOT_QAUX = false;
    current_power = 1;
}

template<typename Symmetry, typename Scalar> MpoTerms<Symmetry,Scalar> MpoTerms<Symmetry,Scalar>::
prod(const MpoTerms<Symmetry,Scalar>& top, const MpoTerms<Symmetry,Scalar>& bottom, const qType& qTot, const double tolerance)
{
    typedef typename Symmetry::qType qType;
    typedef SiteOperator<Symmetry, Scalar> OperatorType;
    typedef Eigen::SparseMatrix<Scalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> MatrixType;
    DMRG::VERBOSITY::OPTION VERB = std::max(top.get_verbosity(), bottom.get_verbosity());
    
    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "MPO multiplication of " << top.get_name() << "*" << bottom.get_name() << " to quantum number {" << Sym::format<Symmetry>(qTot) << "}" << std::endl;
    }
    assert(bottom.is_finalized() and top.is_finalized() and "Error: Multiplying non-finalized MPOs");
    assert(bottom.size() == top.size() and "Error: Multiplying two MPOs of different size");
    assert(bottom.get_boundary_condition() == top.get_boundary_condition() and "Error: Multiplying two MPOs with different boundary conditions");
    BC boundary_condition = bottom.get_boundary_condition();
    
    std::vector<qType> qTots = Symmetry::reduceSilent(bottom.get_qTot(), top.get_qTot());
    auto it = std::find(qTots.begin(), qTots.end(), qTot);
    assert(it != qTots.end() and "Cannot multiply these two operators to an operator with target quantum number");
    
    std::vector<std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>> O_bottom, O_top, O;
    std::vector<Qbasis<Symmetry>> qAux_bottom, qAux_top, qAux;
    std::vector<std::vector<qType>> qPhys, qPhys_check;
    std::vector<std::vector<qType>> qOp_bottom, qOp_top;
    
    std::map<qType,OperatorType> empty_op_map;
    
    qPhys = bottom.get_qPhys();
    qPhys_check = top.get_qPhys();
    
    O_bottom = bottom.get_O();
    O_top = top.get_O();

    qAux_bottom = bottom.get_qAux();
    qAux_top = top.get_qAux();
    
    qOp_bottom = bottom.get_qOp();
    qOp_top = top.get_qOp();
    
    std::size_t N_sites = bottom.size();
    O.resize(N_sites);
    qAux.resize(N_sites+1);

    std::size_t pos_qTot_out = 0;
    if(qTot == Symmetry::qvacuum())
    {
        pos_qTot_out = 1;
    }
    
    std::vector<std::size_t> row_qVac(N_sites);
    std::vector<std::size_t> col_qVac(N_sites);
    std::vector<std::size_t> row_qTot(N_sites);
    std::vector<std::size_t> col_qTot(N_sites);

    
    qAux[0] = qAux_bottom[0].combine(qAux_top[0]);
    
    
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t hilbert_dimension = qPhys[loc].size();
        
        assert(hilbert_dimension != 0 and "Not all Hilbert space dimensions have been set!");
        assert(hilbert_dimension == qPhys_check[loc].size() and "Local Hilbert space dimensions do not match!");
        
        qAux[loc+1] = qAux_bottom[loc+1].combine(qAux_top[loc+1]);
        for(const auto& entry_left : qAux[loc])
        {
            for(const auto& entry_right : qAux[loc+1])
            {
                std::size_t rows = std::get<2>(entry_left).size();
                std::size_t cols = std::get<2>(entry_right).size();
                qType Qin = std::get<0>(entry_left);
                qType Qout = std::get<0>(entry_right);
                std::vector<std::map<qType,OperatorType>> temp_row(cols, empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> temp(rows, temp_row);
                O[loc].insert({{Qin,Qout},temp});
            }
        }
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Lattice site " << loc << ":" << std::endl;
        }
        #endif
        std::vector<qType> Qins = qAux[loc].qs();
        for(const auto& Qin : Qins)
        {
            std::size_t total_rows = qAux[loc].inner_dim(Qin);
            auto qins = Sym::split<Symmetry>(Qin,qAux_top[loc].qs(), qAux_bottom[loc].qs());
            for(const auto& [qin_top, qin_bottom] : qins)
            {
                std::size_t rows_bottom = qAux_bottom[loc].inner_dim(qin_bottom);
                std::size_t rows_top = qAux_top[loc].inner_dim(qin_top);
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\tQin = {" << Sym::format<Symmetry>(Qin) << "} can be reached by {" << Sym::format<Symmetry>(qin_bottom) << "} + {" << Sym::format<Symmetry>(qin_top) << "}" << std::endl;
                }
                #endif
                std::vector<qType> Qouts = qAux[loc+1].qs();
                for(const auto& Qout : Qouts)
                {
                    if(boundary_condition == BC::OPEN and (loc+1 == N_sites) and (Qout != qTot))
                    {
                        continue;
                    }
                    std::size_t total_cols = qAux[loc+1].inner_dim(Qout);
                    auto qouts = Sym::split<Symmetry>(Qout,qAux_top[loc+1].qs(), qAux_bottom[loc+1].qs());
                    for(const auto& [qout_top, qout_bottom] : qouts)
                    {
                        std::size_t cols_bottom = qAux_bottom[loc+1].inner_dim(qout_bottom);
                        std::size_t cols_top = qAux_top[loc+1].inner_dim(qout_top);
                        #if DEBUG_VERBOSITY > 1
                        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                        {
                            lout << "\t\tQout = {" << Sym::format<Symmetry>(Qout) << "} can be reached by {" << Sym::format<Symmetry>(qout_bottom) << "} + {" << Sym::format<Symmetry>(qout_top) << "}" << std::endl;
                        }
                        #endif
                        auto it = O[loc].find({Qin, Qout});
                        auto it_bottom = O_bottom[loc].find({qin_bottom, qout_bottom});
                        auto it_top = O_top[loc].find({qin_top, qout_top});
                        std::size_t in_pos = qAux[loc].leftAmount(Qin, {qin_bottom, qin_top});
                        std::size_t out_pos = qAux[loc+1].leftAmount(Qout, {qout_bottom, qout_top});
                        for(std::size_t row_bottom=0; row_bottom<rows_bottom; ++row_bottom)
                        {
                            for(std::size_t row_top=0; row_top<rows_top; ++row_top)
                            {
                                std::size_t total_row = in_pos + row_bottom*rows_top + row_top;
                                if((qin_bottom == bottom.qVac and row_bottom == bottom.pos_qVac) and (qin_top == top.qVac and row_top == top.pos_qVac) and total_row != row_qVac[loc])
                                {
                                    row_qVac[loc] = total_row;
                                }
                                if((qin_bottom == bottom.qTot and row_bottom == bottom.pos_qTot) and (qin_top == top.qTot and row_top == top.pos_qTot) and total_row != row_qTot[loc])
                                {
                                    row_qTot[loc] = total_row;
                                }
                                for(std::size_t col_bottom=0; col_bottom<cols_bottom; ++col_bottom)
                                {
                                    for(std::size_t col_top=0; col_top<cols_top; ++col_top)
                                    {
                                        for(const auto& [qOp_bottom,op_bottom] : (it_bottom->second)[row_bottom][col_bottom])
                                        {
                                            for(const auto& [qOp_top,op_top] : (it_top->second)[row_top][col_top])
                                            {
                                                std::size_t total_col = out_pos + col_bottom*cols_top + col_top;
                                                if((qout_bottom == bottom.qVac and col_bottom == bottom.pos_qVac) and (qout_top == top.qVac and col_top == top.pos_qVac) and total_col != col_qVac[loc])
                                                {
                                                    col_qVac[loc] = total_col;
                                                }
                                                if((qout_bottom == bottom.qTot and col_bottom == bottom.pos_qTot) and (qout_top == top.qTot and col_top == top.pos_qTot) and total_col != col_qTot[loc])
                                                {
                                                    col_qTot[loc] = total_col;
                                                }
                                                std::map<qType,OperatorType>& ops = (it->second)[total_row][total_col];
                                                std::vector<qType> Qops = Symmetry::reduceSilent(qOp_bottom, qOp_top);
                                                for(const auto& Qop : Qops)
                                                {
                                                    if(std::array<qType,3> qCheck = {Qin,Qop,Qout}; !Symmetry::validate(qCheck))
                                                    {
                                                        continue;
                                                    }
                                                    #ifdef OPLABELS
                                                    OperatorType op(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(hilbert_dimension, hilbert_dimension).sparseView(), Qop, op_top.label+"*"+op_bottom.label);
                                                    #if DEBUG_VERBOSITY > 1
                                                    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                                                    {
                                                        lout << "\t\t\tBlock {" << Sym::format<Symmetry>(Qin) << "}->{" << Sym::format<Symmetry>(Qout) << "},\tPosition [" << total_row << "|" << total_col << "], " << op.label << " {" << Sym::format<Symmetry>(Qop) << "}" << std::endl;
                                                    }
                                                    #endif
                                                    #else
                                                    OperatorType op(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(hilbert_dimension,hilbert_dimension).sparseView(), Qop);
                                                    #if DEBUG_VERBOSITY > 1
                                                    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                                                    {
                                                        lout << "\t\t\tBlock {" << Sym::format<Symmetry>(Qin) << "}->{" << Sym::format<Symmetry>(Qout) << "},\tPosition [" << total_row << "|" << total_col << "], Operator {" << Sym::format<Symmetry>(Qop) << "}" << std::endl;
                                                    }
                                                    #endif
                                                    #endif
                                                    Scalar factor_9j = Symmetry::coeff_tensorProd(qin_bottom, qin_top, Qin, qOp_bottom, qOp_top, Qop, qout_bottom, qout_top, Qout);
                                                    if(std::abs(factor_9j) < ::mynumeric_limits<double>::epsilon())
                                                    {
                                                        continue;
                                                    }
                                                    for(std::size_t n_bottom=0; n_bottom<hilbert_dimension; ++n_bottom)
                                                    {
                                                        for(std::size_t n_top=0; n_top<hilbert_dimension; ++n_top)
                                                        {
                                                            if(std::array<qType,3> qCheck = {qPhys[loc][n_bottom], Qop, qPhys[loc][n_top]}; !Symmetry::validate(qCheck))
                                                            {
                                                                continue;
                                                            }
                                                            #if DEBUG_VERBOSITY > 2
                                                            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                                                            {
                                                                lout << "\t\t\t\tmat(" << n_top << "," << n_bottom << ") = 0";
                                                            }
                                                            #endif
                                                            
                                                            Scalar val = 0;
                                                            for(std::size_t n_middle=0; n_middle<hilbert_dimension; ++n_middle)
                                                            {
                                                                if(std::array<qType,3> qCheck = {qPhys[loc][n_bottom], qOp_bottom, qPhys[loc][n_middle]}; !Symmetry::validate(qCheck))
                                                                {
                                                                    continue;
                                                                }
                                                                if(std::array<qType,3> qCheck = {qPhys[loc][n_middle], qOp_top, qPhys[loc][n_top]}; !Symmetry::validate(qCheck))
                                                                {
                                                                    continue;
                                                                }
                                                                if(std::abs(op_top.data.coeff(n_top, n_middle)) < ::mynumeric_limits<double>::epsilon() or std::abs(op_bottom.data.coeff(n_middle, n_bottom)) < ::mynumeric_limits<double>::epsilon())
                                                                {
                                                                    continue;
                                                                }
                                                                Scalar factor_6j = Symmetry::coeff_Apair(qPhys[loc][n_top], qOp_top, qPhys[loc][n_middle], qOp_bottom, qPhys[loc][n_bottom], Qop);
                                                                if(std::abs(factor_6j) < ::mynumeric_limits<double>::epsilon())
                                                                {
                                                                    continue;
                                                                }
                                                                val += op_top.data.coeff(n_top, n_middle) * op_bottom.data.coeff(n_middle, n_bottom) * factor_9j * factor_6j;
                                                                #if DEBUG_VERBOSITY > 2
                                                                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                                                                {
                                                                    lout << " + " << op_top.data.coeff(n_top, n_middle) << "*" << op_bottom.data.coeff(n_middle, n_bottom) << "*" << factor_9j*factor_6j;
                                                                }
                                                                #endif
                                                            }
                                                            if(std::abs(val) < ::mynumeric_limits<double>::epsilon())
                                                            {
                                                                #if DEBUG_VERBOSITY > 2
                                                                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                                                                {
                                                                    lout << " = 0" << std::endl;
                                                                }
                                                                #endif
                                                                continue;
                                                            }
                                                            op.data.coeffRef(n_top, n_bottom) = val;
                                                            #if DEBUG_VERBOSITY > 2
                                                            if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                                                            {
                                                                lout << " = " << val << std::endl;
                                                            }
                                                            #endif
                                                        }
                                                    }
                                                    if(op.data.norm() > ::mynumeric_limits<double>::epsilon())
                                                    {
                                                        auto it_op = ops.find(Qop);
                                                        if(it_op != ops.end())
                                                        {
                                                            (it_op->second) = (it_op->second) + op;
                                                        }
                                                        else
                                                        {
                                                            ops.insert({Qop,op});
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(boundary_condition == BC::INFINITE)
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Infinite system: Swap rows and columns to ensure identity operators to be at [0|0] and [1|1]" << std::endl;
        }
        #endif
        prod_swap_IBC(O, row_qVac, col_qVac, row_qTot, col_qTot);
    }
    if(boundary_condition == BC::OPEN)
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Open system: Delete columns at last lattice site which do not match target quantum number" << std::endl;
        }
        #endif
        prod_delZeroCols_OBC(O[N_sites-1], qAux[N_sites], qAux[N_sites-1], qTot, col_qTot[N_sites-1]);
    }
    MpoTerms<Symmetry,Scalar> out(N_sites, boundary_condition, qTot);
    out.set_verbosity(VERB);
    out.reconstruct(O, qAux, qPhys, true, boundary_condition, qTot);
    out.set_name(top.get_name()+"*"+bottom.get_name());
    out.compress(tolerance);
    return out;
}

template<typename Symmetry, typename Scalar> MpoTerms<Symmetry,Scalar> MpoTerms<Symmetry,Scalar>::
sum(const MpoTerms<Symmetry,Scalar>& top, const MpoTerms<Symmetry,Scalar>& bottom, const double tolerance)
{
    typedef typename Symmetry::qType qType;
    typedef SiteOperator<Symmetry, Scalar> OperatorType;
    
    DMRG::VERBOSITY::OPTION VERB = std::max(top.get_verbosity(), bottom.get_verbosity());

    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
    {
        lout << "MPO addition of " << bottom.get_name() << "+" << top.get_name() << std::endl;
    }
    
    assert(bottom.is_finalized() and top.is_finalized() and "Error: Adding non-finalized MPOs");
    assert(bottom.size() == top.size() and "Error: Adding two MPOs of different size");

    assert(bottom.get_boundary_condition() == top.get_boundary_condition() and "Error: Adding two MPOs with different boundary conditions");
    BC boundary_condition_out = bottom.get_boundary_condition();
    
    std::vector<std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>> O_bottom, O_top, O_out;
    std::vector<Qbasis<Symmetry>> qAux_bottom, qAux_top, qAux_out;
    std::vector<std::vector<qType>> qPhys, qPhys_check;
    std::vector<std::vector<qType>> qOp_bottom, qOp_top;
    std::map<qType,OperatorType> empty_op_map;
    
    assert(bottom.get_qTot() == top.get_qTot() and "Addition only possible for MPOs with the same total quantum number.");
    qType qTot = bottom.get_qTot();
    
    qPhys = bottom.get_qPhys();
    qPhys_check = top.get_qPhys();
    
    O_bottom = bottom.get_O();
    O_top = top.get_O();

    qAux_bottom = bottom.get_qAux();
    qAux_top = top.get_qAux();
    
    qOp_bottom = bottom.get_qOp();
    qOp_top = top.get_qOp();
    
    std::size_t N_sites = bottom.size();
    O_out.resize(N_sites);
    qAux_out.resize(N_sites+1);
    
    std::vector<std::map<qType,std::size_t>> auxdim(N_sites+1);
    std::vector<std::size_t> auxdim_bottom_starts(N_sites+1);
    
    for(std::size_t loc=0; loc<=N_sites; ++loc)
    {
        for(const auto& entry : qAux_top[loc])
        {
            qType q = std::get<0>(entry);
            std::size_t deg = std::get<2>(entry).size();
            auxdim[loc].insert({q,deg});
            if(q == Symmetry::qvacuum())
            {
                auxdim_bottom_starts[loc] = deg;      
	    }
        }
        for(const auto& entry : qAux_bottom[loc])
        {
            qType q = std::get<0>(entry);
            std::size_t deg = std::get<2>(entry).size();
            auto it = auxdim[loc].find(q);
            if(it != auxdim[loc].end())
            {
                (it->second) += deg;
            }
            else
            {
                auxdim[loc].insert({q,deg});
            }
        }
        for(const auto& [q,deg] : auxdim[loc])
        {
            qAux_out[loc].push_back(q,deg);
        }
    }
    
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Lattice site " << loc << std::endl;
        }
        #endif
        std::size_t hilbert_dimension = qPhys[loc].size();
        assert(hilbert_dimension == qPhys_check[loc].size() and "Local Hilbert space dimensions do not match!");

        for(const auto& [qIn,rows] : auxdim[loc])
        {
            std::size_t row_start = 0;
            if(qAux_top[loc].find(qIn))
            {
                row_start = qAux_top[loc].inner_dim(qIn);
            }


            for(const auto& [qOut,cols] : auxdim[loc+1])
            {
                std::size_t col_start = 0;
                if(qAux_top[loc+1].find(qOut))
                {
                    col_start = qAux_top[loc+1].inner_dim(qOut);
                }
                #if DEBUG_VERBOSITY > 1
                if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                {
                    lout << "\tqIn = {" << Sym::format<Symmetry>(qIn) << "} and qOut = {" << Sym::format<Symmetry>(qOut) << "} is a " << rows << "x" << cols << "-matrix:" << std::endl;
                }
                #endif
                std::vector<std::map<qType,OperatorType>> temp_row(cols, empty_op_map);
                std::vector<std::vector<std::map<qType,OperatorType>>> O_temp(rows, temp_row);
                auto it_top = O_top[loc].find({qIn,qOut});
                auto it_bottom = O_bottom[loc].find({qIn,qOut});
                if(it_top != O_top[loc].end())
                {
                    const std::vector<std::vector<std::map<qType,OperatorType>>>& O_top_temp = it_top->second;
                    std::size_t top_rows = O_top_temp.size();
                    std::size_t top_cols = O_top_temp[0].size();
                    #if DEBUG_VERBOSITY > 1
                    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                    {
                        lout << "\t\t Block exists in top MPO and is written from [0|0] to [" << top_rows-1 << "|" << top_cols-1 << "]";
                    }
                    #endif
                    for(std::size_t row=0; row<top_rows; ++row)
                    {
                        for(std::size_t col=0; col<top_cols; ++col)
                        {
                            O_temp[row][col] = O_top_temp[row][col];
                        }
                    }
                }
                #if DEBUG_VERBOSITY > 1
                else
                {
                    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                    {
                        lout << "\t\t Block does not exist in top MPO";
                    }
                }
                #endif
                if(it_bottom != O_bottom[loc].end())
                {
                    const std::vector<std::vector<std::map<qType,OperatorType>>>& O_bottom_temp = it_bottom->second;
                    std::size_t bottom_rows = O_bottom_temp.size();
                    std::size_t bottom_cols = O_bottom_temp[0].size();
                    #if DEBUG_VERBOSITY > 1
                    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                    {
                        lout << "\t|\t Block exists in bottom MPO and is written from [" << row_start << "|" << col_start << "] to [" << row_start+bottom_rows-1 << "|" << col_start+bottom_cols-1 << "]" << std::endl;
                    }
                    #endif
                    for(std::size_t row=0; row<bottom_rows; ++row)
                    {
                        for(std::size_t col=0; col<bottom_cols; ++col)
                        {
                            O_temp[row_start+row][col_start+col] = O_bottom_temp[row][col];
                        }
                    }
                }
                #if DEBUG_VERBOSITY > 1
                else
                {
                    if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
                    {
                        lout << "\t|\t Block does not exist in bottom MPO" << std::endl;
                    }
                }
                #endif
                O_out[loc].insert({{qIn,qOut},O_temp});
            }
        }
    }

    MpoTerms<Symmetry,Scalar> out(N_sites,boundary_condition_out,qTot);
    out.set_verbosity(VERB);
    out.reconstruct(O_out, qAux_out, qPhys, true, boundary_condition_out,qTot);
    if(boundary_condition_out == BC::OPEN)
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Open System: Add both rows at first lattice site and both columns at last lattice site" << std::endl;
        }
        #endif
        out.add_to_row(0, Symmetry::qvacuum(), 0, out.delete_row(0, Symmetry::qvacuum(),1,false),1.);
        out.decrement_first_auxdim_OBC(Symmetry::qvacuum());
        out.add_to_col(N_sites-1, qTot, 0, out.delete_col(N_sites-1, qTot,1,false),1.);
        out.decrement_last_auxdim_OBC(qTot);
    }
    else
    {
        #if DEBUG_VERBOSITY > 1
        if(VERB != DMRG::VERBOSITY::OPTION::SILENT)
        {
            lout << "Infinite System: Add both incoming operator columns and both outgoing operator rows at each lattice site" << std::endl;
        }
        #endif
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            out.delete_col(loc, Symmetry::qvacuum(), auxdim_bottom_starts[loc+1],false);
            out.delete_row(loc, Symmetry::qvacuum(), auxdim_bottom_starts[loc]+bottom.get_pos_qTot(),false);
            out.add_to_col(loc, Symmetry::qvacuum(), top.get_pos_qTot(), out.delete_col(loc, Symmetry::qvacuum(), auxdim_bottom_starts[loc+1],false),1.);
            out.add_to_row(loc, Symmetry::qvacuum(), 0, out.delete_row(loc, Symmetry::qvacuum(), auxdim_bottom_starts[loc],false),1.);
		}
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            out.decrement_auxdim(loc, Symmetry::qvacuum());
            out.decrement_auxdim(loc, Symmetry::qvacuum());
        }
	}
    out.set_name(top.get_name()+"+"+bottom.get_name());
    out.compress(tolerance);
    return out;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
prod_delZeroCols_OBC(std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>& O_last, Qbasis<Symmetry>& qAux_last, Qbasis<Symmetry>& qAux_prev, const qType& qTot, const std::size_t col_qTot)
{
    std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>> O_new;
    Qbasis<Symmetry> qAux_new;
    qAux_new.push_back(qTot,1);
    for(const auto& entry : qAux_prev)
    {
        qType qIn = std::get<0>(entry);
        std::size_t rows = std::get<2>(entry).size();
        std::map<qType,OperatorType> empty_op_map;
        std::vector<std::map<qType,OperatorType>> temp_row(1,empty_op_map);
        std::vector<std::vector<std::map<qType,OperatorType>>> temp(rows,temp_row);
        auto it = O_last.find({qIn,qTot});
        assert(it != O_last.end());
        for(std::size_t row=0; row<rows; ++row)
        {
            temp[row][0] = (it->second)[row][col_qTot];
        }
        O_new.insert({{qIn,qTot},temp});
    }
    qAux_last.clear();
    qAux_last = qAux_new;
    O_last = O_new;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
prod_swap_IBC(std::vector<std::map<std::array<qType, 2>, std::vector<std::vector<std::map<qType,OperatorType>>>>>& O_out, std::vector<std::size_t>& row_qVac, std::vector<std::size_t>& col_qVac, std::vector<std::size_t>& row_qTot, std::vector<std::size_t>& col_qTot)
{
    std::size_t N_sites = O_out.size();
    qType qVac = Symmetry::qvacuum();

    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        if(row_qVac[loc] != 0)
        {
            for(auto& [qs, ops] : O_out[loc])
            {
                if(std::get<0>(qs) != qVac)
                {
                    continue;
                }
                std::vector<std::map<qType,OperatorType>> temp = ops[0];
                ops[0] = ops[row_qVac[loc]];
                ops[row_qVac[loc]] = temp;
            }
            if(row_qTot[loc] == 0)
            {
                row_qTot[loc] = row_qVac[0];
            }
        }
        if(col_qVac[loc] != 0)
        {
            for(auto& [qs, ops] : O_out[loc])
            {
                if(std::get<1>(qs) != qVac)
                {
                    continue;
                }
                std::size_t rows = ops.size();
                for(std::size_t row=0; row<rows; ++row)
                {
                    std::map<qType,OperatorType> temp = ops[row][0];
                    ops[row][0] = ops[row][col_qVac[loc]];
                    ops[row][col_qVac[loc]] = temp;
                }
            }
            if(col_qTot[loc] == 0)
            {
                col_qTot[loc] = col_qVac[0];
            }
        }
        if(row_qTot[loc] != 1)
        {
            for(auto& [qs, ops] : O_out[loc])
            {
                if(std::get<0>(qs) != qVac)
                {
                    continue;
                }
                std::vector<std::map<qType,OperatorType>> temp = ops[1];
                ops[1] = ops[row_qTot[loc]];
                ops[row_qTot[loc]] = temp;
            }
        }
        if(col_qTot[loc] != 1)
        {
            for(auto& [qs, ops] : O_out[loc])
            {
                if(std::get<1>(qs) != qVac)
                {
                    continue;
                }
                std::size_t rows = ops.size();
                for(std::size_t row=0; row<rows; ++row)
                {
                    std::map<qType,OperatorType> temp = ops[row][1];
                    ops[row][1] = ops[row][col_qTot[loc]];
                    ops[row][col_qTot[loc]] = temp;
                }
            }
        }
    }
}
template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
set_Identity()
{
    got_update();
    O.clear();
    O.resize(N_sites);
    auxdim.clear();
    auxdim.resize(N_sites+1);
    label = "Id";
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        auxdim[loc].insert({qVac,1});
        std::map<qType,OperatorType> op_map;
        OperatorType op = OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(qPhys[loc].size(),qPhys[loc].size()).sparseView(),qVac);
        #ifdef OPLABELS
        op.label = "id";
        #endif
        op_map.insert({qVac,op});
        std::vector<std::map<qType,OperatorType>> temp(1,op_map);
        std::vector<std::vector<std::map<qType,OperatorType>>> Oloc(1,temp);
        O[loc].insert({{qVac,qVac},Oloc});
    }
    auxdim[N_sites].insert({qVac,1});
    FINALIZED = true;
    calc(1);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
set_Zero()
{
    got_update();
    O.clear();
    O.resize(N_sites);
    auxdim.clear();
    auxdim.resize(N_sites+1);
    label = "Zero";
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        auxdim[loc].insert({qVac,1});
        std::map<qType,OperatorType> op_map;
        OperatorType op = OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(qPhys[loc].size(),qPhys[loc].size()).sparseView(),qVac);
        #ifdef OPLABELS
        op.label = "Zero";
        #endif
        op_map.insert({qVac,op});
        std::vector<std::map<qType,OperatorType>> temp(1,op_map);
        std::vector<std::vector<std::map<qType,OperatorType>>> Oloc(1,temp);
        O[loc].insert({{qVac,qVac},Oloc});
    }
    auxdim[N_sites].insert({qVac,1});
    FINALIZED = true;
    calc(1);
}

template<typename Symmetry, typename Scalar> std::vector<std::pair<typename Symmetry::qType, std::size_t>> MpoTerms<Symmetry,Scalar>::
base_order_IBC() const
{
    assert(boundary_condition == BC::INFINITE);
    std::vector<std::pair<qType, std::size_t>> vout(0);
    std::vector<std::pair<qType, std::size_t>> vout_temp(0);
    vout.push_back({qTot, pos_qTot});
    for(const auto& [qIn, rows] : auxdim[0])
    {
        auto it = O[0].find({qIn, qTot});
        assert(it != O[0].end());
        for(std::size_t row=0; row<rows; row++)
        {
            if(qIn == qVac and row == pos_qVac)
            {
                continue;
            }
            if(qIn == qTot and row == pos_qTot)
            {
                continue;
            }
            if((it->second)[row][pos_qTot].size() > 0)
            {
                vout.push_back({qIn,row});
            }
            else
            {
                vout_temp.push_back({qIn,row});
            }
        }
    }
    vout.insert(vout.end(), vout_temp.begin(), vout_temp.end());
    vout.push_back({qVac, pos_qVac});
    return vout;
}

template<typename Symmetry, typename Scalar>
std::tuple<std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, Eigen::SparseMatrix<Scalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>>>>>>, std::vector<Qbasis<Symmetry>>, std::vector<std::vector<typename Symmetry::qType>>, std::vector<std::vector<typename Symmetry::qType>>>
MpoTerms<Symmetry,Scalar>::
get_reversed_data() const
{
    assert(GOT_W and GOT_QAUX and GOT_QOP and check_qPhys());
    std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>> W_reverse(N_sites);
    std::vector<Qbasis<Symmetry>> qAux_reverse(N_sites+1);
    std::vector<std::vector<qType>> qOp_reverse(N_sites);
    std::vector<std::vector<qType>> qPhys_reverse(N_sites);

    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        W_reverse[loc] = W[N_sites-1-loc];
        qAux_reverse[loc] = qAux[N_sites-loc];
        qOp_reverse[loc] = qOp[N_sites-1-loc];
        qPhys_reverse[loc] = qPhys[N_sites-1-loc];
    }
    qAux_reverse[N_sites] = qAux_reverse[0];
    return std::make_tuple(W_reverse,qAux_reverse,qOp_reverse,qPhys_reverse);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
renormalize()
{
    got_update();
    std::vector<double> norm(N_sites, 0.);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(const auto& [qs,ops] : O[loc])
        {
            for(std::size_t row=0; row<ops.size(); ++row)
            {
                for(std::size_t col=0; col<ops[row].size(); ++col)
                {
                    for(const auto& [Q,op] : ops[row][col])
                    {
                        norm[loc] += op.data.norm();
                    }
                }
            }
        }
    }
    double overall_norm = 1;
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        overall_norm *= norm[loc];
    }
    overall_norm = std::pow(overall_norm, 1.0/N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(auto& [qs,ops] : O[loc])
        {
            for(std::size_t row=0; row<ops.size(); ++row)
            {
                for(std::size_t col=0; col<ops[row].size(); ++col)
                {
                    for(auto& [Q,op] : ops[row][col])
                    {
                        double factor = overall_norm/norm[loc];
                        op.data = factor * op.data;
                    }
                }
            }
        }
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
clear_opLabels()
{
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(auto& [qs,ops] : O[loc])
        {
            for(std::size_t row=0; row<ops.size(); ++row)
            {
                for(std::size_t col=0; col<ops[row].size(); ++col)
                {
                    for(auto& [Q,op] : ops[row][col])
                    {
                        op.label = "";
                    }
                }
            }
        }
    }
}

template<typename Symmetry, typename Scalar> std::tuple<double,std::size_t> MpoTerms<Symmetry,Scalar>::
auxdim_infos() const
{
    std::size_t maximum_auxdim = 0;
    double average_auxdim = 0;
    if(N_sites > 1)
    {
        std::size_t minimum_bond = 0;
        if(boundary_condition == BC::OPEN)
        {
            minimum_bond = 1;
        }
        std::size_t sum_auxdim = 0;
        for(std::size_t loc=minimum_bond; loc<N_sites; ++loc)
        {
            std::size_t local_auxdim = 0;
            for(const auto& [q,deg] : auxdim[loc])
            {
                local_auxdim += deg;
            }
            sum_auxdim += local_auxdim;
            if(local_auxdim > maximum_auxdim)
            {
                maximum_auxdim = local_auxdim;
            }
        }
        average_auxdim = (sum_auxdim*1.0) / (N_sites - minimum_bond * 1.0);
    }
    else
    {
        for(const auto& [q,deg] : auxdim[0])
        {
            maximum_auxdim += deg;
        }
        average_auxdim = maximum_auxdim*1.0;
    }
    return std::make_tuple(average_auxdim,maximum_auxdim);
}

template<typename Symmetry, typename Scalar> double MpoTerms<Symmetry,Scalar>::
memory(MEMUNIT memunit) const
{
    double mem_O = 0.;
    for(std::size_t loc=0; loc<O.size(); ++loc)
    {
        mem_O += calc_memory<std::size_t>(2 * Symmetry::Nq * O[loc].size(),memunit);
        for(const auto& [qs,ops] : O[loc])
        {
            for(std::size_t row=0; row<ops.size(); ++row)
            {
                for(std::size_t col=0; col<ops[row].size(); ++col)
                {
                    mem_O += calc_memory<std::size_t>(Symmetry::Nq * ops[row][col].size(),memunit);
                    for(const auto& [Q,op] : ops[row][col])
                    {
                        mem_O += calc_memory(op.data,memunit) + calc_memory(op.label,memunit);
                    }
                }
            }
        }
    }

    double mem_auxdim = 0.;
    for(std::size_t loc=0; loc<=N_sites; ++loc)
    {
        mem_auxdim +=  calc_memory<std::size_t>((Symmetry::Nq + 1) * auxdim[loc].size(),memunit);
    }
    
    double mem_qAux = 0.;
    if(GOT_QAUX)
    {
        for(std::size_t loc=0; loc<qAux.size(); ++loc)
        {
            for(std::size_t i=0; i<qAux[loc].data_.size(); ++i)
            {
                mem_qAux += calc_memory<std::size_t>(Symmetry::Nq + 1,memunit);
                mem_qAux += std::get<2>(qAux[loc].data_[i]).size() * (calc_memory("",memunit)+calc_memory<std::size_t>(1ul,memunit));
            }
            mem_qAux += calc_memory<std::size_t>(Symmetry::Nq * qAux[loc].history.size(),memunit);
            for(const auto& [q,vec] : qAux[loc].history)
            {
                mem_qAux += calc_memory<std::size_t>((2*Symmetry::Nq + 1) * vec.size(), memunit);
            }
        }
    }
    
    double mem_qOp = 0.;
    if(GOT_QOP)
    {
        for(std::size_t loc=0; loc<qOp.size(); ++loc)
        {
            mem_qOp += Symmetry::Nq * calc_memory<std::size_t>(qOp[loc].size(), memunit);
        }
    }
    
    double mem_qPhys = 0.;
    for(std::size_t loc=0; loc<qPhys.size(); ++loc)
    {
        if(GOT_QPHYS[loc])
        {
            mem_qOp += Symmetry::Nq * calc_memory<std::size_t>(qPhys[loc].size(), memunit);
        }
    }
    
    double mem_W = 0.;
    if(GOT_W)
    {
        for(std::size_t loc=0; loc<W.size(); ++loc)
        {
            for(std::size_t m=0; m<W[loc].size(); ++m)
            {
                for(std::size_t n=0; n<W[loc][m].size(); ++n)
                {
                    for(std::size_t t=0; t<W[loc][m][n].size(); ++t)
                    {
                        mem_W += W[loc][m][n][t].memory(memunit) + W[loc][m][n][t].overhead(memunit);
                    }
                }
            }
        }
    }
    
    double mem_info = 0.;
    mem_info += calc_memory(label,memunit);
    for(std::size_t loc=0; loc<info.size(); ++loc)
    {
        for(std::size_t i=0; i<info[loc].size(); ++i)
        {
            mem_info += calc_memory(info[loc][i],memunit);
        }
    }
    
    double mem_powers = 0.;
    for(std::size_t power=2; power<=current_power; ++power)
    {
        for(std::size_t loc=0; loc<W_powers[power-2].size(); ++loc)
        {
            for(std::size_t m=0; m<W_powers[power-2][loc].size(); ++m)
            {
                for(std::size_t n=0; n<W_powers[power-2][loc][m].size(); ++n)
                {
                    for(std::size_t t=0; t<W_powers[power-2][loc][m][n].size(); ++t)
                    {
                        mem_powers += W_powers[power-2][loc][m][n][t].memory(memunit) + W_powers[power-2][loc][m][n][t].overhead(memunit);
                    }
                }
            }
            mem_powers += Symmetry::Nq * calc_memory<std::size_t>(qOp_powers[power-2][loc].size(), memunit);
            for(std::size_t i=0; i<qAux_powers[power-2][loc].data_.size(); ++i)
            {
                mem_powers += calc_memory<std::size_t>(Symmetry::Nq + 1,memunit);
                mem_powers += std::get<2>(qAux_powers[power-2][loc].data_[i]).size() * (calc_memory("",memunit)+calc_memory<std::size_t>(1ul,memunit));
            }
            mem_powers += calc_memory<std::size_t>(Symmetry::Nq * qAux_powers[power-2][loc].history.size(),memunit);
            for(const auto& [q,vec] : qAux_powers[power-2][loc].history)
            {
                mem_powers += calc_memory<std::size_t>((2*Symmetry::Nq + 1) * vec.size(), memunit);
            }
        }
    }
    
    return mem_O + mem_auxdim + mem_qAux + mem_qOp + mem_qPhys + mem_W + mem_info + mem_powers;
}

template<typename Symmetry, typename Scalar> double MpoTerms<Symmetry,Scalar>::
sparsity(const std::size_t power, const bool PER_MATRIX) const
{
    assert(power > 0 and power <= current_power);
    assert(GOT_W);
    assert(GOT_QAUX);
    assert(!PER_MATRIX and "? TODO");
    std::size_t nonzero_elements = 0;
    std::size_t overall_elements = 0;
    const std::vector<Qbasis<Symmetry>>& qAux_ = (power == 1) ? qAux : qAux_powers[power-2];
    const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry,MatrixType>>>>>& W_ = (power == 1) ? W : W_powers[power-2];
    std::vector<std::size_t> dims(N_sites+1);
    for(std::size_t i=0; i<qAux_[0].data_.size(); ++i)
    {
        dims[0] += std::get<2>(qAux_[0].data_[i]).size();
    }
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(std::size_t i=0; i<qAux_[loc+1].data_.size(); ++i)
        {
            dims[loc+1] += std::get<2>(qAux_[loc+1].data_[i]).size();
        }
        std::size_t hd = hilbert_dimension[loc];
        overall_elements += hd * hd * dims[loc] * dims[loc+1];
        for(std::size_t m=0; m<hd; ++m)
        {
            for(std::size_t n=0; n<hd; ++n)
            {
                const auto& left = qAux_[loc].data_;
                const auto& right = qAux_[loc+1].data_;
                for(std::size_t i=0; i<left.size(); ++i)
                {
                    for(std::size_t j=0; j<right.size(); ++j)
                    {
                        Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> entries = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>::Zero(std::get<2>(left[i]).size(),std::get<2>(right[j]).size());
                        for(std::size_t t=0; t<W_[loc][m][n].size(); ++t)
                        {
                            auto it = W_[loc][m][n][t].dict.find({std::get<0>(left[i]),std::get<0>(right[j])});
                            if(it != W_[loc][m][n][t].dict.end())
                            {
                                const MatrixType& mat = W_[loc][m][n][t].block[it->second];
                                assert(mat.rows() == std::get<2>(left[i]).size());
                                assert(mat.cols() == std::get<2>(right[j]).size());
                                for(std::size_t row=0; row<mat.rows(); ++row)
                                {
                                    for(std::size_t col=0; col<mat.cols(); ++col)
                                    {
                                        if(entries(row,col) == 0 and std::abs(mat.coeff(row,col)) > ::mynumeric_limits<double>::epsilon())
                                        {
                                            nonzero_elements++;
                                            entries(row,col) = 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return (nonzero_elements+0.)/(overall_elements+0.);
}
    
#endif
