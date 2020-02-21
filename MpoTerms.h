#ifndef DMRG_HAMILTONIAN_TERMS
#define DMRG_HAMILTONIAN_TERMS
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

/// \cond
#include <vector>
#include <string>
#include "boost/multi_array.hpp"
/// \endcond

#include "numeric_limits.h" // from TOOLS
#include "tensors/SiteOperator.h"
#include "symmetry/qarray.h"
#include "tensors/Qbasis.h"
#include "tensors/Biped.h"

template<typename Symmetry, typename Scalar> class MpoTerms
{
    typedef SiteOperator<Symmetry,Scalar> OperatorType;
    typedef typename Symmetry::qType qType;
    //typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    typedef Eigen::SparseMatrix<Scalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> MatrixType;
    
private:
    
    /**
     *  Operators that have been pushed into this instance of MpoTerms.
     *  Index structure: [Lattice site] Map: {qIn, qOut} -> [row][column]
     */
    std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>> O;
    
    /**
     *  Dimension of the auxiliar MPO basis
     *  Index i extends from 0 to N_sites, auxdim[i] connects lattice site i-1 and lattice site i
     *  Index structure: [i] Map: {q} -> value
     */
    std::vector<std::unordered_map<qType, std::size_t>> auxdim;
    
    /**
     *  Total quantum number of the MPO. All interactions pushed have to lead to this quantum number.
     */
    qType qTot;
    
    /**
     *  Vacuum quantum number of the MPO.
     */
    qType qVac = Symmetry::qvacuum();
    
    /**
     *  Index of right unity operator (operator thas is right to any local operators) in the quantum number block qTot. Equals 1, if qTot=Symmetry::qvacuum(), and 0 else.
     */
    std::size_t pos_qTot;
    
    /**
     *  Index of left unity operator (operator that is left to any local operators) in the quantum number block Symmetry::qvacuum(). Equals 0.
     */
    std::size_t pos_qVac;
    
    /**
     *  Stores whether to use open boundary conditions or not (for VUMPS)
     */
    bool OPEN_BC;
	
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
     *  Operator bases of the squared MPO
     *  Index structure: [Lattice Site][Index of quantum number]
     */
    std::vector<std::vector<qType>> qOpSquare;
    
    /**
     *  MPO bases of the squared MPO.
     *  Index structure: [Lattice site]
     */
    std::vector<Qbasis<Symmetry>> qAuxSquare;
    
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
     *  Are squared values up to date?
     */
    bool GOT_SQUARE = false;
    
    /**
     *  Has qPhys been set at a certrain lattice site?
     *  Index structure: [Lattice Site]
     */
    std::vector<bool> GOT_QPHYS;
    
    /**
     *  Increments the MPO auxiliar basis dimension by one. Also manages allocation of O.
     *  For VUMPS: If OPEN_BC = false, loc >= N_sites is allowed. In this case, the method increments qAux[0] and qAux[N_sites] for loc = N_sites
     *  @param  loc  Lattice site, for OPEN_BC = true: 0 < loc < N_sites
     *  @param  q    Quantum number
     */
    void increment_auxdim(const std::size_t loc, const qType& q);
    
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
     *  For VUMPS: If OPEN_BC = false, loc >= N_sites is allowed. In this case qAux[0] = qAux[N_sites]
     *  @param  loc The auxiliar basis index, i.e. the basis connecting lattice site loc-1 and loc
     *  @param  q   Quantum number
     *  @return Dimension of auxiliar basis
     */
    std::size_t get_auxdim(const std::size_t loc, const qType& q) const;
    
    /**
     *  Checks whether the dimension of an operator matches the local Hilbert space dimension. Sets the latter if it has not been set yet.
     *  @param loc  Lattice site
     *  @param dim  Assumed dimension of local Hilbert space
     */
    void assert_hilbert(const std::size_t loc, const std::size_t dim);
    
    /**
     *  Checks for linearly dependent rows (including zero rows) within a certain block in O and manages their deletion.
     *  @param loc  Lattice site
     *  @param qIn  Quantum number of the block to check
     */
    bool eliminate_linearlyDependent_rows(const std::size_t loc, const qType& qIn);
    
    /**
     *  Checks for linearly dependent columns (including zero columns) within a certain block in O and manages their deletion.
     *  @param loc  Lattice site
     *  @param qOut Quantum number of the block to check
     */
    bool eliminate_linearlyDependent_cols(const std::size_t loc, const qType& qOut);
    
    /**
     *  Deletes a certain row in O
     *  @param  loc              Lattice site
     *  @param  qIn              Quantum number of the block
     *  @param  row_to_delete    Index of the row
     *  @param  SAMESITE    If true, then the row with the same quantum number is skipped. Useful for VUMPS compression, when there is only one lattice site.
     *  @return Deleted row as a map {q} -> [col]
     */
    std::unordered_map<qType, std::vector<SiteOperator<Symmetry,Scalar>>> delete_row(const std::size_t loc, const qType& qIn, const std::size_t row_to_delete, bool SAMESITE=false);
    
    /**
     *  Deletes a certain column in O
     *  @param  loc              Lattice site
     *  @param  qOut             Quantum number of the block
     *  @param  col_to_delete    Index of the column
     *  @param  SAMESITE    If true, then the column with the same quantum number is skipped. Useful for VUMPS compression, when there is only one lattice site.
     *  @return Deleted column as a map {q} -> [row]
     */
    std::unordered_map<qType, std::vector<SiteOperator<Symmetry,Scalar>>> delete_col(const std::size_t loc, const qType& qOut, const std::size_t col_to_delete, bool SAMESITE=false);
    
    /**
     *  Adds a multiple of another row to a certain row
     *  @param  loc     Lattice site
     *  @param  qIn     Quantum number of the block
     *  @param  row     Index of the row
     *  @param  ops     Map {q} -> [col] of the row that shall be added
     *  @param  factor  Optional factor to scale the added row
     */
    void add_to_row(const std::size_t loc, const qType& qIn, const std::size_t row, const std::unordered_map<qType,std::vector<OperatorType>>& ops, const Scalar factor = 1.);
    
    /**
     *  Adds a multiple of another column to a certain column
     *  @param  loc     Lattice site
     *  @param  qOut    Quantum number of the block
     *  @param  row     Index of the column
     *  @param  ops     Map {q} -> [row] of the column that shall be added
     *  @param  factor  Optional factor to scale the added column
     */
    void add_to_col(const std::size_t loc, const qType& qOut, const std::size_t col, const std::unordered_map<qType,std::vector<OperatorType>>& ops, const Scalar factor = 1.);
    
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
     *  Sets GOT_QAUX, GOT_QOP, GOT_W and GOT_SQUARE to false. Called whenever something is changed.
     */
    inline void got_update();
    
    /**
     *  Allows to reconstruct an instance of MpoTerms with preset data.
     *  @param  O_in            Operators, are copied into O
     *  @param  qAux_in         Qbasis from which auxdim is calculated
     *  @param  FINALIZED_IN    Shall this instance be FINALIZED?
     *  @param  OBC  Shall this instance have open boundary conditions? False for VUMPS
     *  @param  qTot_in    Total quantum number of the MPO
     */
    void reconstruct(const std::vector<std::unordered_map<std::array<qType,2>, std::vector<std::vector<OperatorType>>>>& O_in, const std::vector<Qbasis<Symmetry>>& qAux_in, const std::vector<std::vector<qType>>& qPhys_in, const bool FINALIZED_IN, const bool OPEN_BC_IN, const qType& qTot_in = Symmetry::qvacuum());
    
     /**
      *  Clears all relevant members and sets the right size for them.
      */
     void initialize();
    
    /**
     * Checks whether two rows in O can be added with respect to their operator quantum numbers. Needed for compression when for example singlet and triplet operators act in the triplet to triplet sector.
     * @param   loc Lattice site
     * @param   qIn Incoming quantum number
     * @param   row1    First row to check
     * @param   row2    Second row to check
     * @return  True, if both rows can be added.
     */
    bool rows_matching_quantum_numbers(const std::size_t loc, const qType& qIn, const std::size_t row1, const std::size_t row2);

    /**
     * Checks whether two columns in O can be added with respect to their operator quantum numbers. Needed for compression when for example singlet and triplet operators act in the triplet to triplet sector.
     * @param   loc Lattice site
     * @param   qOut    Outgoing quantum number
     * @param   col1    First column to check
     * @param   col2    Second column to check
     * @return  True, if both rows can be added.
     */
    bool cols_matching_quantum_numbers(const std::size_t loc, const qType& qOut, const std::size_t col1, const std::size_t col2);
    
    /**
     *  Compresses the MpoTerms. Needs FINALIZED Terms.
     */
    void compress();

protected:
    
    /**
     *  W matrix that stores the MPO in a convenient way.
     *  Index structure: [Lattice site][Upper Hilbert space basis entry][Lower Hilbert space basis entry][Quantum number of operators] Biped: {qIn,qOut} -> Matrix [row][column]
     */
    std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>> W;
    
    /**
     *  W matrix of the squared MPO
     *  Index structure: [Lattice site][Upper Hilbert space basis entry][Lower Hilbert space basis entry][Quantum number of operators] Biped: {qIn,qOut} -> Matrix [row][column]
     */
    std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>> Wsquare;
    
    /**
     *  Number of lattice sites (for VUMPS: in the unit cell)
     */
    std::size_t N_sites;
    
public:

    /**
     *  Constructor for an instance of MpoTerms with fixed lattice size (for VUMPS: of the unit cell)
     *  @param  L   Lattice size
     *  @param  OBC  Infinite boundary condition. False for VUMPS calculations.
     *  @param  qTot_in Total quantum number of the MPO
     */
    MpoTerms(const std::size_t L=1, const bool OBC=true, const qType& qTot_in=Symmetry::qvacuum());
    
    /**
     *  Same as initialize, but allows to set a new combination of lattice size, total MPO quantum number and boundary condition
     *  @param  L   Lattice size
     *  @param  OBC  Infinite boundary condition. False for VUMPS calculations.
     *  @param  qTot_in Total quantum number of the MPO
     */
    void initialize(const std::size_t L, const bool OBC, const qType& qTot_in);
	
    /**
     *  Pushes an interaction into this instance of MpoTerms.
     *  @param  loc     Lattice site where the interaction starts
     *  @param  opList  Vector of operators that make up the interaction. opList[i] acts on lattice site loc+i
     *  @param  qList   Vector of quantum numbers. qList[i] is left from lattice site loc+i, qList[i+1] is right from it
     *  @param  lambda  Scalar factor for the interaction
     */
    void push(const std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, const Scalar lambda = 1.0);
    
    /**
     *  Generates the quantum number list and then calls push(std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, Scalar lambda = 1.0).
     *  If more than one quantum number branch leads to the vacuum, for each quantum number branch an interaction is pushed.
     *  @param  loc     Lattice site where the interaction starts
     *  @param  opList  Vector of operators that make up the interaction. opList[i] acts on lattice site loc+i
     *  @param  lambda  Scalar factor for the interaction
     */
	void push(const std::size_t loc, const std::vector<OperatorType>& opList, const Scalar lambda = 1.0);
    
    /**
     *  Adds an interaction between lattice sites loc and loc+n to the MpoTerms.
     *  @param n        Distance (n=1 means next-neighbour)
     *  @param loc      Lattice site where the interaction starts
     *  @param lambda   Interaction strength
     *  @param outOp    Outgoing operator at site loc
     *  @param trans    Vector of transfer operators at sites loc+1, ..., loc+n-1
     *  @param inOp     Incoming operator at site loc+m
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    //void push(const std::size_t n, const std::size_t loc, const Scalar lambda, const OperatorType& outOp, const std::vector<OperatorType>& trans, const OperatorType& inOp);
    
    /**
     *  Adds a new local interaction to the MpoTerms
     *  @param loc      Lattice site
     *  @param op       SiteOperator acting on the local Hilbert space of site \p loc
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    //void push_local(const std::size_t loc, const Scalar lambda, const OperatorType& op);
    
    /**
     *  Adds a new nearest-neighbour interaction to the MpoTerms
     *  @param loc      Lattice site of first site
     *  @param op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param op2      SiteOperator acting on the local Hilbert space of site \p loc+1
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    //void push_tight(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& op2);
    
    /**
     *  Adds a new next-nearest-neighbour interaction to the MpoTerms
     *  @param loc      Lattice site of first site
     *  @param op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param trans    SiteOperator acting as transfer operator on the local Hilbert space of site \p loc+1
     *  @param op2      SiteOperator acting on the local Hilbert space of site \p loc+2
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    //void push_nextn(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& trans, const OperatorType& op2);
    
    /**
     *  Prints information about the current state of operators and MPO auxiliar basis.
     */
    void show() const;
    
    /**
     *  @param loc      Lattice site
     *  @param label    Information
     */
    void save_label(const std::size_t loc, const std::string& label);
    
    /**
     *  @param label_in    Name to be given to this instance of MpoTerms
     */
    void set_name(const std::string& label_in) {label = label_in;}
    
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
     *  @param  CALC_SQUARE Shall the squared MPO (W matrix, qOp and qAux) be calculated?
     */
    void finalize(const bool COMPRESS, const bool CALC_SQUARE);
        
    /**
     *  Calculates auxiliar bases, operator bases and the W matrix (if they are not up to date). Possibility to calculate the square of the MPO, filling the auxiliar bases, the operator bases and the W matrix for the squared MPO.
     *  @param  CALC_SQUARE     Shall the squared MPO be calculated?
     */
    void calc(const bool CALC_SQUARE);
    
    /**
     *  @return True, if this instance of MpoTerms has already been finalized.
     */
    bool is_finalized() const {return FINALIZED;}
    
    /**
     *  @return  All operators that make up the W matrix
     */
    const std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>>& get_O() const {return O;}
    
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
     *  @return  W matrix of squared MPO
     */
    const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>>& get_Wsquare() const {assert(GOT_SQUARE and "Squared MPO has not been calculated!"); return Wsquare;}
    
    /**
     *  @return  Auxiliar bases between all lattice sites for the squared MPO
     */
    const std::vector<Qbasis<Symmetry>>& get_qAuxSquare() {assert(GOT_SQUARE and "Squared MPO has not been calculated!"); return qAuxSquare;}
    
    /**
     *  @return  Bases of local operators for the squared MPO
     */
    const std::vector<std::vector<qType>>& get_qOpSquare() {assert(GOT_SQUARE and "Squared MPO has not been calculated!"); return qOpSquare;}
    
    /**
     *  @return Dimension of local Hilbert space. 0, if dimension has not been set.
     */
    std::size_t get_hilbert_dimension(const std::size_t loc) const {return hilbert_dimension[loc];}
    
    /**
     *  @return Boundary condition. False, if infinite boundary conditions (for VUMPS).
     */
    bool get_boundary_condition() const {return OPEN_BC;}
    
    /**
     *  @return Total MPO quantum number
     */
    const qType& get_qTot() const {return qTot;}
    
    /**
     *  @return Position of the total quantum number in its quantum number block.
     */
    const std::size_t get_pos_qTot(){return pos_qTot;}

    /**
     *  @return Have all physical bases of local Hilbert spaces been set?
     */
    bool check_qPhys() const {bool all = true; for(std::size_t loc=0; loc<N_sites; ++loc) if(!GOT_QPHYS[loc]) all = false; return all;}
    
    /**
     *  @return Has the squared MPO been calculated?
     */
    bool check_SQUARE() const {return GOT_SQUARE;}
    
    /**
     *  @return Size of the lattice (for VUMPS: unit cell)
     */
    std::size_t size() const {return N_sites;}
    
    /**
     *  @return  Name of this instance
     */
    std::string get_name() const {return label;}
    
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
    std::vector<std::pair<qType,std::size_t>> VUMPS_base_order() const;
        
    /**
     * Calculates the product of two MPOs.
     *  @param  bottom  MPO that shall be applied first
     *  @param  top MPO that shall be applied second
     *  @param  Qtot    Target quantum number for the product
     *  @return Product of both MPOs.
     */
    static MpoTerms<Symmetry,Scalar> prod(const MpoTerms<Symmetry,Scalar>& bottom, const MpoTerms<Symmetry,Scalar>& top, const qType& Qtot);
    
    /**
     * Calculates the sum of two MPOs. Does not work properly for VUMPS yet.
     *  @param  bottom  MPO for addition
     *  @param  top MPO for addition
     *  @return Sum of both MPOs.
     */
    static MpoTerms<Symmetry,Scalar> sum(const MpoTerms<Symmetry,Scalar>& bottom, const MpoTerms<Symmetry,Scalar>& top);
    
    /**
     * Swaps rows and columns after calculating the product of two MPOs for VUMPS. Ensures that the identity operator that connects the vacua at each lattice site appear at position [0|0] and the identity connecting the target quantum number (for VUMPS vacuum) at position [1|1].
     *  @param  O_out   Operators as created by the prod-method.
     *  @param  row_qVac    List of the rows of the vacuum identity operators for each lattice site
     *  @param  col_qVac    List of the columns of the vacuum identity operators for each lattice site
     *  @param  row_qVac    List of the rows of the target quantum number identity operators for each lattice site
     *  @param  row_qVac    List of the columns of the target quantum number identity operators for each lattice site
     */
    static void VUMPS_prod_swap(std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>>& O_out, std::vector<std::size_t>& row_qVac, std::vector<std::size_t>& col_qVac, std::vector<std::size_t>& row_qTot, std::vector<std::size_t>& col_qTot);
    
    /**
     * Creates an identity MPO
     *
     */
    void set_Identity();
    void set_Zero();
    
    const std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>& W_at(const std::size_t loc) const {return W[loc];}
    const std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>& Wsq_at(const std::size_t loc) const {return Wsquare[loc];}
    const std::vector<std::vector<std::vector<std::vector<Biped<Symmetry, MatrixType>>>>>& W_full() const {return W;}
    const std::vector<qType>& opBasisSq(const std::size_t loc) const {return qOpSquare[loc];}
    const std::vector<qType>& locBasis(const std::size_t loc) const {return qPhys[loc];}
    const Qbasis<Symmetry>& auxBasis(const std::size_t loc) const {return qAux[loc];}
    const Qbasis<Symmetry>& inBasis(const std::size_t loc) const {return qAux[loc];}
    const Qbasis<Symmetry>& outBasis(const std::size_t loc) const {return qAux[loc+1];}
    const std::vector<qType>& opBasis(const std::size_t loc) const {return qOp[loc];}
    const std::vector<std::vector<qType>>& locBasis() const {return qPhys;}
    const std::vector<Qbasis<Symmetry>>& auxBasis() const {return qAux;}
    const std::vector<std::vector<qType>>& opBasis() const {return qOp;}
    const std::vector<std::vector<qType>>& opBasisSq() const {return qOpSquare;}
    const qType& Qtarget() const {return qTot;}
    void setQtarget (const qType& q) {assert(false and "setQtarget should not be called after the MPO has been initialized.");}
    void setLocBasis(const std::vector<std::vector<qType>>& q) {for(std::size_t loc=0; loc<q.size(); ++loc) set_qPhys(loc, q[loc]);}
    void setLocBasis(const std::vector<qType>& q, std::size_t loc) {set_qPhys(loc, q);}
};

template<typename Symmetry> using MpoTermsXd  = MpoTerms<Symmetry,double>;
template<typename Symmetry> using MpoTermsXcd = MpoTerms<Symmetry,std::complex<double> >;

template<typename Symmetry, typename Scalar> MpoTerms<Symmetry,Scalar>::
MpoTerms(const std::size_t L, const bool OBC, const qType& qTot_in)
: N_sites(L), OPEN_BC(OBC), qTot(qTot_in)
{
    initialize();
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
initialize()
{
    std::cout << "Initializing an MPO with L=" << N_sites << ", OPEN_BC=" << OPEN_BC << " and qTot={" << Sym::format<Symmetry>(qTot) << "}" << std::endl;
    assert(OPEN_BC or qTot == qVac);
    if(qTot == Symmetry::qvacuum())
    {
        pos_qVac = 0;
        pos_qTot = 1;
    }
    else
    {
        pos_qVac = 0;
        pos_qTot = 0;
    }
    
    hilbert_dimension.resize(N_sites, 0);
    O.clear();
    O.resize(N_sites);
    auxdim.clear();
    auxdim.resize(N_sites+1);
    qPhys.clear();
    qPhys.resize(N_sites);
    GOT_QPHYS.resize(N_sites, false);
    info.clear();
    info.resize(N_sites);
    
    OperatorType zeroOp(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum());
    if(qTot == Symmetry::qvacuum())
    {
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            auxdim[loc].insert({Symmetry::qvacuum(),2ul});
            std::vector<OperatorType> temp_row(2, zeroOp);
            std::vector<std::vector<OperatorType>> temp(2, temp_row);
            O[loc].insert({{qVac,qVac},temp});
        }
        auxdim[N_sites].insert({Symmetry::qvacuum(),2ul});
    }
    else
    {
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            auxdim[loc].insert({Symmetry::qvacuum(),1ul});
            auxdim[loc].insert({qTot,1ul});
            std::vector<OperatorType> temp_row(1, zeroOp);
            std::vector<std::vector<OperatorType>> temp(1, temp_row);
            O[loc].insert({{qVac,qVac},temp});
            O[loc].insert({{qTot,qVac},temp});
            O[loc].insert({{qVac,qTot},temp});
            O[loc].insert({{qTot,qTot},temp});
        }

        auxdim[N_sites].insert({Symmetry::qvacuum(),1ul});
        auxdim[N_sites].insert({qTot,1ul});
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
initialize(const std::size_t L, const bool OPEN_BC_IN, const qType& qTot_in)
{
    N_sites = L;
    OPEN_BC = OPEN_BC_IN;
    qTot = qTot_in;
    initialize();
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
reconstruct(const std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>>& O_in, const std::vector<Qbasis<Symmetry>>& qAux_in, const std::vector<std::vector<qType>>& qPhys_in, const bool FINALIZED_IN, const bool OPEN_BC_IN, const qType& qTot_in)
{
    N_sites = O_in.size();
    OPEN_BC = OPEN_BC_IN;
    qTot = qTot_in;
    O = O_in;
    FINALIZED = FINALIZED_IN;
    qPhys = qPhys_in;
    
    assert(OPEN_BC or qTot == Symmetry::qvacuum());
    if(qTot == Symmetry::qvacuum())
    {
        pos_qVac = 0;
        pos_qTot = 1;
    }
    else
    {
        pos_qVac = 0;
        pos_qTot = 0;
    }
    
    hilbert_dimension.resize(N_sites);
    auxdim.resize(N_sites+1);
    GOT_QPHYS.resize(N_sites, false);
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
    if(FINALIZED)
    {
        calc(false);
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push(const std::size_t loc, const std::vector<OperatorType>& opList, const std::vector<qType>& qList, const Scalar lambda)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    assert((loc+opList.size() <= N_sites or !OPEN_BC) and "For finite lattices operators must not exceed lattice size");
    assert(!FINALIZED and "Terms have already been finalized");
    if(lambda != 0.)
    {
        save_label(loc, label);
        std::size_t n = opList.size();
        assert(qList.size() == n+1 and "Amount of quantum numbers does not match amount of operators!");
        assert(qList[0] == Symmetry::qvacuum() and qList[n] == qTot and "Quantum number list does not match the total MPO quantum number!");
        for(int i=0; i<n; ++i)
        {
            assert_hilbert((loc+i)%N_sites, opList[i].data.rows());
        }
        if(n == 1)
        {
            std::cout << "Local interaction at site " << loc << ":" << std::endl;
            assert(opList[0].Q == qTot and "Local operator does not match the total MPO quantum number!");
            add(loc, lambda*opList[0], qVac, qTot, pos_qVac, pos_qTot);
        }
        else
        {
            std::cout << n-1 << ".-neighbour interaction between the sites " << loc << " and " << loc+n-1 << ":" << std::endl;
            std::size_t row = pos_qVac;
            std::size_t col = get_auxdim(loc+1, qList[1]);
            increment_auxdim(loc+1, qList[1]);
            add(loc, lambda*opList[0], qVac, qList[1], row, col);
            for(int i=1; i<n-1; ++i)
            {
                row = col;
                col = get_auxdim(loc+1+i, qList[i+1]);
                increment_auxdim(loc+1+i, qList[i+1]);
                add((loc+i)%N_sites, opList[i], qList[i], qList[i+1], row, col);
            }
            row = col;
            col = pos_qTot;
            add((loc+n-1)%N_sites, opList[n-1], qList[n-1], qTot, row, col);
        }
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push(const std::size_t loc, const std::vector<OperatorType>& opList, const Scalar lambda)
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
        assert_hilbert(loc+m, opList[m].data.rows());
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
            std::cout << "This branch of quantum numbers leads to the total MPO quantum number: {" << Sym::format<Symmetry>(qList[0]) << "} -> ";
            for(int j=0; j<n-1; ++j)
            {
                std::cout << "{" << Sym::format<Symmetry>(qList[j+1]) << "} -> ";
            }
            std::cout << "{" << Sym::format<Symmetry>(qList[n]) << "}" << std::endl;
        }
    }
    assert(count == 1 and "More than one quantum number branch leads to the total MPO quantum number");
    push(loc, opList, qList, lambda);
}

/*template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push_local(const std::size_t loc, const Scalar lambda, const OperatorType& op)
{
    push(loc, {op}, lambda);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push_tight(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& op2)
{
    push(loc, {op1, op2}, lambda);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push_nextn(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& trans, const OperatorType& op2)
{
    push(loc, {op1, trans, op2}, lambda);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
push(const std::size_t width, const std::size_t loc, const Scalar lambda, const OperatorType& outOp, const std::vector<OperatorType>& trans, const OperatorType& inOp)
{
    std::vector<OperatorType> oplist(0);
    oplist.push_back(outOp);
    for(std::size_t m=0; m<trans.size(); ++m)
    {
        oplist.push_back(trans[m]);
    }
    oplist.push_back(inOp);
    push(loc, oplist, lambda);
}*/

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
show() const
{
    auto format = [](bool statement)->std::string
    {
        if(statement)
        {
            return "Yes";
        }
        else
        {
            return "No";
        }
    };
    
    std::cout << "\n####################################################################################################" << std::endl;
    std::cout << "Properties of this instance of MpoTerms:\n\tName = " << label << "\n\tBoundary condition: ";
    if(OPEN_BC)
    {
        std::cout << "Open system with " << N_sites << " lattice sites" << std::endl;
    }
    else
    {
        std::cout << "Periodic system for VUMPS with a unit cell of size " << N_sites << std::endl;
    }
    std::cout << "\tFinalized? " << format(FINALIZED) << "\n\tqPhys set? " << format(check_qPhys()) << "\n\tqAux calculated? " << format(GOT_QAUX) << std::endl;
    std::cout << "\tqOp calculated? " << format(GOT_QOP) << "\n\tW matrix calculated? " << format(GOT_W) << "\n\tSquared MPO calculated? " << format(GOT_SQUARE) << std::endl;
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::cout << "Lattice site: " << loc << std::endl;
        std::cout << "\tLocal Hilbert basis (" << hilbert_dimension[loc] << " dim):\t";
        for(std::size_t n=0; n<qPhys[loc].size(); ++n)
        {
            std::cout << "\t{" << Sym::format<Symmetry>(qPhys[loc][n]) << "}";
        }
        std::cout << "\n\tIncoming quantum numbers:\t";
        for(const auto& [qIn, deg] : auxdim[loc])
        {
            std::cout << "\t({" << Sym::format<Symmetry>(qIn) << "} [#=" << deg << "])";
        }
        std::cout << "\n\tOutgoing quantum numbers:\t";
        for(const auto& [qOut, deg] : auxdim[loc+1])
        {
            std::cout << "\t({" << Sym::format<Symmetry>(qOut) << "} [#=" << deg << "])";
        }
        std::cout << "\n\tOperators:" << std::endl;
        for(const auto& [q, op] : O[loc])
        {
            std::size_t rows = get_auxdim(loc, std::get<0>(q));
            std::size_t cols = get_auxdim(loc+1, std::get<1>(q));
            std::cout << "\t\tFor quantum numbers {" << Sym::format<Symmetry>(std::get<0>(q)) << "} -> {" << Sym::format<Symmetry>(std::get<1>(q)) << "}:\t";
            for(std::size_t row=0; row<rows; ++row)
            {
                for(std::size_t col=0; col<cols; ++col)
                {
                    if(op[row][col].data.norm() > ::mynumeric_limits<double>::epsilon())
                    {
                        std::cout << "\t(" << op[row][col].label << ", {" << Sym::format<Symmetry>(op[row][col].Q) << "} [" << row << "|" << col << "])";
                    }
                }
            }
            std::cout << std::endl;
        }
    }
    std::cout << "####################################################################################################\n" << std::endl;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
calc(const bool CALC_SQUARE)
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
    if(CALC_SQUARE and !GOT_SQUARE)
    {
        std::vector<qType> qTot_squares = Symmetry::reduceSilent(qTot,qTot);
        assert(qTot_squares.size() == 1 and "Target quantum number has to be unique for squaring the MPO.");
        MpoTerms square = MpoTerms<Symmetry,Scalar>::prod(*this,*this,qTot_squares[0]);
        square.set_name(label+"-Squared");
        square.calc(false);
        Wsquare = square.get_W();
        qAuxSquare = square.get_qAux();
        qOpSquare = square.get_qOp();
        GOT_SQUARE = true;
    }
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
                                    if((it->second)[row][col].Q == qOp[loc][t] and (it->second)[row][col].data.norm() > ::mynumeric_limits<double>::epsilon())
                                    {
                                        mat.coeffRef(row, col) = (it->second)[row][col].data.coeffRef(m,n);
                                        if(std::abs(mat.coeffRef(row, col)) > ::mynumeric_limits<double>::epsilon())
                                        {
                                            found_match = true;
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
        for(const auto& [q, op] : O[loc])
        {
            std::size_t rows = get_auxdim(loc, std::get<0>(q));
            std::size_t cols = get_auxdim(loc+1, std::get<1>(q));
            for(std::size_t row=0; row<rows; ++row)
            {
                for(std::size_t col=0; col<cols; ++col)
                {
                    qOp_set.insert(op[row][col].Q);
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
        for(const auto& [q,deg] : auxdim[loc])
        {
            qAux[loc].push_back(q,deg);
        }
    }
    GOT_QAUX = true;

}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
finalize(const bool COMPRESS, const bool CALC_SQUARE)
{
    assert(!FINALIZED);
    FINALIZED = true;
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        if(hilbert_dimension[loc] == 0) hilbert_dimension[loc] = 1;
        OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc], hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
        Id.label = "id";
        add(loc, Id, qVac, qVac, pos_qVac, pos_qVac);
        add(loc, Id, qTot, qTot, pos_qTot, pos_qTot);
    }
    if(OPEN_BC)
    {
        delete_row(0, qTot, pos_qTot);
        delete_col(N_sites-1, qVac, pos_qVac);
        
        if(qTot == qVac)
        {
            auto it = auxdim[0].find(qTot);
            (it->second)--;
            auto it2 = auxdim[N_sites].find(qVac);
            (it2->second)--;
        }
        else
        {
            auxdim[0].erase(qTot);
            auxdim[N_sites].erase(qVac);
            for(const auto& [qOut,deg] : auxdim[1])
            {
                O[0].erase({qTot,qOut});
            }
            for(const auto& [qIn,deg] : auxdim[N_sites-1])
            {
                O[N_sites-1].erase({qIn,qVac});
            }
        }
    }
    if(COMPRESS)
    {
        compress();
    }
    calc(CALC_SQUARE);
}



template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
eliminate_linearlyDependent_rows(const std::size_t loc, const qType& qIn)
{
    bool SAMESITE = false;
    if(N_sites == 1 and !OPEN_BC)
    {
        SAMESITE = true;
    }
    assert(hilbert_dimension[loc] > 0);
    std::size_t cols_eff = 0;
    std::size_t hd = hilbert_dimension[loc];
    for(const auto& [qOut, deg] : auxdim[(loc+1)%N_sites])
    {
        cols_eff += deg;
    }
    
    cols_eff = cols_eff * hd * hd;
    
    std::size_t rows = get_auxdim(loc, qIn);
    std::size_t rows_eff = rows;
    std::size_t skipcount = 0;
    if(!OPEN_BC and (loc == 0 or loc == 1) and qIn == qVac)
    {
        rows_eff = rows_eff-2;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(rows_eff, cols_eff);
    for(std::size_t row=0; row<rows; ++row)
    {
        std::size_t count = 0;
        if((!OPEN_BC and (loc == 0 or loc == 1) and qIn == qVac and (row == pos_qVac or row == pos_qTot)))
        {
            skipcount++;
            continue;
        }
        bool zero_row = true;
        for(const auto& [qOut, deg] : auxdim[(loc+1)%N_sites])
        {
            auto it = O[loc].find({qIn,qOut});
            assert(it != O[loc].end());
            for(std::size_t col=0; col<deg; ++col)
            {
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> opMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(hd,hd);
                if((it->second)[row][col].data.norm() > ::mynumeric_limits<double>::epsilon())
                {
                    opMat = (it->second)[row][col].data;
                    zero_row = false;
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
        if(zero_row)
        {
            if(rows == 1)
            {
                std::cout << "Lattice site " << loc << ", quantum number blocks {" << Sym::format<Symmetry>(qIn) << "} -> ...: Row " << row << " is a zero row and the last row in these blocks. Thus it is deleted." << std::endl;
                for(const auto& [qOut, deg] : auxdim[(loc+1)%N_sites])
                {
                    O[loc].erase({qIn,qOut});
                }
                for(const auto& [qPrev, deg] : auxdim[(loc+N_sites-1)%N_sites])
                {
                    O[(loc+N_sites-1)%N_sites].erase({qPrev,qIn});
                }
                auxdim[loc].erase(qIn);
                std::cout << "\tAuxiliar basis " << (loc+N_sites-1)%N_sites << " -> " << loc << ", quantum number {" << Sym::format<Symmetry>(qIn) << "}: Deleted" << std::endl;

                if(!OPEN_BC and loc == 0)
                {
                    auxdim[N_sites].erase(qIn);
                    std::cout << "\tWith respect to VUMPS: Also done at right edge" << std::endl;
                }
            }
            else
            {
                std::cout << "Lattice site " << loc << ", quantum number blocks {" << Sym::format<Symmetry>(qIn) << "} -> ...: Row " << row << " is a zero row, but not the last row in these blocks. Thus it is deleted." << std::endl;
                delete_row(loc, qIn, row);
                delete_col((loc+N_sites-1)%N_sites, qIn, row, SAMESITE);
                auto it = auxdim[loc].find(qIn);
                assert(it != auxdim[loc].end());
                (it->second)--;
                std::cout << "\tAuxiliar basis " << (loc+N_sites-1)%N_sites << " -> " << loc << ", quantum number {" << Sym::format<Symmetry>(qIn) << "}: Dimension reduced to " << it->second << std::endl;
                if(!OPEN_BC and loc == 0)
                {
                    auto it2 = auxdim[N_sites].find(qIn);
                    assert(it2 != auxdim[N_sites].end());
                    (it2->second)--;
                    std::cout << "\tWith respect to VUMPS: Also done at right edge" << std::endl;
                }
            }
            return true;
        }
    }
    if(rows > 3 or ((OPEN_BC or (loc != 0 and loc != 1) or qIn != qVac) and rows> 1))
    {
        std::size_t rowskipcount = 0;
        for(std::size_t row_to_delete=0; row_to_delete<rows; ++row_to_delete)
        {
            if((!OPEN_BC and (loc == 0 or loc == 1) and qIn == qVac and (row_to_delete == pos_qVac or row_to_delete == pos_qTot)))
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
            
            if((tempmat*vec - tempvec).norm() < ::mynumeric_limits<double>::epsilon())
            {
                std::cout << "Lattice site " << loc << ", quantum number blocks {" << Sym::format<Symmetry>(qIn) << "} -> ...: Row " << row_to_delete << " can be written as linear combination of other rows, thus:" << std::endl;

                auto it = auxdim[loc].find(qIn);
                assert(it != auxdim[loc].end());
                bool quantum_numbers_match = true;
                std::size_t colskipcount2 = 0;
                for(std::size_t col=0; col<(it->second); ++col)
                {
                    if((!OPEN_BC and (loc == 0 or loc == 1) and qIn == qVac and (col == pos_qVac or col == pos_qTot)) or col == row_to_delete)
                    {
                        colskipcount2++;
                        continue;
                    }
                    if(std::abs(vec(col-colskipcount2)) > ::mynumeric_limits<double>::epsilon() and !cols_matching_quantum_numbers((loc+N_sites-1)%N_sites, qIn, row_to_delete, col))
                    {
                        quantum_numbers_match = false;
                    }
                }
                
                if(quantum_numbers_match)
                {
                    delete_row(loc, qIn, row_to_delete);
                    std::unordered_map<qType, std::vector<OperatorType>> deleted_col = delete_col((loc+N_sites-1)%N_sites, qIn, row_to_delete, SAMESITE);
                    
                    (it->second)--;
                    std::cout << "\tAuxiliar basis " << (loc+N_sites-1)%N_sites << " -> " << loc << ", quantum number {" << Sym::format<Symmetry>(qIn) << "}: Dimension reduced to " << it->second << std::endl;
                    if(!OPEN_BC and loc == 0)
                    {
                        auto it2 = auxdim[N_sites].find(qIn);
                        assert(it2 != auxdim[N_sites].end());
                        (it2->second)--;
                        std::cout << "\tWith respect to VUMPS: Also done at right edge" << std::endl;
                    }

                    std::size_t colskipcount = 0;
                    for(std::size_t col=0; col<(it->second); ++col)
                    {
                        if((!OPEN_BC and (loc == 0 or loc == 1) and qIn == qVac and (col == pos_qVac or col == pos_qTot)))
                        {
                            colskipcount++;
                            continue;
                        }
                        if(std::abs(vec(col-colskipcount)) > ::mynumeric_limits<double>::epsilon())
                        {
                            add_to_col((loc+N_sites-1)%N_sites, qIn, col, deleted_col, vec(col-colskipcount));
                        }
                    }
                    return true;
                }
                else
                {
                    std::cout << "\tHowever, to resolve the linear dependency addition of operators with different quantum numbers would be required." << std::endl;
                }
            }
        }
    }

    return false;
}

template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
eliminate_linearlyDependent_cols(const std::size_t loc, const qType& qOut)
{
    bool SAMESITE = false;
    if(N_sites == 1 and !OPEN_BC)
    {
        SAMESITE = true;
    }
    assert(hilbert_dimension[loc] > 0);
    std::size_t rows_eff = 0;
    std::size_t hd = hilbert_dimension[loc];
    for(const auto& [qIn, deg] : auxdim[loc])
    {
        rows_eff += deg;
    }
    
    rows_eff = rows_eff * hd * hd;
    std::size_t cols = get_auxdim(loc+1, qOut);
    std::size_t cols_eff = cols;
    std::size_t skipcount = 0;
    if(!OPEN_BC and (loc == N_sites-1 or loc == (2*N_sites-2)%N_sites) and qOut == qVac)
    {
        cols_eff = cols_eff-2;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(rows_eff, cols_eff);
    for(std::size_t col=0; col<cols; ++col)
    {
        std::size_t count = 0;
        if((!OPEN_BC and (loc == N_sites-1 or loc == (2*N_sites-2)%N_sites) and qOut == qVac and (col == pos_qVac or col == pos_qTot)))
        {
            skipcount++;
            continue;
        }
        bool zero_col = true;
        for(const auto& [qIn, deg] : auxdim[loc])
        {
            auto it = O[loc].find({qIn,qOut});
            for(std::size_t row=0; row<deg; ++row)
            {
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> opMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(hd,hd);
                assert(it != O[loc].end());
                if((it->second)[row][col].data.norm() > ::mynumeric_limits<double>::epsilon())
                {
                    opMat = (it->second)[row][col].data;
                    zero_col = false;
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
        if(zero_col)
        {
            if(cols == 1)
            {
                std::cout << "Lattice site " << loc << ", quantum number blocks ... -> {" << Sym::format<Symmetry>(qOut) << "}: Column " << col << " is a zero column and the last column in these blocks. Thus it is deleted." << std::endl;
                for(const auto& [qIn, deg] : auxdim[loc])
                {
                    O[loc].erase({qIn,qOut});
                }
                for(const auto& [qNext, deg] : auxdim[(loc+2)%N_sites])
                {
                    O[(loc+1)%N_sites].erase({qOut,qNext});
                }
                auxdim[(loc+1)%N_sites].erase(qOut);
                std::cout << "\tAuxiliar basis " << loc << " -> " << (loc+1)%N_sites << ", quantum number {" << Sym::format<Symmetry>(qOut) << "}: Deleted" << std::endl;
                if(!OPEN_BC and loc == N_sites-1)
                {
                    auxdim[N_sites].erase(qOut);
                    std::cout << "\tWith respect to VUMPS: Also done at right edge" << std::endl;
                }
            }
            else
            {
                std::cout << "Lattice site " << loc << ", quantum number blocks ... -> {" << Sym::format<Symmetry>(qOut) << "}: Column " << col << " is a zero column, but not the last column in these blocks. Thus it is deleted." << std::endl;
                delete_col(loc, qOut, col);
                delete_row((loc+1)%N_sites, qOut, col, SAMESITE);
                auto it = auxdim[(loc+1)%N_sites].find(qOut);
                assert(it != auxdim[(loc+1)%N_sites].end());
                (it->second)--;
                std::cout << "\tAuxiliar basis " << loc << " -> " << (loc+1)%N_sites << ", quantum number {" << Sym::format<Symmetry>(qOut) << "}: Dimension reduced to " << it->second << std::endl;
                if(!OPEN_BC and loc == N_sites-1)
                {
                    auto it2 = auxdim[N_sites].find(qOut);
                    assert(it2 != auxdim[N_sites].end());
                    (it2->second)--;
                    std::cout << "\tWith respect to VUMPS: Also done at right edge" << std::endl;
                }
            }
            return true;
        }
    }
    if(cols > 3 or ((OPEN_BC or (loc != N_sites-1 and loc != (2*N_sites-2)%N_sites) or qOut != qVac) and cols>1))
    {
        std::size_t colskipcount = 0;
        for(std::size_t col_to_delete=0; col_to_delete<cols; ++col_to_delete)
        {
            if((!OPEN_BC and (loc == N_sites-1 or loc == (2*N_sites-2)%N_sites) and qOut == qVac and (col_to_delete == pos_qVac or col_to_delete == pos_qTot)))
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
            
            if((tempmat*vec - tempvec).norm() < ::mynumeric_limits<double>::epsilon())
            {
                std::cout << "Lattice site " << loc << ", quantum number blocks ... -> {" << Sym::format<Symmetry>(qOut) << "}: Column " << col_to_delete << " can be written as linear combination of other columns, thus:" << std::endl;
                
                auto it = auxdim[(loc+1)%N_sites].find(qOut);
                assert(it != auxdim[(loc+1)%N_sites].end());
                bool quantum_numbers_match = true;
                std::size_t rowskipcount2 = 0;
                for(std::size_t row=0; row<(it->second); ++row)
                {
                    if((!OPEN_BC and (loc == N_sites-1 or loc == (2*N_sites-2)%N_sites) and qOut == qVac and (row == pos_qVac or row == pos_qTot)) or row == col_to_delete)
                    {
                        rowskipcount2++;
                        continue;
                    }
                    if(std::abs(vec(row-rowskipcount2)) > ::mynumeric_limits<double>::epsilon() and !rows_matching_quantum_numbers((loc+1)%N_sites, qOut, col_to_delete, row))
                    {
                        quantum_numbers_match = false;
                    }
                }
                
                if(quantum_numbers_match)
                {
                    delete_col(loc, qOut, col_to_delete);
                    std::unordered_map<qType, std::vector<OperatorType>> deleted_row = delete_row((loc+1)%N_sites, qOut, col_to_delete, SAMESITE);
                    
                    (it->second)--;
                    std::cout << "\tAuxiliar basis " << loc << " -> " << (loc+1)%N_sites << ", quantum number {" << Sym::format<Symmetry>(qOut) << "}: Dimension reduced to " << it->second << std::endl;
                    if(!OPEN_BC and loc+1 == N_sites)
                    {
                        auto it2 = auxdim[N_sites].find(qOut);
                        assert(it2 != auxdim[N_sites].end());
                        (it2->second)--;
                        std::cout << "\tWith respect to VUMPS: Also done at right edge" << std::endl;
                    }
                    std::size_t rowskipcount = 0;
                    for(std::size_t row=0; row<(it->second); ++row)
                    {
                        if((!OPEN_BC and (loc == N_sites-1 or loc == (2*N_sites-2)%N_sites) and qOut == qVac and (row == pos_qVac or row == pos_qTot)))
                        {
                            rowskipcount++;
                            continue;
                        }
                        if(std::abs(vec(row-rowskipcount)) > ::mynumeric_limits<double>::epsilon())
                        {
                            add_to_row((loc+1)%N_sites, qOut, row, deleted_row, vec(row-rowskipcount));
                        }
                    }
                   
                    return true;
                }
                else
                {
                    std::cout << "\tHowever, to resolve the linear dependency addition of operators with different quantum numbers would be required." << std::endl;
                }
            }
        }
    }
    return false;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
compress()
{
    assert(FINALIZED and "Terms need to be finalized before compression.");
    std::cout << "Starting VUMPS compression of this MPO:" << std::endl;
    show();
    assert(OPEN_BC or qVac == qTot);
    bool change = true;
    while(change)
    {
        change = false;
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            if(change)
            {
                break;
            }
            for(const auto& [qIn, rows] : auxdim[loc])
            {
                if(eliminate_linearlyDependent_rows(loc, qIn))
                {
                    change = true;
                    //std::cout << "Current MPO:" << std::endl;
                    //show();
                    break;
                }
            }
            if(change)
            {
                break;
            }
            for(const auto& [qOut, cols] : auxdim[loc+1])
            {
                if(eliminate_linearlyDependent_cols(loc, qOut))
                {
                    change = true;
                    //std::cout << "Current MPO:" << std::endl;
                    //show();
                    break;
                }
            }
        }
    }
    std::cout << "Compressed MPO:" << std::endl;
    show();
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
add_to_row(const std::size_t loc, const qType& qIn, const std::size_t row, const std::unordered_map<qType, std::vector<OperatorType>>& ops, const Scalar factor)
{
    std::cout << "\tLattice site " << loc << ", quantum number blocks {" << Sym::format<Symmetry>(qIn) << "} -> ...: Add another row scaled by factor " << factor << " to row " << row << std::endl;
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
            if((it->second)[row][col].data.norm() < ::mynumeric_limits<double>::epsilon())
            {
                (it->second)[row][col] = factor*(it2->second)[col];
            }
            else if((it2->second)[col].data.norm() > ::mynumeric_limits<double>::epsilon())
            {
                (it->second)[row][col] += factor*(it2->second)[col];
            }

        }
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
add_to_col(const std::size_t loc, const qType& qOut, const std::size_t col, const std::unordered_map<qType, std::vector<OperatorType>>& ops, const Scalar factor)
{
    std::cout << "\tLattice site " << loc << ", quantum number blocks ... -> {" << Sym::format<Symmetry>(qOut) << "}: Add another column scaled by factor " << factor << " to column " << col << std::endl;
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
            if((it->second)[row][col].data.norm() < ::mynumeric_limits<double>::epsilon())
            {
                (it->second)[row][col] = factor*(it2->second)[row];
            }
            else if((it2->second)[row].data.norm() > ::mynumeric_limits<double>::epsilon())
            {
                (it->second)[row][col] += factor*(it2->second)[row];
            }
        }
    }
}

template<typename Symmetry, typename Scalar> std::unordered_map<typename Symmetry::qType, std::vector<SiteOperator<Symmetry, Scalar>>> MpoTerms<Symmetry,Scalar>::
delete_row(const std::size_t loc, const qType& qIn, const std::size_t row_to_delete, bool SAMESITE)
{
    std::cout << "\tLattice site " << loc << ", quantum number blocks {" << Sym::format<Symmetry>(qIn) << "} -> ...: Delete row " << row_to_delete << std::endl;
    if(SAMESITE) std::cout << "\tPaying attention since a column has been deleted at the same site before" << std::endl;
    std::unordered_map<qType, std::vector<OperatorType>> deleted_row;
    std::size_t rows = get_auxdim(loc, qIn);
    assert(row_to_delete < rows and "Trying to delete a nonexisting row");
    got_update();
    for(const auto& [qOut, deg] : auxdim[loc+1])
    {
        auto it = O[loc].find({qIn, qOut});
        std::vector<OperatorType> temp;
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
        }
        (it->second).resize(rows-1);
        deleted_row.insert({qOut, temp});
    }
    return deleted_row;
}

template<typename Symmetry, typename Scalar> std::unordered_map<typename Symmetry::qType, std::vector<SiteOperator<Symmetry, Scalar>>> MpoTerms<Symmetry,Scalar>::
delete_col(const std::size_t loc, const qType& qOut, const std::size_t col_to_delete, bool SAMESITE)
{
    std::cout << "\tLattice site " << loc << ", quantum number blocks ... -> {" << Sym::format<Symmetry>(qOut) << "}: Delete column " << col_to_delete << std::endl;
    if(SAMESITE) std::cout << "\tPaying attention since a row has been deleted at the same site before" << std::endl;
    std::unordered_map<qType, std::vector<OperatorType>> deleted_col;
    std::size_t cols = get_auxdim(loc+1, qOut);
    assert(col_to_delete < cols and "Trying to delete a nonexisting column");
    got_update();
    for(const auto& [qIn, deg] : auxdim[loc])
    {
        auto it = O[loc].find({qIn, qOut});
        std::vector<OperatorType> temp;
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
            (it->second)[row].resize(cols-1);
        }
        deleted_col.insert({qIn, temp});
    }
    return deleted_col;
}

template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
rows_matching_quantum_numbers(const std::size_t loc, const qType& qIn, const std::size_t row1, const std::size_t row2)
{
    for(const auto& [qOut, cols] : auxdim[loc+1])
    {
        auto it = O[loc].find({qIn, qOut});
        assert(it != O[loc].end());
        for(std::size_t col=0; col<cols; ++col)
        {
            if((it->second)[row1][col].data.norm() > ::mynumeric_limits<double>::epsilon() and (it->second)[row2][col].data.norm() > :: mynumeric_limits<double>::epsilon() and (it->second)[row1][col].Q != (it->second)[row2][col].Q)
            {
                std::cout << "\t\tLattice site " << loc << ", quantum number block {" << Sym::format<Symmetry>(qIn) << "} -> {" << Sym::format<Symmetry>(qOut) << "}: " << (it->second)[row1][col].label << " in row " << row1 << " and " << (it->second)[row2][col].label << " in row " << row2 << " do not match!" << std::endl;
                return false;
            }
        }
    }
    return true;
}

template<typename Symmetry, typename Scalar> bool MpoTerms<Symmetry,Scalar>::
cols_matching_quantum_numbers(const std::size_t loc, const qType& qOut, const std::size_t col1, const std::size_t col2)
{
    for(const auto& [qIn, rows] : auxdim[loc])
    {
        auto it = O[loc].find({qIn, qOut});
        assert(it != O[loc].end());
        for(std::size_t row=0; row<rows; ++row)
        {
            if((it->second)[row][col1].data.norm() > ::mynumeric_limits<double>::epsilon() and (it->second)[row][col2].data.norm() > :: mynumeric_limits<double>::epsilon() and (it->second)[row][col1].Q != (it->second)[row][col2].Q)
            {
                std::cout << "\t\tLattice site " << loc << ", quantum number block {" << Sym::format<Symmetry>(qIn) << "} -> {" << Sym::format<Symmetry>(qOut) << "}: " << (it->second)[row][col1].label << " in column " << col1 << " and " << (it->second)[row][col2].label << " in column " << col2 << " do not match!" << std::endl;
                return false;
            }
        }
    }
    return true;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
add(const std::size_t loc, const OperatorType& op, const qType& qIn, const qType& qOut, const std::size_t leftIndex, const std::size_t rightIndex)
{
    std::cout << "Lattice site " << loc << ", quantum number block {" << Sym::format<Symmetry>(qIn) << "} -> {" << Sym::format<Symmetry>(qOut) << "}: New operator with quantum number {" << Sym::format<Symmetry>(op.Q) << "}, degeneragy indices [" << leftIndex << "|" << rightIndex << "]" << std::endl;
    std::size_t rows = get_auxdim(loc, qIn);
    std::size_t cols = get_auxdim(loc+1, qOut);
    auto it = O[loc].find({qIn, qOut});
    assert(leftIndex <= rows and "Index out of bounds");
    assert(rightIndex <= cols and "Index out of bounds");
    assert(it != O[loc].end() and "Quantum numbers not available");
    got_update();
    if((it->second)[leftIndex][rightIndex].data.norm() < ::mynumeric_limits<double>::epsilon())
    {
        (it->second)[leftIndex][rightIndex] = op;
    }
    else
    {
        (it->second)[leftIndex][rightIndex] += op;
    }
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
increment_auxdim(const std::size_t loc, const qType& q)
{
    assert(!OPEN_BC or (0 < loc and loc < N_sites));
    std::cout << "\tAuxiliar basis " << (loc+N_sites-1)%N_sites << " -> " << loc%N_sites << ", quantum number {" << Sym::format<Symmetry>(q) << "}:";
    got_update();
    auto it = auxdim[loc%N_sites].find(q);
    if (it != auxdim[loc%N_sites].end())
    {
        std::cout << " Dimension raised to " << it->second+1 << std::endl;
        for(const auto& [qPrev, dimPrev] : auxdim[(loc+N_sites-1)%N_sites])
        {
            auto it2 = O[(loc+N_sites-1)%N_sites].find({qPrev, q});
            assert(it2 != O[(loc+N_sites-1)%N_sites].end());
            std::cout << "\t\tQuantum number block {" << Sym::format<Symmetry>(qPrev) << "} -> {" << Sym::format<Symmetry>(q) << "}: Create a new column of operators (number of rows: " << dimPrev << ")" << std::endl;
            OperatorType zeroOp = OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum());
            for(std::size_t row=0; row<dimPrev; ++row)
            {
                (it2->second)[row].push_back(zeroOp);
            }
        }
        for(const auto& [qNext, dimNext] : auxdim[(loc+1)%N_sites])
        {
            auto it2 = O[loc%N_sites].find({q, qNext});
            assert(it2 != O[loc%N_sites].end());
            std::cout << "\t\tQuantum number block {" << Sym::format<Symmetry>(q) << "} -> {" << Sym::format<Symmetry>(qNext) << "}: Create a new row of operators (number of columns: " << dimNext << ")" << std::endl;
            std::vector<OperatorType> zeroOps(dimNext, OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum()));
            (it2->second).push_back(zeroOps);
        }
        if(!OPEN_BC and N_sites == 1)
        {
            std::cout << "\t\tWith respect to VUMPS: Add the corner operator" << std::endl;
            OperatorType zeroOp = OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum());
            auto it2 = O[0].find({q,q});
            assert(it2 != O[0].end());
            (it2->second)[(it2->second).size()-1].push_back(zeroOp);
        }
        (it->second)++;
    }
    else
    {
        std::cout << " New counter started" << std::endl;
        for(const auto& [qPrev, dimPrev] : auxdim[(loc+N_sites-1)%N_sites])
        {
            std::cout << "\t\tQuantum number block {" << Sym::format<Symmetry>(qPrev) << "} -> {" << Sym::format<Symmetry>(q) << "}: Block created with one column of operators (number of rows: " << dimPrev << ")" << std::endl;
            std::vector<OperatorType> temp(1, OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum()));
            std::vector<std::vector<OperatorType>> temp2(dimPrev, temp);
            O[(loc+N_sites-1)%N_sites].insert({{qPrev, q}, temp2});
        }
        for(const auto& [qNext, dimNext] : auxdim[(loc+1)%N_sites])
        {
            std::cout << "\t\tQuantum number block {" << Sym::format<Symmetry>(q) << "} -> {" << Sym::format<Symmetry>(qNext) << "}: Block created with one row of operators (number of columns: " << dimNext << ")" << std::endl;
            std::vector<OperatorType> temp(dimNext, OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum()));
            std::vector<std::vector<OperatorType>> temp2(1, temp);
            O[loc%N_sites].insert({{q, qNext}, temp2});
        }
        if(!OPEN_BC and N_sites == 1)
        {
            std::cout << "\t\tWith respect to VUMPS: Add the corner operator" << std::endl;
            std::vector<OperatorType> temp_col(1, OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum()));
            std::vector<std::vector<OperatorType>> temp(1,temp_col);
            O[0].insert({{q,q},temp});
        }
        auxdim[loc%N_sites].insert({q, 1ul});
    }
    if(!OPEN_BC and loc%N_sites == 0)
    {
        auto it3 = auxdim[N_sites].find(q);
        if(it3 != auxdim[N_sites].end())
        {
            std::cout << "\t\tWith respect to VUMPS: Also done at right edge" << std::endl;
            (it3->second)++;
        }
        else
        {
            std::cout << "\t\tWith respect to VUMPS: Also done at right edge" << std::endl;
            auxdim[N_sites].insert({q, 1ul});
        }
    }
}



template<typename Symmetry, typename Scalar> std::size_t MpoTerms<Symmetry,Scalar>::
get_auxdim(const std::size_t loc, const qType& q) const
{
    std::size_t loc_eff;
    if(!OPEN_BC)
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
save_label(const std::size_t loc, const std::string& label)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    if(label != "")
    {
        info[loc].push_back(label);
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
                            (it->second)[row][col] *= factor;
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
                Id.label = "id";
                push(loc, {Id}, offset);
            }
        }
    }
}

template<typename Symmetry, typename Scalar> template<typename OtherScalar> MpoTerms<Symmetry, OtherScalar> MpoTerms<Symmetry,Scalar>::
cast()
{
    MpoTerms<Symmetry, OtherScalar> other(N_sites, OPEN_BC);
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
                        (it->second)[row][col].template cast<OtherScalar>();
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
    
    std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>> O_new(N_sites);
    std::vector<std::unordered_map<qType, std::size_t>> auxdim_new(N_sites+1);
    
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
            for(auto& [qs_key, ops] : O[loc])
            {

                qType qIn = std::get<0>(qs_key);
                qType qOut = std::get<1>(qs_key);
                for(std::size_t n=0; n<symmetries_to_transform.size(); ++n)
                {
                        qIn[symmetries_to_transform[n]] *= length;
                        qOut[symmetries_to_transform[n]] *= length;
                }
                for(std::size_t row=0; row<ops.size(); ++row)
                {
                    for(std::size_t col=0; col<ops[row].size(); ++col)
                    {
                        for(std::size_t n=0; n<symmetries_to_transform.size(); ++n)
                        {
                            ops[row][col].Q[n] *= length;
                        }
                    }
                }
                O_new[loc].insert({{qIn,qOut},ops});
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
        std::cout << "Bases have been transformed by {" << Sym::format<Symmetry>(qShift) << "}" << std::endl;

        GOT_W = false;
        GOT_QAUX = false;
        GOT_QOP = false;
        bool UPDATE_SQUARE = GOT_SQUARE;
        GOT_SQUARE = false;
        calc(UPDATE_SQUARE);
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
                                
                                for(const auto &qOp_merge : qOp_merges)
                                {
                                    if(!qAux[loc+1].find(qOp_merge))
                                    {
                                        continue;
                                    }
                                    
                                    auto qPhys_tops = Symmetry::reduceSilent(qPhys[loc][n_lefttop], qPhys[loc+1][n_righttop]);
                                    auto qPhys_bottoms = Symmetry::reduceSilent(qPhys[loc][n_leftbottom], qPhys[loc+1][n_rightbottom]);
                                    for(const auto &qPhys_top : qPhys_tops)
                                    {
                                        for(const auto &qPhys_bottom : qPhys_bottoms)
                                        {
                                            auto qTensor_top = make_tuple(qPhys[loc][n_lefttop], n_lefttop, qPhys[loc+1][n_righttop], n_righttop, qPhys_top);
                                            std::size_t n_top = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qTensor_top));
                                            
                                            auto qTensor_bottom = make_tuple(qPhys[loc][n_leftbottom], n_leftbottom, qPhys[loc+1][n_rightbottom], n_rightbottom, qPhys_bottom);
                                            std::size_t n_bottom = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qTensor_bottom));

                                            
                                            // tensor product of the MPO operators in the physical space
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
    GOT_SQUARE = false;
}

template<typename Symmetry, typename Scalar> MpoTerms<Symmetry,Scalar> MpoTerms<Symmetry,Scalar>::
prod(const MpoTerms<Symmetry,Scalar>& bottom, const MpoTerms<Symmetry,Scalar>& top, const qType& Qtot)
{
    typedef typename Symmetry::qType qType;
    typedef SiteOperator<Symmetry, Scalar> OperatorType;
    qType qVac = Symmetry::qvacuum();
    
    std::cout << "Starting multiplication of two MPOs " << bottom.get_name() << "*" << top.get_name() << " to quantum number {" << Sym::format<Symmetry>(Qtot) << "}" << std::endl;
    
    assert(bottom.is_finalized() and top.is_finalized() and "Error: Multiplying non-finalized MPOs");
    assert(bottom.size() == top.size() and "Error: Multiplying two MPOs of different size");
    assert(bottom.get_boundary_condition() == top.get_boundary_condition() and "Error: Multiplying two MPOs with different boundary conditions");
    bool OBC = bottom.get_boundary_condition();
    
    std::vector<qType> Qtots = Symmetry::reduceSilent(bottom.get_qTot(), top.get_qTot());
    auto it = std::find(Qtots.begin(), Qtots.end(), Qtot);
    assert(it != Qtots.end() and "Cannot multiply these two operators to an operator with target quantum number");
    
    std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>> O_bottom, O_top, O_out;
    std::vector<Qbasis<Symmetry>> qAux_bottom, qAux_top, qAux_out;
    std::vector<std::vector<qType>> qPhys, qPhys_check;
    std::vector<std::vector<qType>> qOp_bottom, qOp_top;
    
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

    std::size_t pos_qTot_out = 0;
    if(Qtot == qVac)
    {
        pos_qTot_out = 1;
    }
    
    std::vector<std::size_t> row_qVac(N_sites);
    std::vector<std::size_t> col_qVac(N_sites);
    std::vector<std::size_t> row_qTot(N_sites);
    std::vector<std::size_t> col_qTot(N_sites);

    
    qAux_out[0] = qAux_bottom[0].combine(qAux_top[0]);
    
    
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t hilbert_dimension = qPhys[loc].size();
        
        assert(hilbert_dimension != 0 and "Not all Hilbert space dimensions have been set!");
        assert(hilbert_dimension == qPhys_check[loc].size() and "Local Hilbert space dimensions do not match!");
        
        qAux_out[loc+1] = qAux_bottom[loc+1].combine(qAux_top[loc+1]);
        for(auto& entry_left : qAux_out[loc])
        {
            for(auto& entry_right : qAux_out[loc+1])
            {
                std::size_t rows = std::get<2>(entry_left).size();
                std::size_t cols = std::get<2>(entry_right).size();
                qType Qin = std::get<0>(entry_left);
                qType Qout = std::get<0>(entry_right);
                std::vector<OperatorType> temp_row(cols, OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(hilbert_dimension, hilbert_dimension).sparseView(),Symmetry::qvacuum()));
                std::vector<std::vector<OperatorType>> temp(rows, temp_row);
                O_out[loc].insert({{Qin,Qout},temp});
            }
        }

        std::cout << "Lattice site " << loc << ":" << std::endl;
        std::vector<qType> Qins = qAux_out[loc].qs();
        for(const auto& Qin : Qins)
        {
            std::size_t total_rows = qAux_out[loc].inner_dim(Qin);
            std::vector<qType> qins_bottom = qAux_bottom[loc].qs();
            for(const auto& qin_bottom : qins_bottom)
            {
                std::size_t rows_bottom = qAux_bottom[loc].inner_dim(qin_bottom);
                std::vector<qType> qins_top = qAux_top[loc].qs();
                for(const auto& qin_top : qins_top)
                {
                    std::size_t rows_top = qAux_top[loc].inner_dim(qin_top);
                    if(std::array<qType,3> qCheck = {qin_bottom,qin_top,Qin}; !Symmetry::validate(qCheck))
                    {
                        continue;
                    }
                    std::cout << "\tQin = {" << Sym::format<Symmetry>(Qin) << "} can be reached by {" << Sym::format<Symmetry>(qin_bottom) << "} + {" << Sym::format<Symmetry>(qin_top) << "}" << std::endl;
                    std::vector<qType> Qouts = qAux_out[loc+1].qs();
                    for(const auto& Qout : Qouts)
                    {
                        if(OBC and (loc+1 == N_sites) and (Qout != Qtot))
                        {
                            continue;
                        }
                        std::size_t total_cols = qAux_out[loc+1].inner_dim(Qout);
                        std::vector<qType> qouts_bottom = qAux_bottom[loc+1].qs();
                        for(const auto& qout_bottom : qouts_bottom)
                        {
                            std::size_t cols_bottom = qAux_bottom[loc+1].inner_dim(qout_bottom);
                            std::vector<qType> qouts_top = qAux_top[loc+1].qs();
                            for(const auto& qout_top : qouts_top)
                            {
                                std::size_t cols_top = qAux_top[loc+1].inner_dim(qout_top);
                                if(std::array<qType,3> qCheck = {qout_bottom,qout_top,Qout}; !Symmetry::validate(qCheck))
                                {
                                    continue;
                                }
                                std::cout << "\t\tQout = {" << Sym::format<Symmetry>(Qout) << "} can be reached by {" << Sym::format<Symmetry>(qout_bottom) << "} + {" << Sym::format<Symmetry>(qout_top) << "}" << std::endl;
                                auto it_out = O_out[loc].find({Qin, Qout});
                                auto it_bottom = O_bottom[loc].find({qin_bottom, qout_bottom});
                                auto it_top = O_top[loc].find({qin_top, qout_top});
                                std::size_t in_pos = qAux_out[loc].leftAmount(Qin, {qin_bottom, qin_top});
                                std::size_t out_pos = qAux_out[loc+1].leftAmount(Qout, {qout_bottom, qout_top});
                                for(std::size_t row_bottom=0; row_bottom<rows_bottom; ++row_bottom)
                                {
                                    for(std::size_t row_top=0; row_top<rows_top; ++row_top)
                                    {
                                        std::size_t total_row = in_pos + row_bottom*rows_top + row_top;
                                        if(!OBC and (qin_bottom == qVac and row_bottom == bottom.pos_qVac) and (qin_top == qVac and row_top == top.pos_qVac) and total_row != row_qVac[loc])
                                        {
                                            row_qVac[loc] = total_row;
                                        }
                                        if(!OBC and (qin_bottom == bottom.qTot and row_bottom == bottom.pos_qTot) and (qin_top == top.qTot and row_top == top.pos_qTot) and total_row != row_qTot[loc])
                                        {
                                            row_qTot[loc] = total_row;
                                        }
                                        
                                        for(std::size_t col_bottom=0; col_bottom<cols_bottom; ++col_bottom)
                                        {
                                            for(std::size_t col_top=0; col_top<cols_top; ++col_top)
                                            {
                                                
                                                OperatorType& op_bottom = (it_bottom->second)[row_bottom][col_bottom];
                                                OperatorType& op_top = (it_top->second)[row_top][col_top];
                                                std::size_t total_col = out_pos + col_bottom*cols_top + col_top;
                                                
                                                if(!OBC and (qout_bottom == qVac and col_bottom == bottom.pos_qVac) and (qout_top == qVac and col_top == top.pos_qVac) and total_col != col_qVac[loc])
                                                {
                                                    col_qVac[loc] = total_col;
                                                }
                                                
                                                if(!OBC and (qout_bottom == bottom.qTot and col_bottom == bottom.pos_qTot) and (qout_top == top.qTot and col_top == top.pos_qTot) and total_col != col_qTot[loc])
                                                {
                                                    col_qTot[loc] = total_col;
                                                }
                                                
                                                if(op_bottom.data.norm() < ::mynumeric_limits<double>::epsilon() or op_top.data.norm() < :: mynumeric_limits<double>::epsilon())
                                                {
                                                    continue;
                                                }

                                                
                                                OperatorType& op_out = (it_out->second)[total_row][total_col];
                                                std::vector<qType> qOp_out = Symmetry::reduceSilent(op_bottom.Q, op_top.Q);
                                                for(const auto& Qop : qOp_out)
                                                {
                                                    if(std::array<qType,3> qCheck = {Qin,Qop,Qout}; !Symmetry::validate(qCheck))
                                                    {
                                                        continue;
                                                    }
                                                    std::cout << "\t\t\tBlock {" << Sym::format<Symmetry>(Qin) << "} -> {" << Sym::format<Symmetry>(Qout) << "},\tPosition [" << total_row << "|" << total_col << "], Operator " << op_top.label << "*" << op_bottom.label << " with quantum number {" << Sym::format<Symmetry>(Qop) << "}" << std::endl;
                                                    op_out.label = op_top.label+"*"+op_bottom.label;
                                                    op_out.Q = Qop;
                                                    Scalar factor_merge = Symmetry::coeff_tensorProd(qin_bottom, qin_top, Qin, op_bottom.Q, op_top.Q, op_out.Q, qout_bottom, qout_top, Qout);
                                                    if(std::abs(factor_merge) < ::mynumeric_limits<double>::epsilon())
                                                    {
                                                        continue;
                                                    }
                                                    for(std::size_t n_bottom=0; n_bottom<hilbert_dimension; ++n_bottom)
                                                    {
                                                        for(std::size_t n_top=0; n_top<hilbert_dimension; ++n_top)
                                                        {
                                                            if(std::array<qType,3> qCheck = {qPhys[loc][n_bottom], op_out.Q, qPhys[loc][n_top]}; !Symmetry::validate(qCheck))
                                                            {
                                                                continue;
                                                            }
                                                            //std::cout << "\t\t\tmat(" << n_top << "," << n_bottom << ") = 0";
                                                            
                                                            Scalar val = 0;
                                                            for(std::size_t n_middle=0; n_middle<hilbert_dimension; ++n_middle)
                                                            {
                                                                if(std::array<qType,3> qCheck = {qPhys[loc][n_bottom], op_bottom.Q, qPhys[loc][n_middle]}; !Symmetry::validate(qCheck))
                                                                {
                                                                    continue;
                                                                }
                                                                if(std::array<qType,3> qCheck = {qPhys[loc][n_middle], op_top.Q, qPhys[loc][n_top]}; !Symmetry::validate(qCheck))
                                                                {
                                                                    continue;
                                                                }
                                                                if(std::abs(op_top.data.coeffRef(n_top, n_middle)) < ::mynumeric_limits<double>::epsilon() or std::abs(op_bottom.data.coeffRef(n_middle, n_bottom)) < ::mynumeric_limits<double>::epsilon())
                                                                {
                                                                    continue;
                                                                }
                                                                Scalar factor_check = Symmetry::coeff_prod(qPhys[loc][n_top], op_top.Q, qPhys[loc][n_middle], op_bottom.Q, qPhys[loc][n_bottom], op_out.Q);
                                                                if(std::abs(factor_check) < ::mynumeric_limits<double>::epsilon())
                                                                {
                                                                    continue;
                                                                }
                                                                val += op_top.data.coeffRef(n_top, n_middle)*op_bottom.data.coeffRef(n_middle, n_bottom) * factor_check * factor_merge;
                                                                //std::cout << " + " << op_top.data.coeffRef(n_top, n_middle) << "*" << op_bottom.data.coeffRef(n_middle, n_bottom) << "*" << factor_check << "*" << factor_merge;
                                                            }
                                                            if(std::abs(val) < ::mynumeric_limits<double>::epsilon())
                                                            {
                                                                //std::cout << " = 0" << std::endl;
                                                                continue;
                                                            }
                                                            op_out.data.coeffRef(n_top, n_bottom) = val;
                                                            //std::cout << " = " << val << std::endl;
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
    
    if(!OBC)
    {
        VUMPS_prod_swap(O_out, row_qVac, col_qVac, row_qTot, col_qTot);
    }
    MpoTerms<Symmetry,Scalar> out(1);
    out.reconstruct(O_out, qAux_out, qPhys, true, OBC, Qtot);
    out.set_name(top.get_name()+"*"+bottom.get_name());
    out.compress();
    return out;
}

template<typename Symmetry, typename Scalar> MpoTerms<Symmetry,Scalar> MpoTerms<Symmetry,Scalar>::
sum(const MpoTerms<Symmetry,Scalar>& bottom, const MpoTerms<Symmetry,Scalar>& top)
{
    typedef typename Symmetry::qType qType;
    typedef SiteOperator<Symmetry, Scalar> OperatorType;
    
    std::cout << "Starting addition of two MPOs " << bottom.get_name() << "+" << top.get_name() << std::endl;
    
    assert(bottom.is_finalized() and top.is_finalized() and "Error: Adding non-finalized MPOs");
    assert(bottom.size() == top.size() and "Error: Adding two MPOs of different size");

    assert(bottom.get_boundary_condition() == top.get_boundary_condition() and "Error: Adding two MPOs with different boundary conditions");
    bool OBC = bottom.get_boundary_condition();
    
    std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>> O_bottom, O_top, O_out;
    std::vector<Qbasis<Symmetry>> qAux_bottom, qAux_top, qAux_out;
    std::vector<std::vector<qType>> qPhys, qPhys_check;
    std::vector<std::vector<qType>> qOp_bottom, qOp_top;
    
    assert(bottom.get_qTot() == top.get_qTot() and "Addition only possible for MPOs with the same total quantum number.");
    qType Qtot = bottom.get_qTot();
    
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
    std::vector<std::map<qType,std::size_t>> top_amount(N_sites+1);
    
    for(std::size_t loc=0; loc<=N_sites; ++loc)
    {
        for(const auto& entry : qAux_top[loc])
        {
            qType q = std::get<0>(entry);
            std::size_t deg = std::get<2>(entry).size();
            auxdim[loc].insert({q,deg});
            
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
        std::cout << "Lattice site " << loc << std::endl;
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
                std::cout << "\tqIn = {" << Sym::format<Symmetry>(qIn) << "} and qOut = {" << Sym::format<Symmetry>(qOut) << "} is a " << rows << "x" << cols << "-matrix:" << std::endl;
                std::vector<OperatorType> temp_row(cols, OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(hilbert_dimension, hilbert_dimension).sparseView(),Symmetry::qvacuum()));
                std::vector<std::vector<OperatorType>> O_temp(rows, temp_row);
                auto it_top = O_top[loc].find({qIn,qOut});
                auto it_bottom = O_bottom[loc].find({qIn,qOut});
                if(it_top != O_top[loc].end())
                {
                    const std::vector<std::vector<OperatorType>>& O_top_temp = it_top->second;
                    std::size_t top_rows = O_top_temp.size();
                    std::size_t top_cols = O_top_temp[0].size();
                    std::cout << "\t\t Block exists in top MPO and is written from [0|0] to [" << top_rows-1 << "|" << top_cols-1 << "]";
                    for(std::size_t row=0; row<top_rows; ++row)
                    {
                        for(std::size_t col=0; col<top_cols; ++col)
                        {
                            O_temp[row][col] = O_top_temp[row][col];
                        }
                    }
                }
                else
                {
                    std::cout << "\t\t Block does not exist in top MPO";
                }
                if(it_bottom != O_bottom[loc].end())
                {
                    const std::vector<std::vector<OperatorType>>& O_bottom_temp = it_bottom->second;
                    std::size_t bottom_rows = O_bottom_temp.size();
                    std::size_t bottom_cols = O_bottom_temp[0].size();
                    std::cout << "\t|\t Block exists in bottom MPO and is written from [" << row_start << "|" << col_start << "] to [" << row_start+bottom_rows-1 << "|" << col_start+bottom_cols-1 << "]" << std::endl;
                    for(std::size_t row=0; row<bottom_rows; ++row)
                    {
                        for(std::size_t col=0; col<bottom_cols; ++col)
                        {
                            O_temp[row_start+row][col_start+col] = O_bottom_temp[row][col];
                        }
                    }
                }
                else
                {
                    std::cout << "\t|\t Block does not exist in bottom MPO" << std::endl;
                }
                O_out[loc].insert({{qIn,qOut},O_temp});
            }
        }
        std::cout << "Done: Lattice site " << loc << std::endl;
    }
    
    if(OBC)
    {
        qAux_out[0].clear();
        qAux_out[0].push_back(Symmetry::qvacuum(), 1);
        qAux_out[N_sites].clear();
        qAux_out[N_sites].push_back(Qtot, 1);
        
        OperatorType zeroOp(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(1,1).sparseView(),Symmetry::qvacuum());
        
        std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>> O_first, O_last;

        for(const auto& entry : qAux_out[1])
        {
            qType qOut = std::get<0>(entry);
            std::size_t cols = std::get<2>(entry).size();
            
            std::vector<OperatorType> ops_row(cols, zeroOp);
            std::vector<std::vector<OperatorType>> ops(1, ops_row);
            
            auto it = O_out[0].find({Symmetry::qvacuum(), qOut});
            assert(it != O_out[0].end());
            for(std::size_t col=0; col<cols; ++col)
            {
                OperatorType& upper = (it->second)[0][col];
                OperatorType& lower = (it->second)[1][col];
                if(upper.data.norm() > ::mynumeric_limits<double>::epsilon())
                {
                    if(lower.data.norm() > ::mynumeric_limits<double>::epsilon())
                    {
                        ops[0][col] = upper + lower;
                    }
                    else
                    {
                        ops[0][col] = upper;
                    }
                }
                else
                {
                    ops[0][col] = lower;
                }
            }
            O_first.insert({{Symmetry::qvacuum(),qOut}, ops});
        }
        for(const auto& entry : qAux_out[N_sites-1])
        {
            qType qIn = std::get<0>(entry);
            std::size_t rows = std::get<2>(entry).size();
            
            std::vector<OperatorType> ops_row(1, zeroOp);
            std::vector<std::vector<OperatorType>> ops(rows, ops_row);
            
            auto it = O_out[N_sites-1].find({qIn, Qtot});
            assert(it != O_out[N_sites-1].end());
            for(std::size_t row=0; row<rows; ++row)
            {
                OperatorType& left = (it->second)[row][0];
                OperatorType& right = (it->second)[row][1];

                if(left.data.norm() > ::mynumeric_limits<double>::epsilon())
                {
                    if(right.data.norm() > ::mynumeric_limits<double>::epsilon())
                    {
                        ops[row][0] = left + right;
                    }
                    else
                    {
                        ops[row][0] = left;
                    }
                }
                else
                {
                    ops[row][0] = right;
                }
            }
            O_last.insert({{qIn, Qtot}, ops});
        }
        O_out[0] = O_first;
        O_out[N_sites-1] = O_last;
    }

    MpoTerms<Symmetry,Scalar> out(1);
    out.reconstruct(O_out, qAux_out, qPhys, true, OBC, Qtot);
    out.set_name(top.get_name()+"+"+bottom.get_name());
    out.compress();
    return out;
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
VUMPS_prod_swap(std::vector<std::unordered_map<std::array<qType, 2>, std::vector<std::vector<OperatorType>>>>& O_out, std::vector<std::size_t>& row_qVac, std::vector<std::size_t>& col_qVac, std::vector<std::size_t>& row_qTot, std::vector<std::size_t>& col_qTot)
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
                std::vector<OperatorType> temp = ops[0];
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
                    OperatorType temp = ops[row][0];
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
                std::vector<OperatorType> temp = ops[1];
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
                    OperatorType temp = ops[row][1];
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
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        auxdim[loc].insert({qVac,1});
        OperatorType op = OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(qPhys[loc].size(),qPhys[loc].size()).sparseView(),qVac);
        op.label = "id";
        std::vector<OperatorType> temp(1,op);
        std::vector<std::vector<OperatorType>> Oloc(1,temp);
        O[loc].insert({{qVac,qVac},Oloc});
    }
    auxdim[N_sites].insert({qVac,1});
    FINALIZED = true;
    calc(false);
}

template<typename Symmetry, typename Scalar> void MpoTerms<Symmetry,Scalar>::
set_Zero()
{
    got_update();
    O.clear();
    O.resize(N_sites);
    auxdim.clear();
    auxdim.resize(N_sites+1);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        auxdim[loc].insert({qVac,1});
        OperatorType op = OperatorType(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(qPhys[loc].size(),qPhys[loc].size()).sparseView(),qVac);
        op.label = "Zero";
        std::vector<OperatorType> temp(1,op);
        std::vector<std::vector<OperatorType>> Oloc(1,temp);
        O[loc].insert({{qVac,qVac},Oloc});
    }
    auxdim[N_sites].insert({qVac,1});
    FINALIZED = true;
    calc(false);
}

template<typename Symmetry, typename Scalar> std::vector<std::pair<typename Symmetry::qType, std::size_t>> MpoTerms<Symmetry,Scalar>::
VUMPS_base_order() const
{
    assert(!OPEN_BC);
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
            if((it->second)[row][pos_qTot].data.norm() > ::mynumeric_limits<double>::epsilon())
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
#endif
