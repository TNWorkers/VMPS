#ifndef DMRG_HAMILTONIAN_TERMS
#define DMRG_HAMILTONIAN_TERMS

/// \cond
#include <vector>
#include <string>
/// \endcond

#include "SuperMatrix.h"
#include "numeric_limits.h" // from TOOLS
#include "tensors/SiteOperator.h"
#include "symmetry/qarray.h"

template<typename Symmetry, typename Scalar> class HamiltonianTerms
{
private:
    typedef SiteOperator<Symmetry,Scalar> OperatorType;
    typename Symmetry::qType qvac = Symmetry::qvacuum();
    
    /**
     *  Local terms of Hamiltonian, a simple vector
     *  Index structure (Lattice Site)
     */
    std::vector<OperatorType> local;
    
    /**
     *    Stores whether the local operator has been set yet, a simple vector
     *    Index structure (Lattice Site)
     */
    std::vector<bool> localSet;
    
    /**
     *    Incoming nearest-neighbour terms, a twofold vector
     *    Index structure (Lattice Site i, Index incoming operator at i)
     */
    std::vector<std::vector<OperatorType>> tight_in;
    
    /**
     *    Outgoing nearest-neighbour terms, a twofold vector
     *    Index structure (Lattice Site i, Index outgoing operator at i)
     */
    std::vector<std::vector<OperatorType>> tight_out;
    
    /**
     *    Nearest-neighbour interactions, a vector of matrices
     *    Index structure (Lattice Site i, Index outgoing operator at i, Index incoming operator at i+1)
     */
    std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> tight_coupl;
    
    /**
     *    Incoming next-nearest-neighbour terms, a threefold vector
     *    Index structure (Lattice Site i, Index transfer operator at i-1, Index incoming operator at i)
     */
    std::vector<std::vector<std::vector<OperatorType>>> nextn_in;
    
    /**
     *    Outgoing next-nearest-neighbour terms, a threefold vector
     *    Index structure (Lattice Site i, Index transfer operator at i+1, Index outgoing operator at i)
     */
    std::vector<std::vector<std::vector<OperatorType>>> nextn_out;
    
    /**
     *    Transfer operators for next-nearest-neighbour interaction, a twofold vector
     *    Index structure (Lattice Site i, Index transfer operator at i)
     */
    std::vector<std::vector<OperatorType>> nextn_TransOps;
    
    /**
     *    Next-nearest-neighbour interactions, a twofold vector of matrices
     *    Index structure (Lattice Site i, transfer operator at i+1, Index outgoing operator at i, Index incoming operator at i+2)
     */
    std::vector<std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>> nextn_coupl;
    
    /**
     *    Local hilbert space dimensions, a simple vector
     *    Index structure (Lattice Site). Initialized with 0 and set whenever a operator is added at a given site
     */
    std::vector<int> hilbert_dimension;
    
    /**
     *    Number of lattice sites
     */
    std::size_t N_sites;
    
    /**
     *    All informations stored about the HamiltonianTerms, twofold vector
     *    Index structure (Lattice Site i, Index of information string)
     */
    std::vector<std::vector<std::string>> info;
    
    /**
     *    Compressed incoming nearest-neighbour terms, a twofold vector
     *    Index structure (Lattice Site i, Index incoming operator at i)
     */
    std::vector<std::vector<OperatorType>> tight_in_compressed;
    
    /**
     *    Compressed outgoing nearest-neighbour terms, a twofold vector
     *    Index structure (Lattice Site i, Index outgoing operator at i)
     */
    std::vector<std::vector<OperatorType>> tight_out_compressed;
    
    /**
     *    Compressed incoming next-nearest-neighbour terms, a threefold vector
     *    Index structure (Lattice Site i, Index transfer operator at i-1, Index incoming operator at i)
     */
    std::vector<std::vector<std::vector<OperatorType>>> nextn_in_compressed;
    
    /**
     *    Compressed outgoing next-nearest-neighbour terms, a threefold vector
     *    Index structure (Lattice Site i, Index transfer operator at i+1, Index outgoing operator at i)
     */
    std::vector<std::vector<std::vector<OperatorType>>> nextn_out_compressed;
    
    /**
     *    "Compressed" transfer operators for next-nearest-neighbour interaction, a threefold vector
     *    Index structure (Lattice Site i, Index transfer operator at i, Index repetition of operator)
     */
    std::vector<std::vector<std::vector<OperatorType>>> nextn_trans_compressed;
    
    /**
     *    Stores, whether the compressed Term vectors are up to date
     */
    bool COMPRESSED;
    
    /**
     *  A given name for the HamiltonianTerms, such as Heisenberg
     */
    std::string label="";
    
    /**
     *  Stores whether to use periodic boundary conditions or not
     */
    bool OPEN_BC;
    
    /**
     *  Takes the plain interaction vectors and compresses them by singular value decomposition
     */
    void compress();
    
public:
    /**
     *  Default constructor, does nothing
     */
    HamiltonianTerms() {HamiltonianTerms(0);}
    
    string print_info() const
    {
    	stringstream ss;
    	for (size_t l=0; l<N_sites; ++l)
    	{
			for (size_t i=0; i<info[l].size(); ++i)
			{
				ss << info[l][i] << "\t";
			}
			ss << "tight_in=" << tight_in[l].size() << "\t"
			   << "tight_out=" << tight_out[l].size() << "\t"
			   << "nextn_in=" << nextn_in[l].size() << "\t"
			   << "nextn_out=" << nextn_out[l].size();
			ss << endl;
    	}
    	return ss.str();
    }
    
    /**
     *  Constructor
     *  @param L    Lattice size
     *  @param bc   Open boundary conditions
     */
    HamiltonianTerms(std::size_t L, bool bc = true);
    
    /**
     *  Adds a new local interaction to the HamiltonianTerms
     *  @param loc      Lattice site
     *  @param Op       SiteOperator acting on the local Hilbert space of site \p loc
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     */
    void push_local(std::size_t loc, Scalar lambda, OperatorType Op);
    
    /**
     *  Adds a new nearest-neighbour interaction to the HamiltonianTerms
     *  @param loc      Lattice site of first site
     *  @param Op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param Op2      SiteOperator acting on the local Hilbert space of site \p loc+1
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     */
    void push_tight(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Op2);
    
    /**
     *  Adds a new next-nearest-neighbour interaction to the HamiltonianTerms
     *  @param loc      Lattice site of first site
     *  @param Op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param Trans    SiteOperator acting as transfer operator on the local Hilbert space of site \p loc+1
     *  @param Op2      SiteOperator acting on the local Hilbert space of site \p loc+2
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     */
    void push_nextn(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Trans, OperatorType Op2);

	/**
	 * Const reference to the local terms at lattice site \p loc.
	 */
	OperatorType const& localOps(std::size_t loc) const {return local[loc];}

	/**
	 * Const reference to the tight_in terms at lattice site \p loc.
	 */
	std::vector<OperatorType> const& tight_inOps(std::size_t loc) const {return tight_in[loc];}

	/**
	 * Const reference to the tight_out terms at lattice site \p loc.
	 */
	std::vector<OperatorType> const& tight_outOps(std::size_t loc) const {return tight_out[loc];}

	/**
	 * Const reference to the nextn_in terms at lattice site \p loc.
	 */
	std::vector<std::vector<OperatorType>> const& nextn_inOps(std::size_t loc) const {return nextn_in[loc];}

	/**
	 * Const reference to the nextn_out terms at lattice site \p loc.
	 */
	std::vector<std::vector<OperatorType>> const& nextn_outOps(std::size_t loc) const {return nextn_out[loc];}

	/**
	 * Const reference to the nextn terms.
	 */
	vector<OperatorType> const& localOps() const {return local;}

    /**
     *  @param loc      Lattice site
     *  @param label    Information
     */
    void save_label(std::size_t loc, const std::string &label);
    
    /**
     *  @param label    Name to be given to this instance of HamiltonianTerms
     */
    void set_name(const std::string &label_in) {label = label_in;}
    
    /**
     *  @return A vector of formatted strings that contain information about the HamiltonianTerms. Zeroth entry = name.
     */
    std::vector<std::string> get_info() const;
    
    /**
     *  @param loc  Lattice site
     *  @return The dimension of the local Hilbert space. If not set yet, this will return 0
     */
    std::size_t Hilbert_dimension(std::size_t loc) const;
    
    /**
     *  Constructs a vector of SuperMatrix from the compressed interactions. Compresses the interaction if necessary.
     *  Resizes the first and last SuperMatrix to row/column for open boundary conditions
     */
    std::vector<SuperMatrix<Symmetry,Scalar>> construct_Matrix();
    
    /**
     * @return  Given name of this instance of HamiltonianTerms
     */
    std::string name() const {return label;}
    
    
    /**
     *  Scales all interactions by a given factor
     *  @param factor   The factor to scale the interactions with
     *  @param offset
     */
    void scale(double factor, double offset=0.);
    
    /**
     *  Returns the number of lattice sites
     *  @return N_sites
     */
    std::size_t size() const {return N_sites;}
    
    /**
     *  Casts instance of HamiltonianTerms to an instance with another scalar type
     */
    template<typename OtherScalar> HamiltonianTerms<Symmetry, OtherScalar> cast();
};

template<typename Symmetry, typename Scalar> HamiltonianTerms<Symmetry,Scalar>::
HamiltonianTerms(std::size_t L, bool BC) : N_sites(L)
{
    info.resize(N_sites);
    hilbert_dimension.resize(N_sites);
    local.resize(N_sites);
    localSet.resize(N_sites);
    tight_in.resize(N_sites);
    tight_out.resize(N_sites);
    tight_coupl.resize(N_sites);
    nextn_in.resize(N_sites);
    nextn_out.resize(N_sites);
    nextn_TransOps.resize(N_sites);
    nextn_coupl.resize(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        hilbert_dimension[loc] = 0;
        tight_coupl[loc].resize(0,0);
        localSet[loc] = false;
    }
    OPEN_BC = BC;
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
save_label(std::size_t loc, const std::string &label)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    if(label!="")
    {
        info[loc].push_back(label);
    }
}

template<typename Symmetry, typename Scalar> std::vector<std::string> HamiltonianTerms<Symmetry,Scalar>::
get_info() const
{
    std::vector<std::string> res(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::stringstream ss;
        std::copy(info[loc].begin(), info[loc].end()-1, std::ostream_iterator<std::string>(ss,","));
        ss << info[loc].back();
        
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

template<typename Symmetry, typename Scalar> std::size_t HamiltonianTerms<Symmetry,Scalar>::
Hilbert_dimension(std::size_t loc) const
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    return hilbert_dimension[loc];
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
push_local(std::size_t loc, Scalar lambda, OperatorType Op)
{
    if(lambda != 0.)
    {
        assert(loc < N_sites and "Chosen lattice site out of bounds");
        assert(Op.Q == qvac and "Local operator is not a singlet");
        if(!localSet[loc])
        {
            localSet[loc] = true;
            local[loc] = lambda*Op;
            hilbert_dimension[loc] = Op.data.rows();
        }
        else
        {
            assert(hilbert_dimension[loc] == Op.data.rows() and "Dimensions of operator and local Hilbert space do not match!");
            local[loc] += lambda*Op;
        }
    }
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
push_tight(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Op2)
{
    if(lambda != 0.)
    {
        assert(loc < N_sites and "Chosen lattice site out of bounds");
        assert((!OPEN_BC or loc+1 < N_sites) and "Chosen lattice site out of bounds");
        assert(hilbert_dimension[loc] == 0 or hilbert_dimension[loc] == Op1.data.rows() and "Dimensions of first operator and local Hilbert space do not match");
        assert(hilbert_dimension[(loc+1)%N_sites] == 0 or hilbert_dimension[(loc+1)%N_sites] == Op2.data.rows() and 
               "Dimensions of second operator and local Hilbert space do not match");
        COMPRESSED = false;
        std::ptrdiff_t firstit = std::distance(tight_out[loc].begin(), find(tight_out[loc].begin(), tight_out[loc].end(), Op1));
        if(firstit >= tight_out[loc].size())    // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
        {
            tight_out[loc].push_back(Op1);
            tight_coupl[loc].conservativeResize(tight_out[loc].size(), tight_in[(loc+1)%N_sites].size());
            tight_coupl[loc].bottomRows(1).setZero();
            
            hilbert_dimension[loc] = Op1.data.rows();
        }
        std::ptrdiff_t secondit = std::distance(tight_in[(loc+1)%N_sites].begin(), find(tight_in[(loc+1)%N_sites].begin(), tight_in[(loc+1)%N_sites].end(), Op2));
        if(secondit >= tight_in[(loc+1)%N_sites].size())    // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
        {
            tight_in[(loc+1)%N_sites].push_back(Op2);
            tight_coupl[loc].conservativeResize(tight_out[loc].size(), tight_in[(loc+1)%N_sites].size());
            tight_coupl[loc].rightCols(1).setZero();
            hilbert_dimension[(loc+1)%N_sites] = Op2.data.rows();
        }
        tight_coupl[loc](firstit, secondit) += lambda;
    }
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
push_nextn(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Trans, OperatorType Op2)
{
	if (lambda != 0.)
	{
		assert(loc < N_sites and "Chosen lattice site out of bounds");
		assert((!OPEN_BC || loc+2 < N_sites) and "Chosen lattice site out of bounds");
		assert(hilbert_dimension[loc] == 0 or hilbert_dimension[loc] == Op1.data.rows() and 
		       "Dimensions of first operator and local Hilbert space do not match");
		assert(hilbert_dimension[(loc+1)%N_sites] == 0 or hilbert_dimension[(loc+1)%N_sites] == Trans.data.rows() and 
		       "Dimensions of transfer operator and local Hilbert space do not match");
		assert(hilbert_dimension[(loc+2)%N_sites] == 0 or hilbert_dimension[(loc+2)%N_sites] == Op2.data.rows() and 
		       "Dimensions of second operator and local Hilbert space do not match");
		//assert(Trans.Q == qvac and "Transfer operator is not a singlet");
		COMPRESSED = false;
		
		std::ptrdiff_t transit = std::distance(nextn_TransOps[(loc+1)%N_sites].begin(), 
		                                       find(nextn_TransOps[(loc+1)%N_sites].begin(),
		                                            nextn_TransOps[(loc+1)%N_sites].end(), Trans));
		// If the operator cannot be found, push it to the corresponding terms and create a new container for interactions mediated by this transfer operator
		if (transit >= nextn_TransOps[(loc+1)%N_sites].size())
		{
			nextn_TransOps[(loc+1)%N_sites].push_back(Trans);
			hilbert_dimension[(loc+1)%N_sites] = Trans.data.rows();
			Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(1,1);
			nextn_coupl[(loc+1)%N_sites].push_back(matrix);
			std::vector<OperatorType> temp;
			nextn_out[loc].push_back(temp);
			nextn_in[(loc+2)%N_sites].push_back(temp);
		}
		
		std::ptrdiff_t firstit = std::distance(nextn_out[loc][transit].begin(), 
		                                       find(nextn_out[loc][transit].begin(), 
		                                            nextn_out[loc][transit].end(), Op1));
		// If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
		if (firstit >= nextn_out[loc][transit].size())
		{
			nextn_out[loc][transit].push_back(Op1);
			nextn_coupl[(loc+1)%N_sites][transit].conservativeResize(nextn_out[loc][transit].size(), nextn_in[(loc+2)%N_sites][transit].size());
			nextn_coupl[(loc+1)%N_sites][transit].bottomRows(1).setZero();
			hilbert_dimension[loc] = Op1.data.rows();
		}
		
		std::ptrdiff_t secondit = std::distance(nextn_in[(loc+2)%N_sites][transit].begin(), 
		                                        find(nextn_in[(loc+2)%N_sites][transit].begin(), 
		                                             nextn_in[(loc+2)%N_sites][transit].end(), Op2));
		// If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
		if (secondit >= nextn_in[(loc+2)%N_sites][transit].size())
		{
			nextn_in[(loc+2)%N_sites][transit].push_back(Op2);
			nextn_coupl[(loc+1)%N_sites][transit].conservativeResize(nextn_out[loc][transit].size(), nextn_in[(loc+2)%N_sites][transit].size());
			nextn_coupl[(loc+1)%N_sites][transit].rightCols(1).setZero();
			hilbert_dimension[(loc+2)%N_sites] = Op2.data.rows();
		}
		nextn_coupl[(loc+1)%N_sites][transit](firstit, secondit) += lambda;
	}
}

template<typename Symmetry, typename Scalar> std::vector<SuperMatrix<Symmetry, Scalar>> HamiltonianTerms<Symmetry,Scalar>::
construct_Matrix()
{
	if (!COMPRESSED) compress();
	std::vector<SuperMatrix<Symmetry,Scalar> > G;
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		SuperMatrix<Symmetry,Scalar> S;
		if (hilbert_dimension[loc] == 0) // Create a trivial SuperMatrix if no operator has been set.
		{
			hilbert_dimension[loc] = 1;
			OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
			S.set(2,2,hilbert_dimension[loc]);
			S(0,0) = Id;
			S(1,1) = Id;
		}
		else
		{
			// Stores the total number of transfer operators at lattice site loc
			std::size_t transfer = 0;
			// Stores the total number of next-nearest-neighbour interaction terms with loc-2
			std::size_t nextn_rows = 0;
			// Stores the total number of next-nearest-neighbour interaction terms with loc+2
			std::size_t nextn_cols = 0;
			for (std::size_t t=0; t<nextn_in_compressed[loc].size(); ++t)
			{
				nextn_rows += nextn_in_compressed[loc][t].size();
			}
			for (std::size_t t=0; t<nextn_out_compressed[loc].size(); ++t)
			{
				nextn_cols += nextn_out_compressed[loc][t].size();
			}
			for (std::size_t t=0; t<nextn_trans_compressed[loc].size(); ++t)
			{
				transfer += nextn_trans_compressed[loc][t].size();
			}
			
			std::size_t rows = 2 + tight_in_compressed[loc].size()  + nextn_rows + transfer; // Total number of rows
			std::size_t cols = 2 + tight_out_compressed[loc].size() + nextn_cols + transfer; // Total number of columns
//			cout << "tight_in_compressed[loc].size()=" << tight_in_compressed[loc].size() << ", nextn_rows=" << nextn_rows << ", transfer=" << transfer << endl;
//			cout << "tight_out_compressed[loc].size()=" << tight_out_compressed[loc].size() << ", nextn_rows=" << nextn_cols << ", transfer=" << transfer << endl;
//			cout << "rows=" << rows << ", cols=" << cols << endl;
			S.set(rows,cols,hilbert_dimension[loc]);
			OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
			std::size_t current = 0;
			S(current++,0) = Id; // Upper left corner: identity
			
			for (std::size_t i=0; i<tight_in_compressed[loc].size(); ++i)
			{
				// First column: Incoming tight-binding terms
				S(current++,0) = tight_in_compressed[loc][i]; 
			}
			for (std::size_t t=0; t<nextn_in_compressed[loc].size(); ++t)
			for (std::size_t i=0; i<nextn_in_compressed[loc][t].size(); ++i)
			{
//				cout << "loc=" << loc << ", r=" << current << "/last=" << rows-1 << ", c=" << 0 << "/last=" << cols-1
//				     << endl
//				     << "nextn_in_compressed=" << endl << MatrixXd(nextn_in_compressed[loc][t][i].data) << endl << endl;
				// First column: Incoming NNN terms, ordered w.r.t. their transfer operators
				S(current++,0) = nextn_in_compressed[loc][t][i];
			}
			
			// First column: A sufficient number of rows is skipped for the transfer operators
			
			current = 0;
			if (!localSet[loc]) // If no local interaction has been added, the local interaction becomes a dummy SiteOperator with correct dimension
			{
				local[loc] = 0*Id;
			}
			
			S(rows-1,current++) = local[loc]; // Lower left corner: Local interaction
			
			for (std::size_t t=0; t<tight_out_compressed[loc].size(); ++t)
			{
				S(rows-1, current++) = tight_out_compressed[loc][t]; //  Last row: Outgoing tight-binding terms
			}
			
			current += transfer; // Last row: A sufficient number of columns is skipped for the transfer operators
			
			for (std::size_t t=0; t<nextn_out_compressed[loc].size(); ++t)
			for (std::size_t i=0; i<nextn_out_compressed[loc][t].size(); ++i)
			{
//				cout << "loc=" << loc << ", r=" << rows-1 << "/last=" << rows-1 << ", c=" << current << "/last=" << cols-1
//				     << endl
//				     << "nextn_out_compressed=" << endl << MatrixXd(nextn_out_compressed[loc][t][i].data) << endl << endl;
				// Last row: Outgoing NNN terms, ordered w.r.t. their transfer operators
				S(rows-1,current++) = nextn_out_compressed[loc][t][i];
			}
			S(rows-1,cols-1) = Id; // Lower right corner: Identity
			
			std::size_t row_start = 1 + tight_in_compressed[loc].size() + nextn_rows; // Where does the block of transfer operators start?
			std::size_t col_start = 1 + tight_out_compressed[loc].size();
			
			current = 0;
			for (std::size_t t=0; t<nextn_trans_compressed[loc].size(); ++t)
			for (std::size_t i=0; i<nextn_trans_compressed[loc][t].size(); ++i)
			{
				// Since the interaction for each transfer operator is diagonal: only diagonal elements of the transfer block are set.
				S(row_start+current,col_start+current) = nextn_trans_compressed[loc][t][i];
//				cout << endl 
//				     << "loc=" << loc
//				     << ", row_start+current=" << row_start+current << "/last=" << rows-1 << ", col_start+current=" << col_start+current << "/last=" << cols-1
//				     << endl << MatrixXd(nextn_trans_compressed[loc][t][i].data) << endl << endl;
				current++;
			}
		}
		
		if (OPEN_BC and loc==0)
		{
			G.push_back(S.row(S.rows()-1));
		}
		else if (OPEN_BC and loc==N_sites-1)
		{
			G.push_back(S.col(0));
		}
		else
		{
			G.push_back(S);
		}
	}
	return G;
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
compress()
{
    tight_in_compressed.resize(N_sites);
    tight_out_compressed.resize(N_sites);
    nextn_in_compressed.resize(N_sites);
    nextn_out_compressed.resize(N_sites);
    nextn_trans_compressed.resize(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        nextn_in_compressed[loc].resize(nextn_in[loc].size());
        nextn_out_compressed[loc].resize(nextn_out[loc].size());
        nextn_trans_compressed[loc].resize(nextn_coupl[loc].size());
    }
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        if(tight_coupl[loc].rows() > tight_coupl[loc].cols())
        {
            tight_out_compressed[loc] = tight_out[loc] * tight_coupl[loc];
            tight_in_compressed[(loc+1)%N_sites] = tight_in[(loc+1)%N_sites];
        }
        else
        {
            tight_out_compressed[loc] = tight_out[loc];
            tight_in_compressed[(loc+1)%N_sites] = tight_coupl[loc] * tight_in[(loc+1)%N_sites];
        }
        for(std::size_t t=0; t<nextn_coupl[(loc+1)%N_sites].size(); ++t)
        {
            if(nextn_coupl[(loc+1)%N_sites][t].rows() > nextn_coupl[(loc+1)%N_sites][t].cols())
            {
                nextn_out_compressed[loc][t] = nextn_out[loc][t] * nextn_coupl[(loc+1)%N_sites][t];
                nextn_in_compressed[(loc+2)%N_sites][t] = nextn_in[(loc+2)%N_sites][t];
                
                for(std::size_t i=0; i<nextn_coupl[(loc+1)%N_sites][t].cols(); ++i)
                {
                    nextn_trans_compressed[(loc+1)%N_sites][t].push_back(nextn_TransOps[(loc+1)%N_sites][t]);
                }
            }
            else
            {
                nextn_out_compressed[loc][t] = nextn_out[loc][t];
                nextn_in_compressed[(loc+2)%N_sites][t] = nextn_coupl[(loc+1)%N_sites][t] * nextn_in[(loc+2)%N_sites][t];
                for(std::size_t i=0; i<nextn_coupl[(loc+1)%N_sites][t].rows(); ++i)
                {
                    nextn_trans_compressed[(loc+1)%N_sites][t].push_back(nextn_TransOps[(loc+1)%N_sites][t]);
                }
            }
        }
    }
    COMPRESSED = true;
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
scale (double factor, double offset)
{
    COMPRESSED = false;
    if (std::abs(factor-1.) > ::mynumeric_limits<double>::epsilon())
    {
        for (std::size_t loc=0; loc<N_sites; ++loc)
        {
            local[loc] *= factor;
            tight_coupl[loc] *= factor;
            for(std::size_t t=0; t<nextn_coupl[loc].size(); ++t)
            {
                nextn_coupl[loc][t] *= factor;
            }
        }
    }
    
    if (std::abs(offset) > ::mynumeric_limits<double>::epsilon())
    {
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            if(hilbert_dimension[loc] > 0)
            {
                SiteOperator<Symmetry,Scalar> Id;
                Id.data = Matrix<Scalar,Dynamic,Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView();
                push_local(offset, Id);
            }
        }
    }
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
HamiltonianTerms<Symmetry, OtherScalar> HamiltonianTerms<Symmetry,Scalar>::
cast()
{
    HamiltonianTerms<Symmetry, OtherScalar> other(N_sites, OPEN_BC);
    other.set_name(label);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        for(std::size_t i=0; i<info[loc].size(); ++i)
        {
            other.save_label(loc, info[loc][i]);
        }
        other.push_local(loc, 1., local[loc].template cast<OtherScalar>());
        Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> other_tight_coupl = tight_coupl[loc].template cast<OtherScalar>();
        for(std::size_t i=0; i<tight_out[loc].size(); ++i)
        {
            for(std::size_t j=0; j<tight_in[(loc+1)%N_sites].size(); ++j)
            {
                other.push_tight(loc, other_tight_coupl(i,j), tight_out[loc][i].template cast<OtherScalar>(), tight_in[(loc+1)%N_sites][j].template cast<OtherScalar>());
            }
        }
        for(std::size_t t=0; t<nextn_out[loc].size(); ++t)
        {
            Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> other_nextn_coupl = nextn_coupl[loc][t].template cast<OtherScalar>();
            for(std::size_t i=0; i<nextn_out[loc][t].size(); ++i)
            {
                for(std::size_t j=0; j<nextn_in[(loc+2)%N_sites][t].size(); ++j)
                {
                    other.push_nextn(loc, other_nextn_coupl(i,j), nextn_out[loc][t][i].template cast<OtherScalar>(), nextn_TransOps[(loc+1)%N_sites][t].template cast<OtherScalar>(), nextn_in[(loc+2)%N_sites][t][j].template cast<OtherScalar>());
                }
            }
        }
    }
    return other;
}

template<typename Symmetry> using HamiltonianTermsXd  = HamiltonianTerms<Symmetry,double>;
template<typename Symmetry> using HamiltonianTermsXcd = HamiltonianTerms<Symmetry,std::complex<double> >;

#endif
