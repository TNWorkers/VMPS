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
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    
    /**
     *  Local terms of Hamiltonian
     *  Index structure: [Lattice Site]
     */
    std::vector<OperatorType> local;
    
    /**
     *    Stores whether the local operator has been set yet
     *    Index structure: [Lattice Site]
     */
    std::vector<bool> localSet;
    
    /**
     *    Local hilbert space dimensions. Initialized with 0 and set whenever a operator is added at a given site
     *    Index structure: [Lattice Site]
     */
    std::vector<int> hilbert_dimension;
    
    /**
     *    Number of lattice sites
     */
    std::size_t N_sites;
    
    /**
     *    All informations stored about the HamiltonianTerms
     *    Index structure [Lattice Site i][Index of information string]
     */
    std::vector<std::vector<std::string>> info;
    
    /**
     *  A given name for the HamiltonianTerms, such as Heisenberg
     */
    std::string label="";
    
    /**
     *  Stores whether to use periodic boundary conditions or not
     */
    bool OPEN_BC;
    
    /**
     *  Collection of all outgoing operators, needed for n site wide interactions starting between lattice site loc and loc+n
     *  Index structure: [Distance n-1][Lattice site loc][Number of transfer operator set t][Number of outgoing local operator i]
     **/
    std::vector<std::vector<std::vector<std::vector<OperatorType>>>> outgoing;
    
    /**
     *  Collection of all incoming operators, needed for n site wide interactions starting between lattice site loc-n and loc
     *  Index structure: [Distance n-1][Lattice site loc][Number of transfer operator set t][Number of incoming local operator j]
     **/
    std::vector<std::vector<std::vector<std::vector<OperatorType>>>> incoming;
    
    /**
     *  Collection of all transfer operators, needed for n site wide interactions starting between lattice site loc and loc+n
     *  Index structure: [Distance n-1][Lattice site loc][Number of transfer operator set t][Number k of transfer operator acting on site loc+k+1]
     **/
    std::vector<std::vector<std::vector<std::vector<OperatorType>>>> transfer;
    
    /**
     *  Collection of all interaction strengths, needed for n site wide interactions interactions starting between lattice site loc and loc+n
     *  Index structure: [Distance n-1][Lattice site loc][Number of transfer operator set t](Number of outgoing operators at site loc, Number of incoming operators at site loc+n)
     **/
    std::vector<std::vector<std::vector<MatrixType>>> coupling;
    
    /**
     *  Stores the currently widest interaction within the system. Is increased, when a wider interaction is added. (E.g. n_max = 4 refers to a 4 sites wide interaction between lattice sites 0 and 4)
     **/
    std::size_t n_max = 0;
    
    /**
     *  Checks whether the dimension of an operator matches the local Hilbert space dimension. Sets the latter if it has not been set yet.
     *  @param loc  Lattice site
     *  @param dim  Assumed dimension of local Hilbert space
     **/
    void assert_hilbert(std::size_t loc, int dim);
    
    /**
     *  Takes the plain interaction operator vectors and matrices and compresses them (Todo: by singular value decomposition)
     */
    void compress(std::vector<std::vector<std::vector<std::vector<OperatorType>>>> &incoming_compressed, std::vector<std::vector<std::vector<std::vector<OperatorType>>>> &outgoing_compressed);
    
public:
    
    /**
     *  Default constructor, does nothing
     */
    HamiltonianTerms() {HamiltonianTerms(0);}
    
    /**
     *  Constructor
     *  @param L    Lattice size
     *  @param bc   Open boundary conditions
     */
    HamiltonianTerms(std::size_t L, bool bc = true);
    
    /**
     *  Adds an interaction between lattice sites loc and loc+n to the HamiltonianTerms.
     *  @param n        Distance (n=1 means next-neighbour)
     *  @param loc      Lattice site where the interaction starts
     *  @param lambda   Interaction strength
     *  @param outOp    Outgoing operator at site loc
     *  @param trans    Vector of transfer operators at sites loc+1, ..., loc+n-1
     *  @param inOp     Incoming operator at site loc+m
     */
    void push(std::size_t n, std::size_t loc, Scalar lambda, OperatorType outOp, std::vector<OperatorType> trans, OperatorType inOp);
    
    /**
     *  Adds a new local interaction to the HamiltonianTerms
     *  @param loc      Lattice site
     *  @param Op       SiteOperator acting on the local Hilbert space of site \p loc
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(0, loc, lambda, Op, ...)
     */
    void push_local(std::size_t loc, Scalar lambda, OperatorType Op);
    
    /**
     *  Adds a new nearest-neighbour interaction to the HamiltonianTerms
     *  @param loc      Lattice site of first site
     *  @param Op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param Op2      SiteOperator acting on the local Hilbert space of site \p loc+1
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(1, loc, lambda, Op1, ..., Op2)
     */
    void push_tight(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Op2);
    
    /**
     *  Adds a new next-nearest-neighbour interaction to the HamiltonianTerms
     *  @param loc      Lattice site of first site
     *  @param Op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param Trans    SiteOperator acting as transfer operator on the local Hilbert space of site \p loc+1
     *  @param Op2      SiteOperator acting on the local Hilbert space of site \p loc+2
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(2, loc, lambda, Op1, {Trans}, Op2)
     */
    void push_nextn(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Trans, OperatorType Op2);

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
     *  @param  loc  Lattice site
     *  @return The dimension of the local Hilbert space. If not set yet, this will return 0.
     */
    std::size_t Hilbert_dimension(std::size_t loc) const;
    
    /**
     *  Constructs a vector of SuperMatrix from interactions. Compresses the interaction before.
     *  Resizes the first and last SuperMatrix to row/column for open boundary conditions.
     *  @return Vector of constructed SuperMatrix
     */
    std::vector<SuperMatrix<Symmetry,Scalar>> construct_Matrix();
    
    /**
     * @return  Given name of this instance of HamiltonianTerms
     */
    std::string name() const {return label;}
    
    
    /**
     *  Scales all interactions by a given factor.
     *  @param factor   The factor to scale the interactions with
     *  @param offset
     */
    void scale(double factor, double offset=0.);
    
    /**
     *  @return Number of lattice sites \p N_sites
     */
    std::size_t size() const {return N_sites;}
    
    /**
     *  @return Cast instance of HamiltonianTerms with another scalar type
     */
    template<typename OtherScalar> HamiltonianTerms<Symmetry, OtherScalar> cast();
    
    /**
     *  @return Const reference to the local terms at lattice site \p loc
     */
    OperatorType const& localOps(std::size_t loc) const {return local[loc];}
    
    /**
     *  @return Const reference to the incoming \p n.-neighbour interaction terms at lattice site \p loc
     */
    std::vector<std::vector<OperatorType>> const& inOps(std::size_t n, std::size_t loc) const {assert(n > 1 and "Only possible for interactions with ranges > 1"); return incoming[n-1][loc];}
    
    /**
     *  @return Const reference to the outgoing \p n.-neighbour interaction terms at lattice site \p loc
     */
    std::vector<std::vector<OperatorType>> const& outOps(std::size_t n, std::size_t loc) const {assert(n > 1 and "Only possible for interactions with ranges > 1"); return outgoing[n-1][loc];}
    
    /**
     * Const reference to the transfer operator lists of \p n.-neighbour interactions starting at lattice site \p loc
     */
    std::vector<std::vector<OperatorType>> const& transferOps(std::size_t n, std::size_t loc) const {assert(n > 1 and "Only possible for interactions with ranges > 1"); return transfer[n-1][loc];}
    
    /**
     *  @return Const reference to the incoming nearest-neighbour terms at lattice site \p loc
     */
    std::vector<OperatorType> const& tight_inOps(std::size_t loc) const {return incoming[0][loc][0];}
    
    /**
     *  @return Const reference to the outgoing nearest-neighbour terms at lattice site \p loc
     */
    std::vector<OperatorType> const& tight_outOps(std::size_t loc) const {return outgoing[0][loc][0];}
    
    /**
     * Const reference to the incoming next-nearest-neighbour terms at lattice site \p loc
     */
    std::vector<std::vector<OperatorType>> const& nextn_inOps(std::size_t loc) const {return incoming[1][loc];}
    
    /**
     * Const reference to the outgoing next-nearest-neighbour terms at lattice site \p loc
     */
    std::vector<std::vector<OperatorType>> const& nextn_outOps(std::size_t loc) const {return outgoing[1][loc];}
};

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry, Scalar>::assert_hilbert(std::size_t loc, int dim)
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

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
push(std::size_t n, std::size_t loc, Scalar lambda, OperatorType outOp, std::vector<OperatorType> transOps, OperatorType inOp)
{
    assert(loc < N_sites and "Chosen lattice site out of bounds");
    if(lambda != 0.)
    {
        if(n == 0)
        {
            //std::cout << "Local interaction at site " << loc << std::endl;
            assert(outOp.Q == qvac and "Local operator is not a singlet");
            assert_hilbert(loc, outOp.data.rows());
            if(localSet[loc])
            {
                local[loc] += lambda * outOp;
            }
            else
            {
                local[loc] = lambda * outOp;
                localSet[loc] = true;
            }
        }
        else
        {
            //std::cout << n << ".-neighbour interaction between the sites " << loc << " and " << (loc+n)%N_sites << std::endl;
            assert(transOps.size() == n-1 and "Distance does not match to number of transfer operators!");
            if(n > n_max)
            {
                for(int i=n_max; i<n; ++i)
                {
                    std::vector<std::vector<MatrixType>> temp_coup(N_sites);
                    coupling.push_back(temp_coup);
                    std::vector<std::vector<std::vector<OperatorType>>> temp_ops(N_sites);
                    outgoing.push_back(temp_ops);
                    incoming.push_back(temp_ops);
                    transfer.push_back(temp_ops);
                }
                n_max = n;
            }
            
            std::ptrdiff_t transptr;
            if(n == 1)
            {
                transptr = 0;
                if(coupling[0][loc].size() == 0)
                {
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(1,1);
                    std::vector<OperatorType> temp;
                    
                    coupling[0][loc].push_back(matrix);
                    outgoing[0][loc].push_back(temp);
                    incoming[0][(loc+1)%N_sites].push_back(temp);
                }
            }
            else
            {
                transptr = std::distance(transfer[n-1][loc].begin(), find(transfer[n-1][loc].begin(), transfer[n-1][loc].end(), transOps));
                if(transptr >= transfer[n-1][loc].size())    // If the operator cannot be found, push it to the corresponding terms and resize the interaction matrix
                {
                    transfer[n-1][loc].push_back(transOps);
                    for(std::size_t t=0; t<n-1; ++t)
                    {
                        assert_hilbert((loc+t+1)%N_sites, transOps[t].data.rows());
                    }
                    
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(1,1);
                    std::vector<OperatorType> temp;
                    
                    coupling[n-1][loc].push_back(matrix);
                    outgoing[n-1][loc].push_back(temp);
                    incoming[n-1][(loc+n)%N_sites].push_back(temp);
                }
            }
            
            std::ptrdiff_t outptr = std::distance(outgoing[n-1][loc][transptr].begin(), find(outgoing[n-1][loc][transptr].begin(), outgoing[n-1][loc][transptr].end(), outOp));
            if(outptr >= outgoing[n-1][loc][transptr].size())
            {
                assert_hilbert(loc, outOp.data.rows());
                outgoing[n-1][loc][transptr].push_back(outOp);
                coupling[n-1][loc][transptr].conservativeResize(outgoing[n-1][loc][transptr].size(), incoming[n-1][(loc+n)%N_sites][transptr].size());
                coupling[n-1][loc][transptr].bottomRows(1).setZero();
                hilbert_dimension[loc] = outOp.data.rows();
                //std::cout << "Outgoing operator for " << n << ".-neighbour interaction between the sites " << loc << " and " << (loc+n)%N_sites << " was not found, so a new row is added to the coupling matrix and a " << outOp.Q << "-operator is pushed" << std::endl;
            }
            std::ptrdiff_t inptr = std::distance(incoming[n-1][(loc+n)%N_sites][transptr].begin(), find(incoming[n-1][(loc+n)%N_sites][transptr].begin(), incoming[n-1][(loc+n)%N_sites][transptr].end(), inOp));
            if(inptr >= incoming[n-1][(loc+n)%N_sites][transptr].size())
            {
                assert_hilbert((loc+n)%N_sites, inOp.data.rows());
                incoming[n-1][(loc+n)%N_sites][transptr].push_back(inOp);
                coupling[n-1][loc][transptr].conservativeResize(outgoing[n-1][loc][transptr].size(), incoming[n-1][(loc+n)%N_sites][transptr].size());
                coupling[n-1][loc][transptr].rightCols(1).setZero();
                hilbert_dimension[(loc+n)%N_sites] = inOp.data.rows();
                //std::cout << "Incoming operator for " << n << ".-neighbour interaction between the sites " << loc << " and " << (loc+n)%N_sites << " was not found, so no new column is added to the coupling matrix and a " << inOp.Q << "-operator is pushed" << std::endl;

            }
            coupling[n-1][loc][transptr](outptr, inptr) += lambda;
        }
    }
}

template<typename Symmetry, typename Scalar> HamiltonianTerms<Symmetry,Scalar>::
HamiltonianTerms(std::size_t L, bool BC) : N_sites(L)
{
    info.resize(N_sites);
    hilbert_dimension.resize(N_sites, 0);
    local.resize(N_sites);
    localSet.resize(N_sites, false);
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
    push(0, loc, lambda, Op, {}, Op);
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
push_tight(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Op2)
{
    push(1, loc, lambda, Op1, {}, Op2);
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry,Scalar>::
push_nextn(std::size_t loc, Scalar lambda, OperatorType Op1, OperatorType Trans, OperatorType Op2)
{
    push(2, loc, lambda, Op1, {Trans}, Op2);
}

template<typename Symmetry, typename Scalar> void HamiltonianTerms<Symmetry, Scalar>::
compress(std::vector<std::vector<std::vector<std::vector<OperatorType>>>> &outgoing_compressed, std::vector<std::vector<std::vector<std::vector<OperatorType>>>> &incoming_compressed)
{
    outgoing_compressed.resize(n_max);
    incoming_compressed.resize(n_max);
    outgoing_compressed[0].resize(N_sites);
    incoming_compressed[0].resize(N_sites);
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        outgoing_compressed[0][loc].resize(outgoing[0][loc].size());
        incoming_compressed[0][(loc+1)%N_sites].resize(incoming[0][(loc+1)%N_sites].size());
        if(outgoing[0][loc].size() > 0)
        {
            if(outgoing[0][loc][0].size() < incoming[0][(loc+1)%N_sites][0].size())
            {
                //outgoing_compressed[0][loc].push_back(outgoing[0][loc][0]);
                //incoming_compressed[0][(loc+1)%N_sites].push_back(coupling[0][loc][0] * incoming[0][(loc+1)%N_sites][0]);
                outgoing_compressed[0][loc][0] = outgoing[0][loc][0];
                incoming_compressed[0][(loc+1)%N_sites][0] = coupling[0][loc][0] * incoming[0][(loc+1)%N_sites][0];
            }
            else
            {
                //outgoing_compressed[0][loc].push_back(outgoing[0][loc][0] * coupling[0][loc][0]);
                //incoming_compressed[0][(loc+1)%N_sites].push_back(incoming[0][(loc+1)%N_sites][0]);
                outgoing_compressed[0][loc][0] = outgoing[0][loc][0] * coupling[0][loc][0];
                incoming_compressed[0][(loc+1)%N_sites][0] = incoming[0][(loc+1)%N_sites][0];
            }
        }
    }
    for(std::size_t n=1; n<n_max; ++n)
    {
        outgoing_compressed[n].resize(N_sites);
        incoming_compressed[n].resize(N_sites);
        for(std::size_t loc=0; loc<N_sites; ++loc)
        {
            outgoing_compressed[n][loc].resize(transfer[n][loc].size());
            incoming_compressed[n][(loc+n+1)%N_sites].resize(transfer[n][loc].size());
            for(std::size_t t=0; t<transfer[n][loc].size(); ++t)
            {
                if(outgoing[n][loc][t].size() < incoming[n][(loc+n+1)%N_sites][t].size())
                {
                    outgoing_compressed[n][loc][t] = outgoing[n][loc][t];
                    incoming_compressed[n][(loc+n+1)%N_sites][t] = coupling[n][loc][t] * incoming[n][(loc+n+1)%N_sites][t];
                }
                else
                {
                    outgoing_compressed[n][loc][t] = outgoing[n][loc][t] * coupling[n][loc][t];
                    incoming_compressed[n][(loc+n+1)%N_sites][t] = incoming[n][(loc+n+1)%N_sites][t];
                }
            }
        }
    }
}

template<typename Symmetry, typename Scalar> std::vector<SuperMatrix<Symmetry, Scalar>> HamiltonianTerms<Symmetry, Scalar>::
construct_Matrix()
{
    std::vector<std::vector<std::vector<std::vector<OperatorType>>>> outgoing_compressed;
    std::vector<std::vector<std::vector<std::vector<OperatorType>>>> incoming_compressed;
    compress(outgoing_compressed, incoming_compressed);

    std::vector<SuperMatrix<Symmetry, Scalar>> G;
    for (std::size_t loc=0; loc<N_sites; ++loc)
    {
        SuperMatrix<Symmetry,Scalar> S;
        if(hilbert_dimension[loc] == 0) // Create a trivial SuperMatrix if no operator has been set.
        {
            hilbert_dimension[loc] = 1;
            OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
            S.set(2,2,hilbert_dimension[loc]);
            S(0,0) = Id;
            S(1,1) = Id;
            //std::cout << "SuperMatrix at site " << loc << " is trivial (no operators have been added there)\n" << std::endl;
        }
        else if(n_max == 0)
        {
            //std::cout << "Filling the SuperMatrix at site " << loc << ":" << std::endl;
            OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
            S.set(2,2,hilbert_dimension[loc]);
            S(0,0) = Id;
            //std::cout << "\tEntry (0,0): Identity" << std::endl;
            S(1,0) = local[loc];
            //std::cout << "\tEntry (1,0): Local term" << std::endl;
            S(1,1) = Id;
            //std::cout << "\tEntry (1,1): Identity" << std::endl;

            //std::cout << "SuperMatrix at site " << loc << " has dimension 2x2, dimension of local Hilbert space: " << hilbert_dimension[loc] << "\n" << std::endl;
        }
        else
        {
            //std::cout << "Filling the SuperMatrix at site " << loc << ":" << std::endl;
            
            std::size_t rows = 2;
            std::size_t cols = 2;
            if(incoming_compressed[0][loc].size() > 0)
            {
                rows += incoming_compressed[0][loc][0].size();
            }
            if(outgoing_compressed[0][loc].size() > 0)
            {
                cols += outgoing_compressed[0][loc][0].size();
            }
            for(std::size_t n=1; n<n_max; ++n)
            {
                for(std::size_t t=0; t<incoming_compressed[n][loc].size(); ++t)
                {
                    rows += incoming_compressed[n][loc][t].size();
                }
                for(std::size_t t=0; t<outgoing_compressed[n][loc].size(); ++t)
                {
                    cols += outgoing_compressed[n][loc][t].size();
                }

                for(std::size_t m=0; m<n; ++m)
                {
                    for(std::size_t t=0; t<transfer[n][(N_sites+loc-m-1)%N_sites].size(); ++t)
                    {
                        rows += outgoing_compressed[n][(N_sites+loc-m-1)%N_sites][t].size();
                        cols += outgoing_compressed[n][(N_sites+loc-m-1)%N_sites][t].size();
                    }
                }
            }
            
            OperatorType Id(Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(hilbert_dimension[loc],hilbert_dimension[loc]).sparseView(),Symmetry::qvacuum());
            S.set(rows, cols, hilbert_dimension[loc]);
            S(0,0) = Id;
            //std::cout << "\tEntry (0,0): Identity" << std::endl;

            
            std::size_t current = 1;
            if(incoming_compressed[0][loc].size() > 0)
            {
                for(std::size_t i=0; i<incoming_compressed[0][loc][0].size(); ++i)
                {
                    //std::cout << "\tEntry (" << current << ",0): Incoming next-neighbour interaction term" << std::endl;
                    S(current++, 0) = incoming_compressed[0][loc][0][i];
                }
            }
            
            for(std::size_t n=1; n<n_max; ++n)
            {
                for(std::size_t t=0; t<incoming_compressed[n][loc].size(); ++t)
                {
                    for(std::size_t i=0; i<incoming_compressed[n][loc][t].size(); ++i)
                    {
                        //std::cout << "\tEntry (" << current << ",0): Incoming " << n+1 << ".-neighbour interaction term" << std::endl;
                        S(current++, 0) = incoming_compressed[n][loc][t][i];
                    }
                }
                std::size_t temp = current;
                
                for(std::size_t m=0; m<n; ++m)
                {
                    for(std::size_t t=0; t<transfer[n][(N_sites+loc-m-1)%N_sites].size(); ++t)
                    {
                        current += outgoing_compressed[n][(N_sites+loc-m-1)%N_sites][t].size();
                    }
                }
                if(temp < current)
                {
                    //std::cout << "\tEntries (" << temp << ",0)-(" << current-1 << ",0): Skipped due to transfer operators of " << n+1 << ".-neighbour interactions" << std::endl;
                }
            }
            
            if (!localSet[loc]) // If no local interaction has been added, the local interaction becomes a dummy SiteOperator with correct dimension
            {
                local[loc] = 0*Id;
            }
            S(rows-1,0) = local[loc];
            
            //std::cout << "\tEntry (" << rows-1 << ",0): Local term" << std::endl;
            
            current = 1;
            
            if(outgoing_compressed[0][loc].size() > 0)
            {
                for(std::size_t i=0; i<outgoing_compressed[0][loc][0].size(); ++i)
                {
                    //std::cout << "\tEntry (" << rows-1 << "," << current << "): Outgoing next-neighbour interaction term" << std::endl;
                    S(rows-1, current++) = outgoing_compressed[0][loc][0][i];
                }
            }
            
            for(std::size_t n=1; n<n_max; ++n)
            {
                std::size_t temp = current;
                
                for(std::size_t m=0; m<n; ++m)
                {
                    for(std::size_t t=0; t<transfer[n][(N_sites+loc-m-1)%N_sites].size(); ++t)
                    {
                        current += outgoing_compressed[n][(N_sites+loc-m-1)%N_sites][t].size();
                    }
                }
                if(temp < current)
                {
                    //std::cout << "\tEntries (" << rows-1 << "," << temp << ")-(" << rows-1  << "," << current-1 << "): Skipped due to transfer operators of " << n+1 << ".-neighbour interactions" << std::endl;
                }
                for(std::size_t t=0; t<outgoing_compressed[n][loc].size(); ++t)
                {
                    for(std::size_t i=0; i<outgoing_compressed[n][loc][t].size(); ++i)
                    {
                        //std::cout << "\tEntry (" << rows-1 << "," << current << "): Outgoing " << n+1 << ".-neighbour interaction term" << std::endl;
                        S(rows-1, current++) = outgoing_compressed[n][loc][t][i];
                    }
                }
            }
            
            S(rows-1,cols-1) = Id;
            //std::cout << "\tEntry (" << rows-1 << "," << cols-1 << "): Identity" << std::endl;
            
            std::size_t current_row = 1;
            std::size_t current_col = 1;
            if(incoming_compressed[0][loc].size() > 0)
            {
                current_row += incoming_compressed[0][loc][0].size();

            }
            if(outgoing_compressed[0][loc].size() > 0)
            {
                current_col += outgoing_compressed[0][loc][0].size();
            }
            for(std::size_t n=1; n<n_max; ++n)
            {
                for(std::size_t t=0; t<incoming_compressed[n][loc].size(); ++t)
                {
                    current_row += incoming_compressed[n][loc][t].size();
                }
                for(std::size_t m=0; m<n; ++m)
                {
                    for(std::size_t t=0; t<transfer[n][(N_sites + loc - n + m) % N_sites].size(); ++t)
                    {
                        for(std::size_t i=0; i<outgoing_compressed[n][(N_sites + loc - n + m)%N_sites][t].size(); ++i)
                        {
                            //std::cout << "\tEntry (" << current_row << "," << current_col << "): Transfer operator for a " << n+1 << ".-neighbour interaction ranging from site " << (N_sites + loc - n + m)%N_sites << " to site " << (loc + 1 + m)%N_sites << std::endl;
                            S(current_row++, current_col++) = transfer[n][(N_sites + loc - n + m)%N_sites][t][n-1-m];
                        }
                    }
                }
                for(std::size_t t=0; t<outgoing_compressed[n][loc].size(); ++t)
                {
                    current_col += outgoing_compressed[n][loc][t].size();
                }
            }
            
            //std::cout << "SuperMatrix at site " << loc << " has dimension " << rows << "x" << cols << ", dimension of local Hilbert space: " << hilbert_dimension[loc] << "\n" << std::endl;
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
scale (double factor, double offset)
{
    if (std::abs(factor-1.) > ::mynumeric_limits<double>::epsilon())
    {
        for (std::size_t loc=0; loc<N_sites; ++loc)
        {
            local[loc] = factor * local[loc];
            for(std::size_t n=0; n<n_max; ++n)
            {
                for(std::size_t t=0; t<coupling[n][loc].size(); ++t)
                {
                    coupling[n][loc][t] *= factor;
                }
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
                push_local(loc, offset, Id);
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
        
        if(outgoing[0][loc].size() > 0)
        {
            Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> other_coupling = coupling[0][loc][0].template cast<OtherScalar>();
            for(std::size_t i=0; i<other_coupling.rows(); ++i)
            {
                for(std::size_t j=0; j<other_coupling.cols(); ++j)
                {
                    SiteOperator<Symmetry, OtherScalar> other_out = outgoing[0][loc][0][i].template cast<OtherScalar>();
                    SiteOperator<Symmetry, OtherScalar> other_in = incoming[0][(loc+1)%N_sites][0][j].template cast<OtherScalar>();
                    other.push_tight(loc, other_coupling(i,j), other_out, other_in);
                }
            }
        }
        for(std::size_t n=1; n<n_max; ++n)
        {
            for(std::size_t t=0; t<transfer[n][loc].size(); ++t)
            {
                Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> other_coupling = coupling[n][loc][t].template cast<OtherScalar>();
                for(std::size_t i=0; i<other_coupling.rows(); ++i)
                {
                    for(std::size_t j=0; j<other_coupling.cols(); ++j)
                    {
                        SiteOperator<Symmetry, OtherScalar> other_out = outgoing[n][loc][t][i].template cast<OtherScalar>();
                        SiteOperator<Symmetry, OtherScalar> other_in = incoming[n][(loc+1+n)%N_sites][t][j].template cast<OtherScalar>();
                        std::vector<SiteOperator<Symmetry, OtherScalar>> other_transfer;
                        for(std::size_t k=0; k<transfer[n][loc][t].size(); ++k)
                        {
                            other_transfer.push_back(transfer[n][loc][t][k].template cast<OtherScalar>());
                        }
                        other.push(n+1, loc, other_coupling(i,j), other_out, other_transfer, other_in);
                    }
                }
            }
        }
    }
    return other;
}

template<typename Symmetry> using HamiltonianTermsXd  = HamiltonianTerms<Symmetry,double>;
template<typename Symmetry> using HamiltonianTermsXcd = HamiltonianTerms<Symmetry,std::complex<double> >;

#endif
