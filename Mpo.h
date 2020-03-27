#ifndef STRAWBERRY_Mpo_WITH_Q
#define STRAWBERRY_Mpo_WITH_Q

/// \cond
#include "boost/multi_array.hpp"
/// \endcond

#include "termcolor.hpp" //from https://github.com/ikalnytskyi/termcolor

/// \cond
#include <Eigen/SparseCore>
/// \endcond

#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
typedef Eigen::SparseMatrix<double,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
using namespace Eigen;

/// \cond
#include <unsupported/Eigen/KroneckerProduct>
/// \endcond

#include "util/macros.h"
#include "MpoTerms.h"
#include "tensors/Qbasis.h"
#include "DmrgTypedefs.h"

namespace VMPS{};

template<typename Symmetry, typename Scalar> class Mps;
template<typename Symmetry, typename Scalar> class Umps;
template<typename Symmetry, typename MpHamiltonian, typename Scalar> class DmrgSolver;
template<typename Symmetry, typename Scalar, typename MpoScalar> class MpsCompressor;
template<typename Symmetry, typename MpHamiltonian, typename Scalar> class VumpsSolver;


template<typename Symmetry, typename Scalar=double>
class Mpo : public MpoTerms<Symmetry,Scalar>
{
	typedef SparseMatrixXd SparseMatrixType;
	typedef SiteOperator<Symmetry,Scalar> OperatorType;
	static constexpr size_t Nq = Symmetry::Nq;
	typedef typename Symmetry::qType qType;
    typename Symmetry::qType qVac = Symmetry::qvacuum();
	
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class DmrgSolver;
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class VumpsSolver;
	template<typename Symmetry_, typename S1, typename S2> friend class MpsCompressor;
	template<typename H, typename Symmetry_, typename S1, typename S2, typename V> friend class TDVPPropagator;
	template<typename Symmetry_, typename S_> friend class Mpo;
	
public:
	
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	typedef Scalar Scalar_;
    
	Mpo() : MpoTerms<Symmetry, Scalar>(){};
	
	Mpo(size_t L_input);
	
	Mpo(std::size_t L_input, qType Qtot_input, std::string label_input="Mpo", bool HERMITIAN_input=false, bool UNITARY_input=false, bool HAMILTONIAN_input=false, BC BC_input=BC::OPEN);

	template<typename CouplScalar>
    void construct_from_pushlist(const PushType<OperatorType,CouplScalar>& pushlist, const std::vector<std::vector<std::string>>& labellist, size_t Lcell);
	
	void setLocal(std::size_t loc, const OperatorType& op);
	
	void setLocal(std::size_t loc, const OperatorType& op, const OperatorType& signOp);
	
	void setLocal(std::size_t loc, const OperatorType& op, const std::vector<OperatorType>& signOp);
	
    void setLocal(const std::vector<std::size_t>& locs, const std::vector<OperatorType>& ops);
	
	void setLocal(const std::vector<std::size_t>& locs, const std::vector<OperatorType>& ops, const OperatorType& signOp);
	
	void setLocal(const std::vector<std::size_t>& locs, const std::vector<OperatorType>& ops, const std::vector<OperatorType>& signOps);
	
	void setLocalStag(std::size_t loc, const OperatorType& op, const std::vector<OperatorType>& stagSignOps);
	
	void setLocalSum(const OperatorType& op, Scalar (*f)(int)=localSumTrivial);
	
	void setLocalSum(const std::vector<OperatorType>& op, std::vector<Scalar> coeffs);
	
	void setProductSum(const OperatorType& op1, const OperatorType& op2);
	
	void scale(double factor=1., double offset=0.);
		
	void precalc_TwoSiteData(bool FORCE=false);

	std::string info() const;
	
	//double memory(MEMUNIT memunit=GB) const;
	
	//double sparsity(bool USE_SQUARE=false, bool PER_MATRIX=true) const;

	inline std::size_t length() const {return this->size();}
	
	inline std::size_t volume() const {return N_phys;}

	template<typename T, typename ... Operator>
	static std::vector<T> get_N_site_interaction(T const & Op0, Operator const & ... Ops) {std::vector<T> out { {Op0, Ops ...} }; return out;};
	//inline std::size_t auxrows(std::size_t loc) const {return this->get_qAux()[loc].fullM();}
	//inline std::size_t auxcols(std::size_t loc) const {return this->get_qAux()[loc+1].fullM();}
    //inline void setOpBasis   (const vector<vector<qType> > &q) {qOp=q;}
    //inline void setOpBasisSq (const vector<vector<qType> > &qOpSq_in) {qOpSq=qOpSq_in;}
    //inline const unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<Scalar> > &Vsq_at (size_t loc) const {return Vsq[loc];};

	
	inline int locality() const {return LocalSite;}
	inline void set_locality(std::size_t LocalSite_input) {LocalSite = LocalSite_input;}
	inline OperatorType localOperator() const {return LocalOp;}
	inline void set_localOperator (OperatorType LocalOp_input) {LocalOp = LocalOp_input;}
	
    static Mpo<Symmetry,Scalar> Identity(const std::vector<std::vector<qType>>& qPhys);
    static Mpo<Symmetry,Scalar> Zero(const std::vector<std::vector<qType>>& qPhys);
	
	inline bool IS_UNITARY() const {return UNITARY;};
	
	inline bool IS_HERMITIAN() const {return HERMITIAN;};
	
	inline bool IS_HAMILTONIAN() const {return HAMILTONIAN;};
	
	inline bool HAS_TWO_SITE_DATA() const {return GOT_TWO_SITE_DATA;};
	
	
    boost::multi_array<Scalar,4> H2site(std::size_t loc, bool HALF_THE_LOCAL_TERM=false) const {assert(false and "Method H2site is deprecated.");}

	typedef Mps<Symmetry,double> StateXd;
	typedef Umps<Symmetry,double> StateUd;
	typedef Mps<Symmetry,std::complex<double>> StateXcd;
	typedef Umps<Symmetry,std::complex<double>> StateUcd;
	typedef MpsCompressor<Symmetry,double,double> CompressorXd;
	typedef MpsCompressor<Symmetry,std::complex<double>,double>	CompressorXcd;
	typedef Mpo<Symmetry> Operator;
	
	bool GOT_TWO_SITE_DATA = false;
	std::vector<std::vector<TwoSiteData<Symmetry,Scalar>>> TSD;
	
	bool UNITARY = false;
	bool HERMITIAN  = false;
	bool HAMILTONIAN = false;
	bool GOT_SEMIOPEN_LEFT = false;
	bool GOT_SEMIOPEN_RIGHT = false;
	
	OperatorType LocalOp;
	int LocalSite = -1;
		
	std::size_t N_phys = 0;
	
	//void initialize();
	
	void generate_label(std::size_t Lcell);
        
 
    /**
     *  Adds an interaction between lattice sites loc and loc+n to the Hamiltonianthis->
     *  @param n        Distance (n=1 means next-neighbour)
     *  @param loc      Lattice site where the interaction starts
     *  @param lambda   Interaction strength
     *  @param outOp    Outgoing operator at site loc
     *  @param trans    Vector of transfer operators at sites loc+1, ..., loc+n-1
     *  @param inOp     Incoming operator at site loc+m
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    void push_width(const std::size_t n, const std::size_t loc, const Scalar lambda, const OperatorType& outOp, const std::vector<OperatorType>& trans, const OperatorType& inOp);
    
    /**
     *  Adds a new local interaction to the MpoTerms
     *  @param loc      Lattice site
     *  @param op       SiteOperator acting on the local Hilbert space of site \p loc
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    void push_local(const std::size_t loc, const Scalar lambda, const OperatorType& op);
    
    /**
     *  Adds a new nearest-neighbour interaction to the MpoTerms
     *  @param loc      Lattice site of first site
     *  @param op1      SiteOperator acting on the local Hilbert space of site \p loc
     *  @param op2      SiteOperator acting on the local Hilbert space of site \p loc+1
     *  @param lambda   Scalar of interaction strength that is multiplied to the operator
     *
     *  For convenience, redirects to push(std::size_t loc, std::vector<OperatorType> opList, Scalar lambda)
     */
    void push_tight(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& op2);

    
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
    void push_nextn(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& trans, const OperatorType& op2);
    
    static Mpo<Symmetry,Scalar> cast_Terms_to_Mpo(const MpoTerms<Symmetry,Scalar>& input);
};

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar>::
Mpo(std::size_t L_input)
: MpoTerms<Symmetry, Scalar>(L_input){}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar>::
Mpo (std::size_t L_input, qType Qtot_input, string label_input, bool HERMITIAN_input, bool UNITARY_input, bool HAMILTONIAN_input, BC BC_input)
: MpoTerms<Symmetry, Scalar>(L_input, BC_input, Qtot_input), HERMITIAN(HERMITIAN_input), UNITARY(UNITARY_input), HAMILTONIAN(HAMILTONIAN_input) {this->set_name(label_input);}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
push_width(const std::size_t width, const std::size_t loc, const Scalar lambda, const OperatorType& outOp, const std::vector<OperatorType>& trans, const OperatorType& inOp)
{
    std::vector<OperatorType> oplist(0);
    oplist.push_back(outOp);
    for(std::size_t m=0; m<trans.size(); ++m)
    {
        oplist.push_back(trans[m]);
    }
    oplist.push_back(inOp);
    this->push(loc, oplist, lambda);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
push_local(const std::size_t loc, const Scalar lambda, const OperatorType& op)
{
    this->push(loc, {op}, lambda);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
push_tight(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& op2)
{
    this->push(loc, {op1, op2}, lambda);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
push_nextn(const std::size_t loc, const Scalar lambda, const OperatorType& op1, const OperatorType& trans, const OperatorType& op2)
{
    this->push(loc, {op1, trans, op2}, lambda);
}


template<typename Symmetry, typename Scalar>
string Mpo<Symmetry,Scalar>::
info() const
{
	std::stringstream ss;
	ss << termcolor::colorize << termcolor::bold << this->get_name() << termcolor::reset << "→ L=" << this->size();
	if(N_phys > this->size()){
        ss << ",V=" << N_phys;
    }
	ss << ", " << Symmetry::name() << ", ";
	
	ss << "UNITARY=" << boolalpha << UNITARY << ", ";
	ss << "HERMITIAN=" << boolalpha << HERMITIAN << ", ";
	ss << "maxPower=" << this->maxPower() << ", ";
	ss << "BC=" << this->get_boundary_condition() << ", ";
	ss << "2SITE_DATA=" << boolalpha << GOT_TWO_SITE_DATA << ", ";
	if(LocalSite != -1)
	{
		ss << "locality=" << LocalSite << ", ";
	}
	
    auto qAux = this->get_qAux();
    std::vector<int> dAux(this->size()+1);
    dAux[0] = this->auxBasis(0).fullM();
	std::set<std::pair<int,int> > dAux_set;
	for (std::size_t loc=0; loc<this->size(); ++loc)
	{
        dAux[loc+1] = this->auxBasis(loc+1).fullM();
		dAux_set.insert(std::make_pair(dAux[loc],dAux[loc+1]));
	}
	ss << "dAux=";
	for (const auto& dAux_pair : dAux_set)
	{
		ss << dAux_pair.first << "x" << dAux_pair.second;
		ss << ",";
	}
	ss << " ";
	
		ss << "mem=" << round(this->memory(GB),3) << "GB";
	ss << ", sparsity=" << this->sparsity();
	// if(this->check_SQUARE())
    // {
    //     ss << ", sparsity(sq)=" << sparsity(true);
    // }
	return ss.str();
}

/*template<typename Symmetry, typename Scalar>
double Mpo<Symmetry,Scalar>::
memory(MEMUNIT memunit) const
{
	double res = 0.;
    
    for (std::size_t loc=0; loc<this->W.size(); ++loc)
    {
        std::size_t hd = this->get_hilbert_dimension(loc);
        for (std::size_t n1=0; n1<this->W[loc].size(); ++n1)
        {
            for (std::size_t n2=0; n2<this->W[loc][n1].size(); ++n2)
            {
                for (std::size_t t=0; t<this->W[loc][n1][n2].size(); ++t)
                {
                    res += this->W[loc][n1][n2][t].memory(memunit);
                }
            }
        }
    }
	return res;
}*/

/*template<typename Symmetry, typename Scalar>
double Mpo<Symmetry,Scalar>::
sparsity (bool USE_SQUARE, bool PER_MATRIX) const
{
	if (USE_SQUARE) {assert(this->check_power(2ul));}
	double N_nonZeros = 0.;
	double N_elements = 0.;
	double N_matrices = 0.;
	
	for (std::size_t loc=0; loc<this->size(); ++loc)
	{
        std::size_t hd = this->get_hilbert_dimension(loc);
		N_matrices += hd * hd * this->opBasis(loc).size();
                
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		for (size_t k=0; k<this->qOp[l].size(); ++k)
		{
			// if constexpr (Symmetry::NON_ABELIAN) {N_nonZeros += this->W[l][s1][s2][k].nonZeros();}
			// if constexpr (Symmetry::NON_ABELIAN) {N_elements += this->W[l][s1][s2][k].rows()   * this->W[l][s1][s2][k].cols();}
			// else
			// {
			// 	N_nonZeros += (USE_SQUARE)? Wsq[l][s1][s2][k].nonZeros() : this->W[l][s1][s2][k].nonZeros();
			// 	N_elements += (USE_SQUARE)? Wsq[l][s1][s2][k].rows() * Wsq[l][s1][s2][k].cols():
			// 		this->W[l][s1][s2][k].rows()   * this->W[l][s1][s2][k].cols();
			// }
		}
	}
	
	//return (PER_MATRIX)? N_nonZeros/N_matrices : N_nonZeros/N_elements;
    return 0.;
}*/

template<typename Symmetry, typename Scalar> Mpo<Symmetry,Scalar> Mpo<Symmetry,Scalar>::
Identity(const std::vector<std::vector<qType>>& qPhys)
{
    Mpo<Symmetry,Scalar> out(qPhys.size(), Symmetry::qvacuum(), "Id", true, true, false, BC::OPEN);
    for(std::size_t loc=0; loc<qPhys.size(); ++loc)
    {
        out.set_qPhys(loc, qPhys[loc]);
    }
    out.set_Identity();
	return out;
}

template<typename Symmetry, typename Scalar> Mpo<Symmetry,Scalar> Mpo<Symmetry,Scalar>::
Zero(const std::vector<std::vector<qType>>& qPhys)
{
    Mpo<Symmetry,Scalar> out(qPhys.size(), Symmetry::qvacuum(), "Zero", true, true, false, BC::OPEN);
    for(std::size_t loc=0; loc<qPhys.size(); ++loc)
    {
        out.set_qPhys(loc, qPhys[loc]);
    }
    out.set_Zero();
	return out;
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
generate_label(std::size_t Lcell)
{
	std::stringstream ss;
	ss << this->get_name();
	std::vector<std::string> info = this->get_info();
	
	std::map<std::string,std::set<std::size_t> > cells;
	
	for (std::size_t loc=0; loc<info.size(); ++loc)
	{
		cells[info[loc]].insert(loc%Lcell);
	}
	
	if (cells.size() == 1)
	{
		ss << "(" << info[0] << ")";
	}
	else
	{
		std::vector<std::pair<std::string,std::set<std::size_t> > > cells_resort(cells.begin(), cells.end());
		
		// sort according to smallest l, not according to label
		sort(cells_resort.begin(), cells_resort.end(),
			 [](const std::pair<std::string,std::set<std::size_t> > &a, const std::pair<std::string,std::set<std::size_t> > &b) -> bool
			 {
				 return *min_element(a.second.begin(),a.second.end()) < *min_element(b.second.begin(),b.second.end());
			 });
		
		ss << ":" << std::endl;
		for (auto c:cells_resort)
		{
			ss << " •l=";
			//			for (auto s:c.second)
			//			{
			//				cout << s << ",";
			//			}
			//			cout << endl;
			if (c.second.size() == 1)
			{
				ss << *c.second.begin(); // one site
			}
			else
			{
				// check mod 2
				if (std::all_of(c.second.cbegin(), c.second.cend(), [](int i){ return i%2==0;}) and c.second.size() == this->size()/2)
				{
					ss << "even";
				}
				else if (std::all_of(c.second.cbegin(), c.second.cend(), [](int i){ return i%2==1;}) and c.second.size() == this->size()/2)
				{
					ss << "odd";
				}
				// check mod 4
				else if (std::all_of(c.second.cbegin(), c.second.cend(), [](int i){ return i%4==0;}) and c.second.size() == this->size()/4)
				{
					ss << "0mod4";
				}
				else if (std::all_of(c.second.cbegin(), c.second.cend(), [](int i){ return i%4==1;}) and c.second.size() == this->size()/4)
				{
					ss << "1mod4";
				}
				else if (std::all_of(c.second.cbegin(), c.second.cend(), [](int i){ return i%4==2;}) and c.second.size() == this->size()/4)
				{
					ss << "2mod4";
				}
				else if (std::all_of(c.second.cbegin(), c.second.cend(), [](int i){ return i%4==3;}) and c.second.size() == this->size()/4)
				{
					ss << "3mod4";
				}
				else
				{
					if (c.second.size() == 2)
					{
						ss << *c.second.begin() << "," << *c.second.rbegin(); // two sites
					}
					else
					{
						bool CONSECUTIVE = true;
						for (auto it=c.second.begin(); it!=c.second.end(); ++it)
						{
							if (next(it) != c.second.end() and *next(it)!=*it+1ul)
							{
								CONSECUTIVE = false;
							}
						}
						if (CONSECUTIVE)
						{
							ss << *c.second.begin() << "-" << *c.second.rbegin(); // range of sites
						}
						else
						{
							for (auto it=c.second.begin(); it!=c.second.end(); ++it)
							{
								ss << *it << ","; // some unknown order
							}
							ss.seekp(-1,ios_base::end); // delete last comma
						}
					}
				}
			}
			//			ss.seekp(-1,ios_base::end); // delete last comma
			ss << ": " << c.first << std::endl;
		}
	}
	
	this->set_name(ss.str());
}

template<typename Symmetry, typename Scalar>
template<typename CouplScalar>
void Mpo<Symmetry,Scalar>::
construct_from_pushlist(const PushType<OperatorType,CouplScalar>& pushlist, const std::vector<std::vector<std::string>>& labellist, size_t Lcell)
{
    for(std::size_t i=0; i<pushlist.size(); ++i)
    {
        const auto& [loc, ops, coupling] = pushlist[i];
		if ( std::abs(coupling) != 0. )
		{
			this->push(loc, ops, coupling);
		}
    }
    for(std::size_t loc=0; loc<this->size(); ++loc)
    {
        for(std::size_t i=0; i<labellist[loc].size(); ++i)
        {
            this->save_label(loc, labellist[loc][i]);
        }
    }
	generate_label(Lcell);
}


template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocal(std::size_t loc, const OperatorType& op)
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
	LocalOp   = op;
	LocalSite = loc;
    this->push(loc, {op});
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocal(std::size_t loc, const OperatorType& op, const OperatorType& signOp)
{
    std::vector<OperatorType> signOps(loc, signOp);
    setLocal(loc, op, signOps);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocal(std::size_t loc, const OperatorType& op, const std::vector<OperatorType>& signOps)
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
    assert(signOps.size() == loc and "Number of sign operators does not match the chosen lattice site");
    LocalOp   = op;
    LocalSite = loc;
    std::vector<OperatorType> ops = signOps;
    ops.push_back(op);
    this->push(0, ops);
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocalStag(std::size_t loc, const OperatorType& op, const std::vector<OperatorType>& stagSignOps)
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
    assert(stagSignOps.size() == this->size() and "Number of staggered sign operators does not match the chosen lattice size");
    LocalOp   = op;
    LocalSite = loc;
    
    std::vector<OperatorType> ops(this->size());
    ops[loc] = op;
    for(std::size_t loc2=0; loc2<this->size(); ++loc2)
    {
        if(loc2 != loc)
        {
            ops[loc2] = stagSignOps[loc2];
        }
    }
    this->push(0, ops);
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocal(const std::vector<std::size_t>& locs, const std::vector<OperatorType>& ops)
{
    auto Id = ops[0];
    Id.setIdentity();
    Id.label = "id";
    setLocal(locs, ops, Id);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocal(const std::vector<std::size_t>& locs, const std::vector<OperatorType>& ops, const std::vector<OperatorType>& signOps)
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
    assert(locs.size() == 2 and "setLocal(...) only works for two operators!");
    int left = 0;
    int right = 1;
    if(locs[left] > locs[right])
    {
        left = 1;
        right = 0;
    }
    assert(locs[left] != locs[right] and "setLocal(...) needs to local operators at different sites!");
    assert(signOps.size() == locs[right]-locs[left]-1);
    
    std::vector<OperatorType> ops_with_signs;
    ops_with_signs.push_back(ops[left]);
    for(std::size_t pos=0; pos<signOps.size(); ++pos)
    {
        ops_with_signs.push_back(signOps[pos]);
    }
    ops_with_signs.push_back(ops[right]);
    this->push(locs[left], ops_with_signs);
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocal(const std::vector<std::size_t>& locs, const std::vector<OperatorType>& ops, const OperatorType& signOp)
{
    assert(locs.size() == 2 and "setLocal(...) only works for two operators!");
    assert(locs[0] != locs[1] and "setLocal(...) needs to local operators at different sites!");
    int distance = locs[1]-locs[0];
    if(distance<0)
    {
        distance *= -1;
    }
    std::vector<OperatorType> signOps(distance-1, signOp);
    setLocal(locs, ops, signOps);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocalSum(const OperatorType& op, Scalar (*f)(int))
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
    for (std::size_t loc=0; loc<this->size(); ++loc)
    {
        this->push(loc, {f(loc)*op});
    }
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setLocalSum(const std::vector<OperatorType>& ops, std::vector<Scalar> coeffs)
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
    for (std::size_t loc=0; loc<this->size(); ++loc)
    {
        this->push(loc, {coeffs[loc]*ops[loc]});
    }
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
setProductSum(const OperatorType& op1, const OperatorType& op2)
{
    assert(this->check_qPhys() and "Physical bases have to be set before");
    for(std::size_t loc=0; loc<this->size()-1; ++loc)
    {
        this->push(loc, {op1, op2});
    }
    this->finalize(PROP::COMPRESS, 1);
}

template<typename Symmetry, typename Scalar> void Mpo<Symmetry,Scalar>::
scale(double factor, double offset)
{
	if (LocalSite != -1)
	{
		auto Id = LocalOp;
		Id.setIdentity();
        Id.label = "id";
		if(std::abs(factor - 1.0) > ::mynumeric_limits<Scalar>::epsilon())
        {
            LocalOp = factor * LocalOp;
        }
		if(std::abs(offset) > ::mynumeric_limits<Scalar>::epsilon())
        {
            LocalOp += offset * Id;
        }
		setLocal(LocalSite, LocalOp, this->get_boundary_condition());
	}
	else
	{
		this->scale(factor, offset);
        this->finalize(PROP::COMPRESS, 1);
	}
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
precalc_TwoSiteData(bool FORCE)
{
	if(!GOT_TWO_SITE_DATA or FORCE)
    {
        TSD.clear();
        TSD = this->calc_TwoSiteData();
        GOT_TWO_SITE_DATA = true;
    }
}

template<typename Symmetry, typename Scalar>
std::ostream &operator<<(std::ostream& os, const Mpo<Symmetry,Scalar>& O)
{
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "Mpo: L=" << O.length();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for(std::size_t loc=0; loc<O.length(); ++loc)
	{
		for(std::size_t s1=0; s1<O.locBasis(loc).size(); ++s1)
        {
            for(std::size_t s2=0; s2<O.locBasis(loc).size(); ++s2)
            {
                for(std::size_t k=0; k<O.opBasis(loc).size(); ++k)
                {
                    // TODO: Angepasster Code für W=Biped
                    /*
                    if(O.W_at(loc)[s1][s2][k].nonZeros() > 0)
                    {
                        std::array<typename Symmetry::qType,3> qCheck = {O.locBasis(loc)[s2],O.opBasis(l)[k],O.locBasis(loc)[s1]};
                        if(!Symmetry::validate(qCheck))
                        {
                            continue;
                        }
                        os << "[l=" << l << "]\t|" << Sym::format<Symmetry>(O.locBasis(loc)[s1]) << "><" << Sym::format<Symmetry>(O.locBasis(loc)[s2]) << "|:" << std::endl;
                        os << Matrix<Scalar,Dynamic,Dynamic>(O.W_at(loc)[s1][s2][k]) << std::endl;
                    }
                    */
                    
                }
            }
        }
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if(loc != O.length()-1)
        {
            os << std::endl;
        }
	}
	return os;
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void compare(const Mpo<Symmetry,Scalar1>& O1, const Mpo<Symmetry,Scalar2>& O2)
{
	lout << setfill('-') << setw(30) << "-" << setfill(' ');
	lout << "Mpo: L=" << O1.length();
	lout << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (std::size_t loc=0; loc<O1.length(); ++loc)
	{
		for (std::size_t s1=0; s1<O1.locBasis(loc).size(); ++s1)
        {
            for (std::size_t s2=0; s2<O1.locBasis(loc).size(); ++s2)
            {
                for (std::size_t k=0; k<O1.opBasis(loc).size(); ++k)
                {
                    // TODO: Angepasster Code für W=Biped
                    /*
                     lout << "[l=" << loc << "]\t|" << Sym::format<Symmetry>(O1.locBasis(loc)[s1]) << "><" << Sym::format<Symmetry>(O1.locBasis(loc)[s2]) << "|:" << std::endl;
                    auto M1 = Matrix<Scalar1,Dynamic,Dynamic>(O1.W_at(loc)[s1][s2][k]);
                    auto Mtmp = Matrix<Scalar2,Dynamic,Dynamic>(O2.W_at(loc)[s1][s2][k]);
                    auto M2 = Mtmp.template cast<Scalar1>();
                    lout << "norm(diff)=" << (M1-M2).norm() << endl;
                    if((M1-M2).norm() > ::mynumeric_limits<Scalar1>::epsilon() or (M1-M2).norm() > ::mynumeric_limits<Scalar2>::epsilon())
                    {
                        lout << "M1=" << endl << M1 << endl << endl;
                        lout << "M2=" << endl << Matrix<Scalar2,Dynamic,Dynamic>(O2.W_at(loc)[s1][s2][k]) << endl << std::endl;
                    }
                    */
                }
            }
        }
		lout << setfill('-') << setw(80) << "-" << setfill(' ');
		if(loc != O1.length()-1)
        {
            lout << std::endl;
        }
	}
	lout << std::endl;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> Mpo<Symmetry,Scalar>::
cast_Terms_to_Mpo(const MpoTerms<Symmetry,Scalar>& input)
{
    Mpo<Symmetry,Scalar> output(input.size(), input.get_qTot(), input.label, false, false, false, input.get_boundary_condition());
    output.reconstruct(input.get_O(), input.get_qAux(), input.get_qPhys(), input.is_finalized(), input.get_boundary_condition(), input.get_qTot());
    return output;
}

#endif
