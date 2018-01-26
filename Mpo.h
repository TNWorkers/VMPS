#ifndef STRAWBERRY_Mpo_WITH_Q
#define STRAWBERRY_Mpo_WITH_Q

#include "boost/multi_array.hpp"

#include <Eigen/SparseCore>
#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
typedef Eigen::SparseMatrix<double,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
using namespace Eigen;

#include "symmetry/U0.h"
#include "SuperMatrix.h"
#include "symmetry/qarray.h"
#include "symmetry/qbasis.h"
#include "tensors/Biped.h"
#include "pivot/DmrgPivotStuff1.h"
#include <unsupported/Eigen/KroneckerProduct>
#include "DmrgJanitor.h"
#include "DmrgExternal.h"
#if !defined DONT_USE_LAPACK_SVD || !defined DONT_USE_LAPACK_QR
	#include "LapackWrappers.h"
#endif

/**Namespace VMPS to distinguish names from ED equivalents.*/
namespace VMPS{};

//Forward declarations
template<typename Symmetry, typename Scalar> class Mps;
template<typename Symmetry, typename Scalar> class Mpo;
template<typename Symmetry, typename MpHamiltonian, typename Scalar> class DmrgSolver;
template<typename Symmetry, typename Scalar, typename MpoScalar> class MpsCompressor;
template<typename Symmetry, typename MpHamiltonian, typename Scalar> class VumpsSolver;

/**
 * Matrix Product Operator with conserved quantum numbers (Abelian and non-abelian symmetries).
 * Just adds a target quantum number and a bunch of labels on top of Mpo.
 * \describe_Symmetry
 * \describe_Scalar
 */
template<typename Symmetry, typename Scalar=double>
class Mpo
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	typedef SparseMatrixXd SparseMatrixType;
	typedef SiteOperator<Symmetry,Scalar> OperatorType;
	static constexpr size_t Nq = Symmetry::Nq;
	typedef typename Symmetry::qType qType;

	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class DmrgSolver;
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class VumpsSolver;
	template<typename Symmetry_, typename S1, typename S2> friend class MpsCompressor;
	template<typename H, typename Symmetry_, typename S1, typename S2, typename V> friend class TDVPPropagator;
	template<typename Symmetry_, typename S_> friend class Mpo;

	template<typename Symmetry_, typename S1, typename S2> friend void HxV  (const Mpo<Symmetry_,S1> &H,
																			 const Mps<Symmetry_,S2> &Vin,
																			 Mps<Symmetry_,S2> &Vout,
																			 DMRG::VERBOSITY::OPTION VERBOSITY);

	template<typename Symmetry_, typename S1, typename S2> friend void OxV (const Mpo<Symmetry_,S1> &H,
																			const Mps<Symmetry_,S2> &Vin,
																			Mps<Symmetry_,S2> &Vout,
																			DMRG::BROOM::OPTION TOOL);
public:
	
	Mpo(){};
	
	Mpo (size_t L_input);
	
	/**
	 * Basic Mpo constructor.
	 * \warning Note that qloc and qOp have to be set separately.
	 * \param L_input : amount of sites/supersites which are swept
	 * \param Qtot_input : the total change in quantum number
	 * \param qlabel_input : how to label the conserved quantum numbers in outputs
	 * \param label_input : how to label the Mpo itself in outputs
	 * \param format_input : format function for the quantum numbers (e.g.\ half-integers for S=1/2).
	 * \param UNITARY_input : if the Mpo is known to be unitary, this can be further exploited
	 */
	Mpo (size_t L_input, 
	     qarray<Nq> Qtot_input,
	     std::array<string,Nq> qlabel_input=defaultQlabel<Nq>(), 
	     string label_input="Mpo", 
	     string (*format_input)(qarray<Nq> qnum)=noFormat, 
	     bool UNITARY_input=false);
	
	//---set whole Mpo for special cases, modify---
	
	///\{
	/**
	 * Set to a local operator \f$O_i\f$
	 * \param loc : site index
	 * \param Op : the local operator in question
	 */
	void setLocal (size_t loc, const OperatorType &Op, bool OPEN_BC=true);
	
	/**
	 * Set to a local operator \f$O_i\f$ but add a chain of sign operators (useful for fermionic operators)
	 * \param loc : site index
	 * \param Op : the local operator in question
	 * \param SignOp : elementary operator for the sign chain.
	 */
	void setLocal (size_t loc, const OperatorType& Op, const OperatorType &SignOp, bool OPEN_BC=true);
	
	/**
	 * Set to a product of local operators \f$O^1_i O^2_j O^3_k \ldots\f$
	 * \param loc : list of locations
	 * \param Op : list of operators
	*/
	void setLocal (const vector<size_t> &loc, const vector<OperatorType> &Op, bool OPEN_BC=true);
	
	/**
	 * Set to a product of local operators \f$O^1_i O^2_j O^3_k \ldots\f$ with sign chains in between
	 * \param loc : list of locations
	 * \param Op : list of operators
	 * \param SignOp : elementary operator for the sign chain.
	 */
	void setLocal (const vector<size_t> &loc, const vector<OperatorType> &Op, const OperatorType& SignOp, bool OPEN_BC=true);
	
	/**
	 * Set to a sum of of local operators \f$\sum_i f(i) O_i\f$
	 * \param Op : the local operator in question
	 * \param f : the function in question$
	 */
	void setLocalSum (const OperatorType &Op, Scalar (*f)(int)=localSumTrivial, bool OPEN_BC=true);
	
	/**
	 * Set to a sum of nearest-neighbour products of local operators \f$\sum_i O^1_i O^2_{i+1}\f$
	 * \param Op1 : first local operator
	 * \param Op2 : second local operator
	 */
	void setProductSum (const OperatorType &Op1, const OperatorType &Op2, bool OPEN_BC=true);
	
	/**Makes a linear transformation of the Mpo: \f$H' = factor*H + offset\f$.*/
	void scale (double factor=1., double offset=0.);
	///\}
	
	//---info stuff---
	
	///\{
	/**\describe_info*/
	string info() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**
	 * Calculates a measure of the sparsity of the given Mpo.
	 * \param USE_SQUARE : If \p true, apply it to the stored square.
	 * \param PER_MATRIX : If \p true, calculate the amount of non-zeros per matrix. If \p false, calculate the fraction of non-zero elements.
	 */
	double sparsity (bool USE_SQUARE=false, bool PER_MATRIX=true) const;
	///\}
	
	//---formatting stuff---
	
	///\{
	/**Format function for the quantum numbers (e.g.\ half-integers for S=1/2).*/
	string (*format)(qarray<Nq> qnum);
	
	/**How this Mpo should be called in outputs.*/
	string label;
	
	/**Label for quantum numbers in output (e.g.\ \f$M\f$ for magnetization; \f$N_\uparrow\f$,\f$N_\downarrow\f$ for particle numbers etc.).*/
	std::array<string,Nq> qlabel;
	///\}
	
	//---return stuff, set parts, check stuff---
	
	///\{
	/**Returns the length of the chain (the amount of sites or supersites which are swept).*/
	inline size_t length() const {return N_sites;}
	
	/**Returns the volume of the chain (all the physical sites).*/
	inline size_t volume() const {return N_phys;}
	
	/**\describe_Daux*/
	inline size_t auxdim() const {return Daux;}
	
	/**Returns the total quantum number of the Mpo.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
	
	/**Sets the total quantum number of the Mpo.*/
	inline void setQtarget(const qType& Q) {Qtot=Q;};
	
	/**Returns the local basis at \p loc.*/
	inline vector<qarray<Nq> > locBasis   (size_t loc) const {return qloc[loc];}
	/**Returns the right auxiliary basis at \p loc.*/
	inline Qbasis<Symmetry>    auxBasis   (size_t loc) const {return qaux[loc];}
	
	/**Returns the operator basis at \p loc.*/
	inline vector<qarray<Nq> > opBasis   (size_t loc) const {return qOp[loc];}
	/**Returns the operator basis of the squared Mpo at \p loc.*/
	inline vector<qarray<Nq> > opBasisSq (size_t loc) const {return qOpSq[loc];}
	
	/**Returns the full local basis.*/
	inline vector<vector<qarray<Nq> > > locBasis()   const {return qloc;}
	/**Returns the full auxiliary basis.*/
	inline vector<Qbasis<Symmetry> >    auxBasis()   const {return qaux;}
	
	/**Returns the full operator basis.*/
	inline vector<vector<qarray<Nq> > > opBasis()   const {return qOp;}
	/**Returns the full operator basis of the squared Mpo.*/
	inline vector<vector<qarray<Nq> > > opBasisSq() const {return qOpSq;}
	
	/**Sets the local basis at \p loc.*/
	inline void setLocBasis (const vector<qType> &q,    size_t loc) {qloc[loc]=q;}
	
	/**Sets the operator basis at \p loc.*/
	inline void setOpBasis (const vector<qType>& q, size_t loc) {qOp[loc] = q;}
	
	/**Sets the full local basis.*/
	inline void setLocBasis (const vector<vector<qType> >    &q) {qloc=q;}
	
	/**Sets the full operator basis.*/
	inline void setOpBasis   (const vector<vector<qType> > &q)        {qOp=q;}
	/**Sets the full operator basis of the squared Mpo.*/
	inline void setOpBasisSq (const vector<vector<qType> > &qOpSq_in) {qOpSq=qOpSq_in;}
	
	/**Checks whether the Mpo is a unitary operator.*/
	inline bool IS_UNITARY() const {return UNITARY;};
	
	/**Checks if the square of the Mpo was calculated and stored.*/
	inline bool check_SQUARE() const {return GOT_SQUARE;}
	
	/**Returns the W-matrix at a given site by const reference.*/
	inline const vector<vector<vector<SparseMatrix<Scalar> > > > &W_at   (size_t loc) const {return W[loc];};
	
	/**Returns the W-matrix of the squared operator at a given site by const reference.*/
	inline const vector<vector<vector<SparseMatrix<Scalar> > > > &Wsq_at (size_t loc) const {return Wsq[loc];};
	
	/**\warning Needs updating to new Mpo scheme.*/
//	template<typename TimeScalar> Mpo<Symmetry,TimeScalar> BondPropagator (TimeScalar dt, PARITY P) const;
	
	/**Reconstructs the full two-site Hamiltonian from the Mpo entries at \p loc1 and \p loc2. Needed for VUMPS.*/
	boost::multi_array<Scalar,4> H2site (size_t loc1, size_t loc2, bool HALF_THE_LOCAL_TERM=false) const;
	///\}
	
	//--- compression and propagation stuff (do not delete, could still be of use) ---/
	
//	/**Resets the Mpo from a dummy Mps which has been swept.*/
//	void setFromFlattenedMpo (const Mps<Sym::U0,Scalar> &Op, bool USE_SQUARE=false);
//	
//	/**Sets the product of a left-side and right-side operator in the Heisenberg picture.*/
//	template<typename OtherSymmetry> void set_HeisenbergPicture (const Mpo<OtherSymmetry,Scalar> Op1, const Mpo<OtherSymmetry,Scalar> Op2);
//	
//	void SVDcompress (bool USE_SQUARE=false, double eps_svd=1e-7, size_t N_halfsweeps=2);
//	string test_ortho() const;
//	void init_compression();
//	void rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixQ<Nq,Scalar> *H=NULL);
//	void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixQ<Nq,Scalar> *H=NULL);
//	void flatten_to_Mps (Mps<0,Scalar> &V);
//	vector<vector<vector<MatrixType> > > A;
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef Mps<Symmetry,double>                              StateXd;
	typedef Mps<Symmetry,complex<double> >                    StateXcd;
	typedef DmrgSolver<Symmetry,Mpo<Symmetry,Scalar>,Scalar>  Solver;
	typedef VumpsSolver<Symmetry,Mpo<Symmetry,Scalar>,Scalar> uSolver;
	typedef MpsCompressor<Symmetry,double,double>             CompressorXd;
	typedef MpsCompressor<Symmetry,complex<double>,double>    CompressorXcd;
	typedef Mpo<Symmetry>                                     Operator;
	///@}
	
protected:
	
	vector<vector<qarray<Nq> > > qloc, qOp, qOpSq;
	vector<Qbasis<Symmetry> > qaux;
	
	qarray<Nq> Qtot;
	
	bool UNITARY    = false;
	bool GOT_SQUARE = false;
	
	size_t N_sites;
	size_t N_phys=0;
	size_t Daux;
	
	/**Resizes the relevant containers with \p N_sites.*/
	void initialize();
	
	/**Calculates the auxiliary basis.*/
	void calc_auxBasis();
	
	/**Construct with \p vector<SuperMatrix> and input \p qOp. Most general of the construct routines, all the source code is here.*/
	void construct (const vector<SuperMatrix<Symmetry,Scalar> > &Gvec_input,
	                vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &Wstore,
	                vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
	                bool CALC_SQUARE=false,
	                bool OPEN_BC=true);
	
	/**Construct with \p SuperMatrix (homogeneously extended) and input \p qOp.*/
	void construct (const SuperMatrix<Symmetry,Scalar> &G_input,
	                vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &Wstore,
	                vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
	                const vector<vector<qType> > &qOp_in,
	                bool CALC_SQUARE=false,
	                bool OPEN_BC=true);
	
	/**Construct with \p SuperMatrix and stored \p qOp.*/
	void construct (const SuperMatrix<Symmetry,Scalar> &G_input,
	                vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &Wstore,
	                vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
	                bool CALC_SQUARE=false,
	                bool OPEN_BC=true);
	
	/**Construct with \p vector<SuperMatrix> and stored \p qOp.*/
	void construct (const vector<SuperMatrix<Symmetry,Scalar> > &Gvec_input,
	                vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &Wstore,
	                vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
	                const vector<vector<qType> > &qOp_in,
	                bool CALC_SQUARE=false,
	                bool OPEN_BC=true);
	
	vector<SuperMatrix<Symmetry,Scalar> > make_localG (size_t loc, const OperatorType &Op);
	vector<SuperMatrix<Symmetry,Scalar> > make_localG (const vector<size_t> &loc, const vector<OperatorType> &Op);
	
	vector<SuperMatrix<Symmetry,Scalar> > Gvec;
	vector<vector<vector<vector<SparseMatrix<Scalar> > > > > W;
	
	vector<SuperMatrix<Symmetry,Scalar> > GvecSq;
	vector<vector<vector<vector<SparseMatrix<Scalar> > > > > Wsq;
	
	/**Generates the Mpo label from the info stored in \p HamiltonianTerms.*/
	void generate_label (string mainlabel, const vector<HamiltonianTerms<Symmetry,Scalar> > &Terms, size_t Lcell);
	
	// compression stuff (do not delete, could still be of use)
	// ArrayXd truncWeight;
};

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar>::
Mpo (size_t L_input)
:N_sites(L_input)
{
	initialize();
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar>::
Mpo (size_t L_input, qarray<Nq> Qtot_input, 
     std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum), 
     bool UNITARY_input)
:N_sites(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input), UNITARY(UNITARY_input)
{
	initialize();
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
initialize()
{
	qloc.resize(N_sites);
	qOp.resize(N_sites);
	W.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		W[l].resize(qloc[l].size());
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		{
			W[l][s1].resize(qloc[l].size());
			for (size_t s2=0; s2<qloc[l].size(); ++s2)
			{
				W[l][s1][s2].resize(1);
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
construct (const SuperMatrix<Symmetry,Scalar> &G_input,
           vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &Wstore,
           vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
           bool CALC_SQUARE,
           bool OPEN_BC)
{
	construct(G_input, Wstore, Gstore, this->qOp, CALC_SQUARE, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
construct (const vector<SuperMatrix<Symmetry,Scalar> > &Gvec_input,
           vector<vector<vector<vector<SparseMatrix<Scalar> > > > >  &Wstore,
           vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
           bool CALC_SQUARE,
           bool OPEN_BC)
{
	construct(Gvec_input, Wstore, Gstore, this->qOp, CALC_SQUARE, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
construct (const SuperMatrix<Symmetry,Scalar> &G_input,
           vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &Wstore,
           vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
           const vector<vector<qType> > &qOp_in,
           bool CALC_SQUARE,
           bool OPEN_BC)
{
	vector<SuperMatrix<Symmetry,Scalar> > Gvec(N_sites);
	size_t D = G_input(0,0).data.rows();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Gvec[l].setMatrix(G_input.auxdim(),D);
		Gvec[l] = G_input;
	}
	
	construct(Gvec, Wstore, Gstore, qOp_in, false, OPEN_BC);
	
	// make squared Mpo if desired
	if (CALC_SQUARE == true)
	{
		qOpSq.resize(N_sites);
		for(size_t l=0; l<N_sites; l++)
		{
			qOpSq[l] = Symmetry::reduceSilent(qOp[l],qOp[l]);
		}
		construct(tensor_product(G_input,G_input), Wsq, GvecSq, qOpSq, false, OPEN_BC); //use false here, otherwise one would also calclate H⁴.
		GOT_SQUARE = true;
	}
	else
	{
		GOT_SQUARE = false;
	}
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
construct (const vector<SuperMatrix<Symmetry,Scalar> > &Gvec_input,
           vector<vector<vector<vector<SparseMatrix<Scalar> > > > >  &Wstore,
           vector<SuperMatrix<Symmetry,Scalar> > &Gstore,
           const vector<vector<qType> > &qOp_in,
           bool CALC_SQUARE,
           bool OPEN_BC)
{
	Wstore.resize(N_sites);
	Gstore = Gvec_input;
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Wstore[l].resize(qloc[l].size());
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		{
			Wstore[l][s1].resize(qloc[l].size());
		}
	}
	
	// open boundary conditions: use only last row
	if (OPEN_BC)
	{
		size_t l=0;
		
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		{
			Wstore[l][s1][s2].resize(qOp_in[l].size());
			for (size_t k=0; k<qOp_in[l].size(); ++k)
			{
				Wstore[l][s1][s2][k].resize(1,Gstore[l].cols());
			}
			for (size_t a2=0; a2<Gstore[l].cols(); ++a2)
			{
				Scalar val = Gstore[l](Gstore[l].rows()-1,a2).data.coeffRef(s1,s2);
				if (val != 0.)
				{
					qType Q = Gstore[l](Gstore[l].rows()-1,a2).Q;
					size_t match;
					for (size_t k=0; k<qOp_in[l].size(); ++k)
					{
						if(qOp_in[l][k] == Q) {match=k; break;}
						// assert(k == qOp[l].size()-1 and "The SuperMatrix is not well defined.");
					}
					Wstore[l][s1][s2][match].insert(0,a2) = val;
				}
			}
		}
	}
	
	size_t l_frst = (OPEN_BC)? 1:0;
	size_t l_last = (OPEN_BC)? N_sites-1:N_sites;
	
	for (size_t l=l_frst; l<l_last; ++l)
	for (size_t s1=0; s1<qloc[l].size(); ++s1)
	for (size_t s2=0; s2<qloc[l].size(); ++s2)
	{
		Wstore[l][s1][s2].resize(qOp_in[l].size());
		for (size_t k=0; k<qOp_in[l].size(); ++k)
		{
			Wstore[l][s1][s2][k].resize(Gstore[l].rows(), Gstore[l].cols());
		}
		for (size_t a1=0; a1<Gstore[l].rows(); ++a1)
		for (size_t a2=0; a2<Gstore[l].cols(); ++a2)
		{
			Scalar val = Gstore[l](a1,a2).data.coeffRef(s1,s2);
			if (val != 0.)
			{
				qType Q = Gstore[l](a1,a2).Q;
				size_t match;
				for(size_t k=0; k<qOp_in[l].size(); ++k)
				{
					if (qOp_in[l][k] == Q) {match=k; break;}
					// assert(k == qOp[l].size()-1 and "The SuperMatrix is not well defined.");
				}
				Wstore[l][s1][s2][match].insert(a1,a2) = val;
			}
		}
	}
	
	// open boundary conditions: use only first column
	if (OPEN_BC)
	{
		size_t l=l_last;
		
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		{
			Wstore[l][s1][s2].resize(qOp_in[l].size());
			for (size_t k=0; k<qOp_in[l].size(); ++k)
			{
				Wstore[l][s1][s2][k].resize(Gstore[l].rows(),1);
			}
			for (size_t a1=0; a1<Gstore[l].rows(); ++a1)
			{
				Scalar val = Gstore[l](a1,0).data.coeffRef(s1,s2);
				if (val != 0.)
				{
					qType Q = Gstore[l](a1,0).Q;
					size_t match;
					for(size_t k=0; k<qOp_in[l].size(); ++k)
					{
						if (qOp_in[l][k] == Q) {match=k; break;}
						// assert(k == qOp[l].size()-1 and "The SuperMatrix is not well defined.");
					}
					Wstore[l][s1][s2][match].insert(a1,0) = val;
				}
			}
		}
	}
	
	// make squared Mpo if desired
	if (CALC_SQUARE == true)
	{
		qOpSq.resize(N_sites);
		vector<SuperMatrix<Symmetry,Scalar> > GvecSq_tmp(N_sites);
		for (size_t l=0; l<N_sites; ++l)
		{
			qOpSq[l] = Symmetry::reduceSilent(qOp[l],qOp[l]);
			GvecSq_tmp[l].setMatrix(Gvec_input[l].auxdim()*Gvec_input[l].auxdim(), Gvec_input[l].D());
			GvecSq_tmp[l] = tensor_product(Gvec_input[l],Gvec_input[l]);
		}
		construct(GvecSq_tmp, Wsq, GvecSq, qOpSq, false, OPEN_BC); //use false here, otherwise one would also calclate H⁴.
		GOT_SQUARE = true;
	}
	else
	{
		GOT_SQUARE = false;
	}
	
	// auxiliary Basis
	calc_auxBasis();
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
calc_auxBasis()
{
	auto calc_qnums_on_segment = [this](int l_frst, int l_last) -> std::set<qType>
	{
		size_t L = (l_last < 0 or l_frst >= qOp.size())? 0 : l_last-l_frst+1;
		std::set<qType > qset;
		
		if (L > 0)
		{
			// add qnums of local basis on l_frst to qset_tmp
			std::set<qType> qset_tmp;
			
			for (const auto& k : qOp[l_frst])
			{
				qset_tmp.insert(k);
			}
			
			for (size_t l=l_frst+1; l<=l_last; ++l)
			{
				for (const auto& k : qOp[l])
				for (auto it=qset_tmp.begin(); it!=qset_tmp.end(); ++it)
				{
					auto qVec = Symmetry::reduceSilent(*it,k);
					for (const auto& q : qVec)
					{
						qset.insert(q);
					}
				}
				// swap qset and qset_tmp to continue
				std::swap(qset_tmp,qset);
			}
			qset = qset_tmp;
		}
		else
		{
			qset.insert(Symmetry::qvacuum());
		}
		return qset;
	};
	
	this->qaux.resize(this->N_sites+1);
	//set aux basis on right end to Qtot.
	qaux[this->N_sites].push_back(Qtot,1);//auxdim());
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		Qbasis<Symmetry> qauxtmp;
		std::unordered_set<qType> uniqueControl;
		int lprev = l-1;
		int lnext = l+1;
		std::set<qType> qlset = calc_qnums_on_segment(0,lprev); // length=l
		std::set<qType> qrset = calc_qnums_on_segment(lnext,this->N_sites-1); // length=L-l-1
		for (const auto& k : qOp[l])
		for (auto ql=qlset.begin(); ql!=qlset.end(); ++ql)
		{
			auto qVec = Symmetry::reduceSilent(*ql,k);
			std::vector<std::set<qType> > qrSetVec; qrSetVec.resize(qVec.size());
			for (size_t i=0; i<qVec.size(); i++)
			{
				auto qVectmp = Symmetry::reduceSilent(Symmetry::flip(qVec[i]),Qtot);
				for (size_t j=0; j<qVectmp.size(); j++) { qrSetVec[i].insert(qVectmp[j]); }
				for (auto qr = qrSetVec[i].begin(); qr!=qrSetVec[i].end(); qr++)
				{
					auto itqr = qrset.find(*qr);
					if (itqr != qrset.end())
					{
						auto qin = *ql;
						if(auto it=uniqueControl.find(qin) == uniqueControl.end())
						{
							uniqueControl.insert(qin);
							if(l==0) { qauxtmp.push_back(qin,1); }
							else { qauxtmp.push_back(qin,auxdim()); }
						}
					}
				}
			}
		}
		qaux[l] = qauxtmp;
	}
}

template<typename Symmetry, typename Scalar>
string Mpo<Symmetry,Scalar>::
info() const
{
	stringstream ss;
	ss << label << "L=" << N_sites;
	if (N_phys>N_sites) {ss << ",V=" << N_phys;}
	ss << ", ";
	
	ss << "Daux=" << Daux << ", ";
//	ss << "trunc_weight=" << truncWeight.sum() << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	ss << ", sparsity=" << sparsity();
	if (GOT_SQUARE) {ss << ", sparsity(sq)=" << sparsity(true);}
	return ss.str();
}

template<typename Symmetry, typename Scalar>
double Mpo<Symmetry,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	
	if (W.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		for (size_t k=0; k<qOp[l].size(); ++k)
		{
			res += calc_memory(W[l][s1][s2][k],memunit);
			if (GOT_SQUARE)
			{
				res += calc_memory(Wsq[l][s1][s2][k],memunit);
			}
		}
	}
	
	if (Gvec.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			res += Gvec[l].memory(memunit);
			if (GOT_SQUARE)
			{
				res += GvecSq[l].memory(memunit);
			}
		}
	}
	
	return res;
}

template<typename Symmetry, typename Scalar>
double Mpo<Symmetry,Scalar>::
sparsity (bool USE_SQUARE, bool PER_MATRIX) const
{
	if (USE_SQUARE) {assert(GOT_SQUARE);}
	double N_nonZeros = 0.;
	double N_elements = 0.;
	double N_matrices = 0.;
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_matrices += pow(qloc[l].size(),2);
		
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		for (size_t k=0; k<qOp[l].size(); ++k)
		{
			N_nonZeros += (USE_SQUARE)? Wsq[l][s1][s2][k].nonZeros() : W[l][s1][s2][k].nonZeros();
			N_elements += (USE_SQUARE)? Wsq[l][s1][s2][k].rows() * Wsq[l][s1][s2][k].cols():
			                            W[l][s1][s2][k].rows()   * W[l][s1][s2][k].cols();
		}
	}
	
	return (PER_MATRIX)? N_nonZeros/N_matrices : N_nonZeros/N_elements;
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
generate_label (string mainlabel, const vector<HamiltonianTerms<Symmetry,Scalar> > &Terms, size_t Lcell)
{
	stringstream ss;
	ss << mainlabel;
	
	map<string,set<size_t> > cells;
	
	for (size_t l=0; l<Terms.size(); ++l)
	{
		cells[Terms[l%Lcell].get_info()].insert(l%Lcell);
	}
	
	if (cells.size() == 1)
	{
		ss << "(" << Terms[0].get_info() << "): ";
	}
	else
	{
		vector<pair<string,set<size_t> > > cells_resort(cells.begin(), cells.end());
		
		// sort according to smallest l, not according to label
		sort(cells_resort.begin(), cells_resort.end(), 
		[](const pair<string,set<size_t> > &a, const pair<string,set<size_t> > &b) -> bool
		{
			return *min_element(a.second.begin(),a.second.end()) < *min_element(b.second.begin(),b.second.end());
		});
		
		ss << ":" << endl;
		for (auto c:cells_resort)
		{
			ss << " •l=";
			for (auto s:c.second)
			{
				ss << s << ",";
			}
			ss.seekp(-1,ios_base::end); // delete last comma
			ss << ": " << c.first << endl;
		}
	}
	
	label = ss.str();
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
setLocal (size_t loc, const OperatorType &Op, bool OPEN_BC)
{
	auto G = make_localG(loc,Op);
	construct(G, W, Gvec, false, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
setLocal (size_t loc, const OperatorType &Op, const OperatorType &SignOp, bool OPEN_BC)
{
	auto G = make_localG(loc,Op);
	for (size_t l=0; l<loc; ++l) {G[l](0,0) = SignOp;}
	construct(G, W, Gvec, false, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
vector<SuperMatrix<Symmetry,Scalar> > Mpo<Symmetry,Scalar>::
make_localG (size_t loc, const OperatorType &Op)
{
	assert(Op.data.rows() == qloc[loc].size() and Op.data.cols() == qloc[loc].size());
	assert(loc < N_sites);
	
	for (size_t l=0; l<N_sites; l++)
	{
		qOp[l].resize(1);
		qOp[l][0] = (l==loc) ? Op.Q : Symmetry::qvacuum();
	}
	
	Daux = 1;
	vector<SuperMatrix<Symmetry,Scalar> > G(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		G[l].setMatrix(Daux,qloc[l].size());
		if (l==loc) {G[l](0,0) = Op;}
		else        {G[l](0,0).data.setIdentity(); G[l](0,0).Q = Symmetry::qvacuum();}
	}
	
	return G;
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
setLocal (const vector<size_t> &loc, const vector<OperatorType> &Op, bool OPEN_BC)
{
	auto G = make_localG(loc,Op);
	construct(G, W, Gvec, false, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
setLocal (const vector<size_t> &loc, const vector<OperatorType> &Op, const OperatorType &SignOp, bool OPEN_BC)
{
//	assert(loc.size() >= 1 and Op.size() == loc.size());
//	
//	// For non-abelian symmetries, the operators have to be on different sites,
//	// since multiplication of the operators is only possible in the format of SiteOperatorQ and here SiteOperator is used.
//	// --> The multiplication has to be done in the model-classes, before calling setLocal().
//	if constexpr( Symmetry::NON_ABELIAN )
//	{
//		for(size_t pos1=0; pos1<loc.size(); pos1++)
//		for(size_t pos2=pos1+1; pos2<loc.size(); pos2++)
//		{
//			assert(loc[pos1] != loc[pos2] and
//				   "setLocal() can only be called with several operators on different sites, when using non-abelian symmetries.");
//		}
//	}
//	
//	auto [min,max] = std::minmax_element(loc.begin(),loc.end());
//	size_t locMin,locMax;
//	locMin = loc[min-loc.begin()];
//	locMax = loc[max-loc.begin()];
//	
//	Daux = 1;
//	vector<SuperMatrix<Symmetry,Scalar> > G(N_sites);
//	
//	for (size_t l=0; l<N_sites; l++)
//	{
//		qOp[l].resize(1);
//		bool GATE = true;
//		for (size_t pos=0; pos<loc.size(); pos++)
//		{
//			if (l==loc[pos]) {qOp[l][0] = Op[pos].Q; GATE=false;}
//			if (GATE)        {qOp[l][0] = Symmetry::qvacuum();}
//		}
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		G[l].setMatrix(Daux,qloc[l].size());
//		if (auto it=find(loc.begin(),loc.end(),l) == loc.end())
//		{
//			if (l>locMin and l<locMax) {G[l](0,0) = SignOp;}
//			else {G[l](0,0).data.setIdentity(); G[l](0,0).Q = Symmetry::qvacuum();}
//		}
//		else
//		{
//			G[l](0,0).data.setIdentity(); G[l](0,0).Q = Symmetry::qvacuum();
//		}
//	}
//	
//	for (size_t i=0; i<loc.size(); ++i)
//	{
//		assert(loc[i] < N_sites);
//		assert(Op[i].data.rows() == qloc[loc[i]].size() and Op[i].data.cols() == qloc[loc[i]].size());
//		
//		G[loc[i]](0,0).data = G[loc[i]](0,0).data * Op[i].data;
//		G[loc[i]](0,0).Q = Symmetry::reduceSilent(G[loc[i]](0,0).Q, Op[i].Q)[0]; // We can use the 0th component here.
//	}
	
	auto G = make_localG(loc,Op);
	
	auto [min,max] = minmax_element(loc.begin(),loc.end());
	size_t locMin = loc[min-loc.begin()];
	size_t locMax = loc[max-loc.begin()];
	for (size_t l=locMin+1; l<locMax; ++l) {G[l](0,0) = SignOp;}
	
	construct(G, W, Gvec, false, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
vector<SuperMatrix<Symmetry,Scalar> > Mpo<Symmetry,Scalar>::
make_localG (const vector<size_t> &loc, const vector<OperatorType> &Op)
{
	assert(loc.size() >= 1 and Op.size() == loc.size());
	
	// For non-abelian symmetries, the operators have to be on different sites,
	// since multiplication of the operators is only possible in the format of SiteOperatorQ and here SiteOperator is used.
	// --> The multiplication has to be done in the model-classes, before calling setLocal().
	if constexpr( Symmetry::NON_ABELIAN )
	{
		for(size_t pos1=0; pos1<loc.size(); pos1++)
		for(size_t pos2=pos1+1; pos2<loc.size(); pos2++)
		{
			assert(loc[pos1] != loc[pos2] and
			       "setLocal() can only be called with several operators on different sites, when using non-abelian symmetries.");
		}
	}
	
	Daux = 1;
	vector<SuperMatrix<Symmetry,Scalar> > G(N_sites);
	
	for (size_t l=0; l<N_sites; l++)
	{
		qOp[l].resize(1);
		bool GATE = true;
		for(size_t pos=0; pos<loc.size(); pos++)
		{
			if(l == loc[pos]) { qOp[l][0] = Op[pos].Q; GATE = false; }
			if( GATE ) { qOp[l][0] = Symmetry::qvacuum(); }
		}
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		G[l].setMatrix(Daux,qloc[l].size());
		G[l](0,0).data.setIdentity();
		G[l](0,0).Q = Symmetry::qvacuum();
	}
	
	for (size_t i=0; i<loc.size(); ++i)
	{
		assert(loc[i] < N_sites);
		assert(Op[i].data.rows() == qloc[loc[i]].size() and Op[i].data.cols() == qloc[loc[i]].size());
		G[loc[i]](0,0).data = G[loc[i]](0,0).data * Op[i].data;
		G[loc[i]](0,0).Q = Symmetry::reduceSilent(G[loc[i]](0,0).Q,Op[i].Q)[0]; // We can use the 0th component here.
	}
	
	return G;
}

// sum_i f(i)*O(i)
template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
setLocalSum (const OperatorType &Op, Scalar (*f)(int), bool OPEN_BC)
{
	for (size_t l=0; l<N_sites; ++l)
	{
		assert(Op.data.rows() == qloc[l].size() and Op.data.cols() == qloc[l].size());
		if (Op.Q == Symmetry::qvacuum())
		{
			qOp[l].resize(1); qOp[l][0] = Symmetry::qvacuum();
		}
		else
		{
			qOp[l].resize(2); qOp[l][0] ==Symmetry::qvacuum();
			qOp[l][1] = Op.Q;
		}
	}
	
	Daux = 2;
	vector<SuperMatrix<Symmetry,Scalar> > G(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		G[l].setMatrix(Daux,qloc[l].size());
		G[l](0,0).data.setIdentity();
		G[l](0,0).Q = Symmetry::qvacuum();
		G[l](0,1).data.setZero();
		G[l](0,1).Q = Symmetry::qvacuum();
		G[l](1,0).data = f(l) * Op.data;
		G[l](1,0).Q = Op.Q;
		G[l](1,1).data.setIdentity();
		G[l](1,1).Q = Symmetry::qvacuum();
	}
	
	construct(G, W, Gvec, false, OPEN_BC);
}

// O1(1)*O2(2)+O1(2)*O1(3)+...+O1(L-1)*O2(L)
template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
setProductSum (const OperatorType &Op1, const OperatorType &Op2, bool OPEN_BC)
{
	bool BOTH_VACUUM=false, ONE_VACUUM=false;
	if (Op1.Q == Symmetry::qvacuum() and Op2.Q == Symmetry::qvacuum()) {BOTH_VACUUM=true;}
	if (Op1.Q == Symmetry::qvacuum() xor Op2.Q == Symmetry::qvacuum()) {ONE_VACUUM=true;}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		assert(Op1.data.rows() == qloc[l].size() and Op1.data.cols() == qloc[l].size() and 
		       Op2.data.rows() == qloc[l].size() and Op2.data.cols() == qloc[l].size());
		
		if (BOTH_VACUUM)
		{
			qOp[l].resize(1);
			qOp[l][0] = Symmetry::qvacuum();
		}
		else if (ONE_VACUUM)
		{
			qOp[l].resize(2);
			qOp[l][0] = Symmetry::qvacuum();
			qOp[l][1] = (Op1.Q == Symmetry::qvacuum()) ? Op2.Q : Op1.Q;
		}
		else
		{
			qOp[l].resize(2);
			qOp[l][0] = Symmetry::qvacuum();
			qOp[l][1] = Op1.Q;
			qOp[l][1] = Op2.Q;
		}
	}
	
	Daux = 3;
	vector<SuperMatrix<Symmetry,Scalar> > G(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		G[l].setMatrix(Daux,qloc[l].size());
		G[l].setZero();
		G[l](0,0).data.setIdentity();
		G[l](0,0).Q = Symmetry::qvacuum();
		G[l](1,0).data = Op1.data;
		G[l](1,0).Q = Op1.Q;
		G[l](2,1).Q = Op2.Q;
		G[l](2,2).data.setIdentity();
		G[l](2,2).Q = Symmetry::qvacuum();
	}
	
	construct(G, W, Gvec, false, OPEN_BC);
}

template<typename Symmetry, typename Scalar>
void Mpo<Symmetry,Scalar>::
scale (double factor, double offset)
{
	/**Example for where to apply the scaling factor, 3-site Heisenberg:
	\f$\left(-f \cdot B_x \cdot S^x_1, -f \cdot J \cdot S^z_1, I\right)
	
	\cdot 
	
	\left(
	\begin{array}{lll}
	I & 0 & 0 \\
	S^z_2 & 0 & 0 \\
	-f\cdot B_x\cdot S^x_2 & -f\cdot J\cdot S^z_2 & I
	\end{array}
	\right)
	
	\cdot
	
	\left(
	\begin{array}{l}
	I \\
	S^z_3 \\
	-f\cdot B_x \cdot S^x_3
	\end{array}
	\right)
	
	= -f \cdot B_x \cdot (S^x_1 + S^x_2 + S^x_3) - f \cdot J \cdot (S^z_1 \cdot S^z_2 + S^z_2 \cdot S^z_3)
	= f \cdot H\f$*/
	
	// apply to Gvec
	if (factor != 1.)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			size_t a1 = (l==0)? 0 : Daux-1;
			for (size_t a2=0; a2<Daux-1; ++a2)
			{
				Gvec[l](a1,a2) *= factor;
			}
		}
		Gvec[N_sites-1](Daux-1,0) *= factor;
	}
	
	if (offset != 0.)
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			size_t a1 = (l==0)? 0 : Daux-1;
			MatrixType Id(Gvec[l](a1,0).rows(), Gvec[l](a1,0).cols());
			Id.setIdentity();
			Gvec[l](a1,0) += offset/N_sites * Id;
		}
	}
	
	// calc W from Gvec
	if (factor != 1.)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			size_t a1 = (l==0)? 0 : Daux-1;
			for (size_t s1=0; s1<qloc[l].size(); ++s1)
			for (size_t s2=0; s2<qloc[l].size(); ++s2)
			for (size_t a2=0; a2<Daux-1; ++a2)
			{
				W[l][s1][s2].coeffRef(a1,a2) *= factor;
			}
		}
		
		for (size_t s1=0; s1<qloc[N_sites-1].size(); ++s1)
		for (size_t s2=0; s2<qloc[N_sites-1].size(); ++s2)
		{
			W[N_sites-1][s1][s2].coeffRef(Daux-1,0) *= factor;
		}
	}
	if (offset != 0.)
	{
		// apply offset to local part:
		// leftmost element on first site
		// downmost element on last site
		// down left corner element for the rest
		for (size_t l=0; l<N_sites; ++l)
		{
			size_t a1 = (l==0)? 0 : Daux-1;
			for (size_t s=0; s<qloc[l].size(); ++s)
			{
				W[l][s][s].coeffRef(a1,0) += offset/N_sites;
			}
		}
	}
	
	if (GOT_SQUARE == true and (factor!=1. or offset!=0.))
	{
		// apply to GvecSq
		for (size_t l=0; l<N_sites; ++l)
		{
			GvecSq[l] = tensor_product(Gvec[l],Gvec[l]);
		}
		
		// calc Wsq to GvecSq
		for (size_t l=0; l<N_sites; ++l)
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		{
			Wsq[l][s1][s2].resize(GvecSq[l].rows(), GvecSq[l].cols());
			
			for (size_t a1=0; a1<GvecSq[l].rows(); ++a1)
			for (size_t a2=0; a2<GvecSq[l].cols(); ++a2)
			{
				Scalar val = GvecSq[l](a1,a2)(s1,s2);
				if (val != 0.)
				{
					Wsq[l][s1][s2].insert(a1,a2) = val;
				}
			}
		}
	}
}

//template<typename Symmetry, typename Scalar>
//template<typename TimeScalar>
//Mpo<Symmetry,TimeScalar> Mpo<Symmetry,Scalar>::
//BondPropagator (TimeScalar dt, PARITY P) const
//{
//	string TevolLabel = label;
//	stringstream ss;
//	ss << ",exp(" << dt << "*H),";
//	TevolLabel += ss.str();
//	TevolLabel += (P==EVEN)? "evn" : "odd";
//	
//	Mpo<Symmetry,TimeScalar> Mout(N_sites, N_legs, locBasis(), qvacuum<Nq>(), qlabel, TevolLabel, format, true);
//	Mout.Daux = Gvec[0].auxdim();
//	
//	Mout.W.resize(N_sites);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Mout.W[l].resize(qloc[l].size());
//		for (size_t q=0; q<qloc[l].size(); ++q)
//		{
//			Mout.W[l][q].resize(qloc[l].size());
//		}
//	}
//	
//	//----------<set non-bonds to identity>----------
//	vector<size_t> IdList;
//	size_t l_frst, l_last;
//	
//	if (N_sites%2 == 0)
//	{
//		if (P == ODD)
//		{
//			IdList.push_back(0);
//			IdList.push_back(N_sites-1);
//			l_frst = 1;
//			l_last = N_sites-3;
//		}
//		else
//		{
//			l_frst = 0;
//			l_last = N_sites-2;
//		}
//	}
//	else
//	{
//		if (P == ODD)
//		{
//			IdList.push_back(0);
//			l_frst = 1;
//			l_last = N_sites-2;
//		}
//		else
//		{
//			IdList.push_back(N_sites-1);
//			l_frst = 0;
//			l_last = N_sites-3;
//		}
//	}
//	
//	for (size_t i=0; i<IdList.size(); ++i)
//	{
//		size_t l = IdList[i];
//		for (size_t s=0; s<qloc[l].size(); ++s)
//		for (size_t r=0; r<qloc[l].size(); ++r)
//		{
//			Mout.W[l][s][r].resize(1,1);
//			if (s == r)
//			{
//				Mout.W[l][s][r].coeffRef(0,0) = 1.;
//			}
//		}
//	}
//	//----------</set non-bonds to identity>----------
//	
//	for (size_t l=l_frst; l<=l_last; l+=2)
//	{
//		size_t D1 = qloc[l].size();
//		size_t D2 = qloc[l+1].size();
//		
//		size_t Grow = (l==0)? 0 : Daux-1; // last row
//		size_t Gcol = 0; // first column
//		
//		Matrix<Scalar,Dynamic,Dynamic> Hbond(D1*D2,D1*D2);
//		Hbond.setZero();
//		
//		// local part
//		// variant 1: distribute local term evenly among the sites
//		double locFactor1 = (l==0)? 1. : 0.5;
//		double locFactor2 = (l+1==N_sites-1)? 1. : 0.5;
//		// variant 2: put local term on the left site of each bond
////		double locFactor1 = 1.;
////		double locFactor2 = (l+1==N_sites-1)? 1. : 0.;
//		SparseMatrixXd IdD1 = MatrixXd::Identity(D1,D1).sparseView();
//		SparseMatrixXd IdD2 = MatrixXd::Identity(D2,D2).sparseView();
//		Hbond += locFactor1 * kroneckerProduct(Gvec[l](Grow,0), IdD2);
//		Hbond += locFactor2 * kroneckerProduct(IdD1, Gvec[l+1](Daux-1,Gcol));
//		
//		// tight-binding part
//		for (size_t a=1; a<Daux-1; ++a)
//		{
//			Hbond += kroneckerProduct(Gvec[l](Grow,a), Gvec[l+1](a,Gcol));
//		}
//		
//		SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> > Eugen(Hbond);
//		Matrix<TimeScalar,Dynamic,Dynamic> Hexp = Eugen.eigenvectors() * 
//		                                         (Eugen.eigenvalues()*dt).array().exp().matrix().asDiagonal() * 
//		                                          Eugen.eigenvectors().adjoint();
//		
//		Matrix<TimeScalar,Dynamic,Dynamic> HexpPermuted(D1*D1,D2*D2);
//		
//		for (size_t s1=0; s1<D1; ++s1)
//		for (size_t s2=0; s2<D2; ++s2)
//		for (size_t r1=0; r1<D1; ++r1)
//		for (size_t r2=0; r2<D2; ++r2)
//		{
//			size_t r = s2 + D2*s1;
//			size_t c = r2 + D2*r1;
//			size_t a = r1 + D1*s1;
//			size_t b = r2 + D2*s2;
//			
//			HexpPermuted(a,b) = Hexp(r,c);
//		}
//		
////		#ifdef DONT_USE_LAPACK_SVD
////		BDCSVD<Matrix<TimeScalar,Dynamic,Dynamic> > Jack;
////		#else
////		LapackSVD<TimeScalar> Jack;
////		#endif
////		
////		#ifdef DONT_USE_LAPACK_SVD
////		Jack.compute(HexpPermuted,ComputeThinU|ComputeThinV);
////		#else
////		Jack.compute(HexpPermuted);
////		#endif
//		// always use Eigen for higher accuracy:
//		JacobiSVD<Matrix<TimeScalar,Dynamic,Dynamic> > Jack(HexpPermuted,ComputeThinU|ComputeThinV);
//		Matrix<TimeScalar,Dynamic,Dynamic> U1 = Jack.matrixU() * Jack.singularValues().cwiseSqrt().asDiagonal();
//		Matrix<TimeScalar,Dynamic,Dynamic> U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixV().adjoint();
//		
////		// U:
////		Matrix<TimeScalar,Dynamic,Dynamic> U1 = Jack.matrixU() * Jack.singularValues().cwiseSqrt().asDiagonal();
////		// V^T:
////		#ifdef DONT_USE_LAPACK_SVD
////		Matrix<TimeScalar,Dynamic,Dynamic> U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixV().adjoint();
////		#else
////		Matrix<TimeScalar,Dynamic,Dynamic> U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixVT();
////		#endif
//		
//		for (size_t s1=0; s1<D1; ++s1)
//		for (size_t r1=0; r1<D1; ++r1)
//		{
//			Mout.W[l][s1][r1].resize(1,U1.cols());
//			
//			for (size_t k=0; k<U1.cols(); ++k)
//			{
//				if (abs(U1(r1+D1*s1,k)) > 1e-15)
//				{
//					Mout.W[l][s1][r1].coeffRef(0,k) = U1(r1+D1*s1,k);
//				}
//			}
//		}
//		
//		for (size_t s2=0; s2<D2; ++s2)
//		for (size_t r2=0; r2<D2; ++r2)
//		{
//			Mout.W[l+1][s2][r2].resize(U2.rows(),1);
//			
//			for (size_t k=0; k<U2.rows(); ++k)
//			{
//				if (abs(U2(k,r2+D2*s2)) > 1e-15)
//				{
//					Mout.W[l+1][s2][r2].coeffRef(k,0) = U2(k,r2+D2*s2);
//				}
//			}
//		}
//	}
//	
//	return Mout;
//}

template<typename Symmetry, typename Scalar>
boost::multi_array<Scalar,4> Mpo<Symmetry,Scalar>::
H2site (size_t loc1, size_t loc2, bool HALF_THE_LOCAL_TERM) const
{
	size_t D1 = qloc[loc1].size();
	size_t D2 = qloc[loc2].size();
	
	size_t Grow = Daux-1; // last row
	size_t Gcol = 0;      // first column
	
	Matrix<Scalar,Dynamic,Dynamic> Hfull(D1*D2,D1*D2);
	Hfull.setZero();
	
	// local part
	SparseMatrixXd IdD1 = MatrixXd::Identity(D1,D1).sparseView();
	SparseMatrixXd IdD2 = MatrixXd::Identity(D2,D2).sparseView();
	double factor = (HALF_THE_LOCAL_TERM==true)? 0.5:1.;
	Hfull += factor * kroneckerProduct(Gvec[loc1](Grow,0).data, IdD2);
	Hfull += factor * kroneckerProduct(IdD1, Gvec[loc2](Daux-1,Gcol).data);
	
	// tight-binding part
	for (size_t a=1; a<Daux-1; ++a)
	{
		Hfull += kroneckerProduct(Gvec[loc1](Grow,a).data, Gvec[loc2](a,Gcol).data);
	}
	
	boost::multi_array<Scalar,4> Mout(boost::extents[D1][D1][D2][D2]);
	
	for (size_t s1=0; s1<D1; ++s1)
	for (size_t s2=0; s2<D1; ++s2)
	for (size_t s3=0; s3<D2; ++s3)
	for (size_t s4=0; s4<D2; ++s4)
	{
		size_t r = s1 + D2*s3;
		size_t c = s2 + D2*s4;
		
		Mout[s1][s2][s3][s4] = Hfull(r,c);
	}
	
	return Mout;
}

//template<typename Symmetry, typename Scalar>
//void Mpo<Symmetry,Scalar>::
//SVDcompress (bool USE_SQUARE, double eps_svd, size_t N_halfsweeps)
//{
//	Mps<Sym::U0,Scalar> PsiDummy;
//	PsiDummy.setFlattenedMpo(*this,USE_SQUARE);
//	PsiDummy.eps_svd = eps_svd;
//	
//	PsiDummy.sweep(0,DMRG::BROOM::SVD);
//	for (int i=0; i<N_halfsweeps-1; ++i)
//	{
//		PsiDummy.skim(DMRG::BROOM::SVD);
//	}
//	
//	lout << "SVD-W";
//	if (USE_SQUARE) {lout << "²";}
//	lout << ": " << label << ": " << PsiDummy.info() << endl;
//	setFromFlattenedMpo(PsiDummy,USE_SQUARE);
//}

//template<typename Symmetry, typename Scalar>
//void Mpo<Symmetry,Scalar>::
//init_compression()
//{
//	A.resize(N_sites);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		size_t D = qloc[l].size();
//		A[l].resize(D);
//		for (size_t s1=0; s1<D; ++s1)
//		{
//			A[l][s1].resize(D);
//			for (size_t s2=0; s2<D; ++s2)
//			{
//				A[l][s1][s2] = MatrixType(W[l][s1][s2]);
//			}
//		}
//	}
//}

//template<typename Symmetry, typename Scalar>
//void Mpo<Symmetry,Scalar>::
//rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixQ<Nq,Scalar> *H)
//{
//	size_t Nrows = A[loc][0][0].rows();
//	size_t Ncols = A[loc][0][0].cols();
//	size_t D = qloc[loc].size();
//	
//	MatrixType Aclump(D*D*Nrows,Ncols);
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	{
//		size_t s = s2 + D*s1;
//		Aclump.block(s*Nrows,0, Nrows,Ncols) = A[loc][s1][s2];
//	}
//	
//	#ifdef DONT_USE_EIGEN_QR
//	LapackQR<Scalar> Quirinus; MatrixType Qmatrix, Rmatrix; // Lapack QR
//	#else
//	HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // Eigen QR
//	#endif
//	
//	Quirinus.compute(Aclump);
//	#ifdef DONT_USE_EIGEN_QR
//	Qmatrix = Quirinus.Qmatrix();
//	Rmatrix = Quirinus.Rmatrix();
//	#else
//	Qmatrix = Quirinus.householderQ() * MatrixType::Identity(Aclump.rows(),Aclump.cols());
//	Rmatrix = MatrixType::Identity(Aclump.cols(),Aclump.rows()) * Quirinus.matrixQR().template triangularView<Upper>();
//	#endif
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	{
//		size_t s = s2 + D*s1;
//		A[loc][s1][s2] = Qmatrix.block(s*Nrows,0, Nrows,Ncols);
//	}
//	
//	if (loc != N_sites-1)
//	{
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			A[loc+1][s1][s2] = Rmatrix * A[loc+1][s1][s2];
//		}
//	}
//	
//	this->pivot = (loc==N_sites-1)? N_sites-1 : loc+1;
//}

//template<typename Symmetry, typename Scalar>
//void Mpo<Symmetry,Scalar>::
//leftSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixQ<Nq,Scalar> *H)
//{
//	size_t Nrows = A[loc][0][0].rows();
//	size_t Ncols = A[loc][0][0].cols();
//	size_t D = qloc[loc].size();
//	
//	MatrixType Aclump(Nrows, D*D*Ncols);
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	{
//		size_t s = s2 + D*s1;
//		Aclump.block(0,s*Ncols, Nrows,Ncols) = A[loc][s1][s2];
//	}
//	
//	#ifdef DONT_USE_EIGEN_QR
//	LapackQR<Scalar> Quirinus; MatrixType Qmatrix, Rmatrix; // Lapack QR
//	#else
//	HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // Eigen QR
//	#endif
//	
//	Quirinus.compute(Aclump.adjoint());
//	#ifdef DONT_USE_EIGEN_QR
//	Qmatrix = Quirinus.Qmatrix().adjoint();
//	Rmatrix = Quirinus.Rmatrix().adjoint();
//	#else
//	Qmatrix = (Quirinus.householderQ() * MatrixType::Identity(Aclump.cols(),Aclump.rows())).adjoint();
//	Rmatrix = (MatrixType::Identity(Aclump.rows(),Aclump.cols()) * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
//	#endif
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	{
//		size_t s = s2 + D*s1;
//		A[loc][s1][s2] = Qmatrix.block(0,s*Ncols, Nrows,Ncols);
//	}
//	
//	if (loc != 0)
//	{
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			A[loc-1][s1][s2] = A[loc-1][s1][s2] * Rmatrix;
//		}
//	}
//	
//	this->pivot = (loc==0)? 0 : loc-1;
//}

//template<typename Symmetry, typename Scalar>
//string Mpo<Symmetry,Scalar>::
//test_ortho() const
//{
//	MatrixType Test;
//	string sout = "";
//	std::array<string,4> normal_token  = {"A","B","M","X"};
//	std::array<string,4> special_token = {"\e[4mA\e[0m","\e[4mB\e[0m","\e[4mM\e[0m","\e[4mX\e[0m"};
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		size_t D = qloc[l].size();
//		
//		Test.resize(A[l][0][0].cols(), A[l][0][0].cols());
//		Test.setZero();
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			Test += A[l][s1][s2].adjoint() * A[l][s1][s2];
//		}
//		Test -= Matrix<Scalar,Dynamic,Dynamic>::Identity(Test.rows(),Test.cols());
////		cout << "A l=" << l << ", Test=" << Test << endl << endl;
//		bool A_CHECK = Test.template lpNorm<Infinity>()<1e-10 ? true : false;
//		
//		Test.resize(A[l][0][0].rows(), A[l][0][0].rows());
//		Test.setZero();
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			Test += A[l][s1][s2] * A[l][s1][s2].adjoint();
//		}
//		Test -= Matrix<Scalar,Dynamic,Dynamic>::Identity(Test.rows(),Test.cols());
////		cout << "B l=" << l << ", Test=" << Test << endl << endl;
//		bool B_CHECK = Test.template lpNorm<Infinity>()<1e-10 ? true : false;
//		
//		// interpret result
//		if (A_CHECK and B_CHECK)
//		{
//			sout += TCOLOR(MAGENTA);
//			sout += (l==this->pivot) ? special_token[3] : normal_token[3]; // X
//		}
//		else if (A_CHECK)
//		{
//			sout += TCOLOR(RED);
//			sout += (l==this->pivot) ? special_token[0] : normal_token[0]; // A
//		}
//		else if (B_CHECK)
//		{
//			sout += TCOLOR(BLUE);
//			sout += (l==this->pivot) ? special_token[1] : normal_token[1]; // B
//		}
//		else
//		{
//			sout += TCOLOR(GREEN);
//			sout += (l==this->pivot) ? special_token[2] : normal_token[2]; // M
//		}
//	}
//	
//	sout += TCOLOR(BLACK);
//	return sout;
//}

//template<typename Symmetry, typename Scalar>
//void Mpo<Symmetry,Scalar>::
//flatten_to_Mps (Mps<0,Scalar> &V)
//{
////	V.format = format;
////	V.qlabel = qlabel;
//	
//	// extended basis 
//	vector<vector<qarray<0> > > qext;
//	qext.resize(N_sites);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		size_t D = qloc[l].size();
//		qext[l].resize(D*D);
//		
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			qext[l][s2+D*s1] = qvacuum<0>();
//		}
//	}
//	
//	V = Mps<0,Scalar>(N_sites, qext, qvacuum<0>());
//	
//	// outer resize
////	V.A.resize(N_sites);
////	for (size_t l=0; l<N_sites; ++l)
////	{
////		V.A[l].resize(qext[l].size());
////	}
////	V.inset.resize(N_sites);
////	V.outset.resize(N_sites);
////	V.truncWeight.resize(N_sites); V.truncWeight.setZero();
////	V.entropy.resize(N_sites); V.entropy.setConstant(numeric_limits<double>::quiet_NaN());
//	V.outerResizeNoSymm();
//	
//	// inner resize
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		size_t D = qloc[l].size();
//		
//		V.inset[l].push_back(qvacuum<0>());
//		V.outset[l].push_back(qvacuum<0>());
//		
//		for (size_t s=0; s<D*D; ++s)
//		{
//			V.A[l][s].in.push_back(qvacuum<0>());
//			V.A[l][s].out.push_back(qvacuum<0>());
//			V.A[l][s].dict.insert({qarray2<0>{qvacuum<0>(),qvacuum<0>()}, 0});
//			V.A[l][s].dim = 1;
//			V.A[l][s].block.resize(1);
//		}
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		size_t D = qloc[l].size();
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			size_t s = s2 + D*s1;
//			V.A[l][s].block[0] = MatrixType(W[l][s1][s2]);
//		}
//	}
//}

//template<typename Symmetry, typename Scalar>
//template<typename OtherSymmetry>
//void Mpo<Symmetry,Scalar>::
//set_HeisenbergPicture (const Mpo<OtherSymmetry,Scalar> Op1, const Mpo<OtherSymmetry,Scalar> Op2)
//{
//	assert(Op1.length()   == Op2.length());
//	assert(Op1.locBasis() == Op2.locBasis());
//	
//	N_sites = Op1.length();
//	Qtot = qvacuum<0>(); 
//	UNITARY = true;
//	label = Op1.label + " ⊗ " + Op2.label;
//	
//	// extended basis 
//	qloc.resize(N_sites);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		size_t D = Op1.locBasis(l).size();
//		qloc[l].resize(D*D);
//		
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			qloc[l][s2+D*s1] = qvacuum<0>();
//		}
//	}
//	
//	W.resize(N_sites);
//	for (size_t l=0; l<Op1.length(); ++l)
//	{
//		size_t D = Op1.locBasis(l).size();
//		W[l].resize(D*D);
//		
//		for (size_t s=0; s<D*D; ++s)
//		{
//			W[l][s].resize(D*D);
//		}
//	}
//	
//	for (size_t l=0; l<Op1.length(); ++l)
//	{
//		size_t D = Op1.locBasis(l).size();
//		
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		for (size_t s3=0; s3<D; ++s3)
//		for (size_t s4=0; s4<D; ++s4)
//		{
//			size_t r1 = s4 + D*s1;
//			size_t r2 = s3 + D*s2;
//			
//			W[l][r1][r2] = kroneckerProduct(Op1.W_at(l)[s1][s2], Op2.W_at(l)[s3][s4]);
//		}
//	}
//}

//template<typename Symmetry, typename Scalar>
//void Mpo<Symmetry,Scalar>::
//setFromFlattenedMpo (const Mps<Sym::U0,Scalar> &Op, bool USE_SQUARE)
//{
//	for (size_t l=0; l<this->N_sites; ++l)
//	{
//		size_t D = qloc[l].size();
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			size_t s = s2 + D*s1;
//			
//			if (USE_SQUARE)
//			{
//				Wsq[l][s1][s2] = Op.A_at(l)[s].block[0].sparseView(1e-15);
//			}
//			else
//			{
//				W[l][s1][s2] = Op.A_at(l)[s].block[0].sparseView(1e-15);
//			}
//		}
//	}
//}

template<typename Symmetry, typename Scalar>
ostream &operator<< (ostream& os, const Mpo<Symmetry,Scalar> &O)
{
	assert (O.format and "Empty pointer to format function in Mpo!");
	
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "Mpo: L=" << O.length() << ", Daux=" << O.auxdim();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<O.length(); ++l)
	{
		for (size_t s1=0; s1<O.locBasis(l).size(); ++s1)
		for (size_t s2=0; s2<O.locBasis(l).size(); ++s2)
		for (size_t k=0; k<O.opBasis(l).size(); ++k)
		{
			if (O.W_at(l)[s1][s2][k].nonZeros()>0)
			{
				std::array<typename Symmetry::qType,3> qCheck = {O.locBasis(l)[s2],O.opBasis(l)[k],O.locBasis(l)[s1]};
				if(!Symmetry::validate(qCheck)) {continue;}
				os << "[l=" << l << "]\t|" << O.format(O.locBasis(l)[s1]) << "><" << O.format(O.locBasis(l)[s2]) << "|:" << endl;
				os << Matrix<Scalar,Dynamic,Dynamic>(O.W_at(l)[s1][s2][k]) << endl;
			}
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != O.length()-1) {os << endl;}
	}
	return os;
}

template<typename Symmetry, typename Scalar1, typename Scalar2>
void compare (const Mpo<Symmetry,Scalar1> &O1, const Mpo<Symmetry,Scalar2> &O2)
{
	assert (O1.format and "Empty pointer to format function in Mpo!");
	assert (O2.format and "Empty pointer to format function in Mpo!");
	
	lout << setfill('-') << setw(30) << "-" << setfill(' ');
	lout << "Mpo: L=" << O1.length() << ", Daux=" << O1.auxdim();
	lout << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<O1.length(); ++l)
	{
		for (size_t s1=0; s1<O1.locBasis(l).size(); ++s1)
		for (size_t s2=0; s2<O1.locBasis(l).size(); ++s2)
		for (size_t k=0; k<O1.opBasis(l).size(); ++k)
		{
			lout << "[l=" << l << "]\t|" << O1.format(O1.locBasis(l)[s1]) << "><" << O1.format(O1.locBasis(l)[s2]) << "|:" << endl;
			auto M1 = Matrix<Scalar1,Dynamic,Dynamic>(O1.W_at(l)[s1][s2][k]);
			auto Mtmp = Matrix<Scalar2,Dynamic,Dynamic>(O2.W_at(l)[s1][s2][k]);
			auto M2 = Mtmp.template cast<Scalar1>();
			lout << "norm(diff)=" << (M1-M2).norm() << endl;
			if ((M1-M2).norm() > 0.)
			{
				lout << "M1=" << endl << M1 << endl << endl;
				lout << "M2=" << endl << Matrix<Scalar2,Dynamic,Dynamic>(O2.W_at(l)[s1][s2][k]) << endl << endl;
			}
		}
		lout << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != O1.length()-1) {lout << endl;}
	}
	lout << endl;
}

#endif