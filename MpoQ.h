#ifndef STRAWBERRY_MPO_WITH_Q
#define STRAWBERRY_MPO_WITH_Q

#include <Eigen/SparseCore>
#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
typedef Eigen::SparseMatrix<double,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
using namespace Eigen;

#include "SuperMatrix.h"
#include "qarray.h"
#include "Biped.h"
#include "DmrgPivotStuffQ.h"
#include <unsupported/Eigen/KroneckerProduct>
#include "DmrgJanitor.h"
#ifndef DONT_USE_LAPACK_SVD
	#include "LapackWrappers.h"
#endif

/**Dummy for models without symmetries.*/
const std::array<qarray<0>,2> qloc2dummy {qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,3> qloc3dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,4> qloc4dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<string,0>    labeldummy{};

/**Namespace VMPS to distinguish names from ED equivalents.*/
namespace VMPS{};

template<size_t Nq, typename MatrixType> class MpsQ;
template<size_t Nq, typename Scalar> class MpoQ;
template<size_t Nq, typename MpHamiltonian> class DmrgSolverQ;
template<size_t Nq, typename Scalar, typename MpoScalar> class MpsQCompressor;

/**Matrix Product Operator with conserved quantum numbers (Abelian symmetries). Just adds a target quantum number and a bunch of labels on top of Mpo.
\describe_Nq
\describe_Scalar*/
template<size_t Nq, typename Scalar=double>
class MpoQ
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

template<size_t Nq_, typename MpHamiltonian> friend class DmrgSolverQ;
template<size_t Nq_, typename S1, typename S2> friend class MpsQCompressor;
template<size_t Nq_, typename S_> friend class MpoQ;
template<size_t Nq_, typename S1, typename S2> friend void HxV (const MpoQ<Nq_,S1> &H, const MpsQ<Nq_,S2> &Vin, MpsQ<Nq_,S2> &Vout, DMRG::VERBOSITY::OPTION VERBOSITY);
template<size_t Nq_, typename S1, typename S2> friend void OxV (const MpoQ<Nq_,S1> &H, const MpsQ<Nq_,S2> &Vin, MpsQ<Nq_,S2> &Vout, DMRG::BROOM::OPTION TOOL);

public:
	
	//---constructors---
	///\{
	/**Do nothing.*/
	MpoQ (){};
	
	/**Construct with all values and a homogeneous basis.*/
	MpoQ (size_t L_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input=defaultQlabel<Nq>(), string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat, 
	      bool UNITARY_input=false);
	
	/**Construct with all values and an arbitrary basis.*/
	MpoQ (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input=defaultQlabel<Nq>(), string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat, 
	      bool UNITARY_input=false);
	
	/**Construct with all values and a SuperMatrix (useful when constructing an MpoQ by another MpoQ).*/
	MpoQ (size_t L_input, const SuperMatrix<Scalar> &G_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input=defaultQlabel<Nq>(), string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat, 
	      bool UNITARY_input=false);
	
	/**Construct with all values and a vector of SuperMatrices (useful when constructing an MpoQ by another MpoQ).*/
	MpoQ (size_t L_input, const vector<SuperMatrix<Scalar> > &Gvec_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input=defaultQlabel<Nq>(), string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat, 
	      bool UNITARY_input=false);
	///\}
	
	//---set special, modify---
	///\{
	/**Set to a local operator \f$O_i\f$
	\param loc : site index
	\param Op : the local operator in question
	*/
	void setLocal (size_t loc, const MatrixType &Op);

	/**Set to a product of local operators \f$O^1_i O^2_j O^3_k \ldots\f$
	\param loc : list of locations
	\param Op : list of operators
	*/
	void setLocal (vector<size_t> loc, vector<MatrixType> Op);

	/**Set to a sum of of local operators \f$\sum_i O_i\f$
	\param Op : the local operator in question
	*/
	void setLocalSum (const MatrixType &Op);
	
	/**Set to a sum of nearest-neighbour products of local operators \f$\sum_i O^1_i O^2_{i+1}\f$
	\param Op1 : first local operator
	\param Op2 : second local operator
	*/
	void setProductSum (const MatrixType &Op1, const MatrixType &Op2);
	
	/**Makes a linear transformation of the MpoQ: \f$H' = factor*H + offset\f$.*/
	void scale (double factor=1., double offset=0.);
	
	/**Resets the MpoQ from a dummy MpsQ which has been swept.*/
	void setFromFlattenedMpoQ (const MpsQ<0,Scalar> &Op, bool USE_SQUARE=false);
	
	/**Sets the product of a left-side and right-side operator in the Heisenberg picture.*/
	template<size_t OtherNq> void setHeisenbergProduct (const MpoQ<OtherNq,Scalar> Op1, const MpoQ<OtherNq,Scalar> Op2);
	///\}
	
	//---info stuff---
	///\{
	/**\describe_info*/
	string info() const;
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	/**Calculates a measure of the sparsity of the given MpoQ.
	\param USE_SQUARE : If \p true, apply it to the stored square.
	\param PER_MATRIX : If \p true, calculate the amount of non-zeros per matrix. If \p false, calculate the fraction of non-zero elements.*/
	double sparsity (bool USE_SQUARE=false, bool PER_MATRIX=true) const;
	///\}
	
	//---formatting stuff---
	///\{
	/**Format function for the quantum numbers (e.g.\ half-integers for S=1/2).*/
	string (*format)(qarray<Nq> qnum);
	/**How this MpoQ should be called in outputs.*/
	string label;
	/**Label for quantum numbers in output (e.g.\ \f$M\f$ for magnetization; \f$N_\uparrow\f$,\f$N_\downarrow\f$ for particle numbers etc.).*/
	std::array<string,Nq> qlabel;
	///\}
	
	//---return stuff---
	///\{
	/**Returns the length of the chain.*/
	inline size_t length() const {return N_sites;}
	/**\describe_Daux*/
	inline size_t auxdim() const {return Daux;}
	/**Returns the total change in quantum numbers induced by the MpoQ.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
	/**Returns the local basis at \p loc.*/
	inline vector<qarray<Nq> > locBasis (size_t loc) const {return qloc[loc];}
	/**Returns the full local basis.*/
	inline vector<vector<qarray<Nq> > > locBasis()   const {return qloc;}
	/**Checks whether the MPO is a unitary operator.*/
	inline bool IS_UNITARY() const {return UNITARY;};
	/**Checks if the square of the MPO was calculated and stored.*/
	inline bool check_SQUARE() const {return GOT_SQUARE;}
	/**Returns the W-matrix at a given site by const reference.*/
	inline const vector<vector<SparseMatrix<Scalar> > > &W_at   (size_t loc) const {return W[loc];};
	/**Returns the W-matrix of the squared operator at a given site by const reference.*/
	inline const vector<vector<SparseMatrix<Scalar> > > &Wsq_at (size_t loc) const {return Wsq[loc];};
	
	template<typename TimeScalar> MpoQ<Nq,TimeScalar> BondPropagator (TimeScalar dt, PARITY P) const;
	///\}
	
	class qarrayIterator;
	
	void SVDcompress (bool USE_SQUARE=false, double eps_svd=1e-7, size_t N_halfsweeps=2);
	
	// compression stuff
//	string test_ortho() const;
//	void init_compression();
//	void rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixQ<Nq,Scalar> *H=NULL);
//	void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrixQ<Nq,Scalar> *H=NULL);
//	void flatten_to_MpsQ (MpsQ<0,Scalar> &V);
//	vector<vector<vector<MatrixType> > > A;
	
protected:
	
	vector<vector<qarray<Nq> > > qloc;
	
	qarray<Nq> Qtot;
	
	bool UNITARY = false;
	bool GOT_SQUARE = false;
	
	size_t N_sites;
	size_t Daux;
	
//	ArrayXd truncWeight;
	
	void construct (const SuperMatrix<Scalar> &G_input, vector<vector<vector<SparseMatrix<Scalar> > > > &Wstore, vector<SuperMatrix<Scalar> > &Gstore);
	void construct (const vector<SuperMatrix<Scalar> > &Gvec_input, vector<vector<vector<SparseMatrix<Scalar> > > > &Wstore, vector<SuperMatrix<Scalar> > &Gstore);
	
	vector<SuperMatrix<Scalar> > Gvec;
	vector<vector<vector<SparseMatrix<Scalar> > > > W;
	
	vector<SuperMatrix<Scalar> > GvecSq;
	vector<vector<vector<SparseMatrix<Scalar> > > > Wsq;
};

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum), 
      bool UNITARY_input)
:N_sites(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input), UNITARY(UNITARY_input)
{
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		qloc[l].resize(qloc_input.size());
		for (size_t s=0; s<qloc_input.size(); ++s)
		{
			qloc[l][s] = qloc_input[s];
		}
	}
	
	W.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		W[l].resize(qloc[l].size());
		for (size_t q=0; q<qloc[l].size(); ++q)
		{
			W[l][q].resize(qloc[l].size());
		}
	}
}

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum), 
      bool UNITARY_input)
:N_sites(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input), UNITARY(UNITARY_input)
{
	qloc = qloc_input;
	
	W.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		W[l].resize(qloc[l].size());
		for (size_t q=0; q<qloc[l].size(); ++q)
		{
			W[l][q].resize(qloc[l].size());
		}
	}
}

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, const SuperMatrix<Scalar> &G_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum), 
      bool UNITARY_input)
:N_sites(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input), UNITARY(UNITARY_input)
{
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		qloc[l].resize(qloc_input.size());
		for (size_t s=0; s<qloc_input.size(); ++s)
		{
			qloc[l][s] = qloc_input[s];
		}
	}
	Daux = G_input.auxdim();
	construct(G_input, W, Gvec);
}

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, const vector<SuperMatrix<Scalar> > &Gvec_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum), 
      bool UNITARY_input)
:N_sites(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input), UNITARY(UNITARY_input)
{
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		qloc[l].resize(qloc_input.size());
		for (size_t s=0; s<qloc_input.size(); ++s)
		{
			qloc[l][s] = qloc_input[s];
		}
	}
	Daux = Gvec_input[0].auxdim();
	construct(Gvec_input, W, Gvec);
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
construct (const SuperMatrix<Scalar> &G_input, vector<vector<vector<SparseMatrix<Scalar> > > >  &Wstore, vector<SuperMatrix<Scalar> > &Gstore)
{
	vector<SuperMatrix<Scalar> > Gvec(N_sites);
	size_t D = G_input(0,0).rows();
	
//	make W^[0] from last row
	Gvec[0].setRowVector(G_input.auxdim(),D);
	for (size_t i=0; i<G_input.cols(); ++i)
	{
		Gvec[0](0,i) = G_input(G_input.rows()-1,i);
	}
	
//	make W^[i], i=1,...,L-2
	for (size_t l=1; l<N_sites-1; ++l)
	{
		Gvec[l].setMatrix(G_input.auxdim(),D);
		Gvec[l] = G_input;
	}
	
//	make W^[L-1] from first column
	Gvec[N_sites-1].setColVector(G_input.auxdim(),D);
	for (size_t i=0; i<G_input.rows(); ++i)
	{
		Gvec[N_sites-1](i,0) = G_input(i,0);
	}
	
//	make Mpo
	construct(Gvec,Wstore,Gstore);
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
construct (const vector<SuperMatrix<Scalar> > &Gvec_input, vector<vector<vector<SparseMatrix<Scalar> > > >  &Wstore, vector<SuperMatrix<Scalar> > &Gstore)
{
	Wstore.resize(N_sites);
	Gstore = Gvec_input;
	
	for (size_t l=0; l<N_sites;  ++l)
	{
		Wstore[l].resize(qloc[l].size());
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		{
			Wstore[l][s1].resize(qloc[l].size());
		}
		
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		{
			Wstore[l][s1][s2].resize(Gstore[l].rows(), Gstore[l].cols());
			
			for (size_t a1=0; a1<Gstore[l].rows(); ++a1)
			for (size_t a2=0; a2<Gstore[l].cols(); ++a2)
			{
				double val = Gstore[l](a1,a2)(s1,s2);
				if (val != 0.)
				{
					Wstore[l][s1][s2].insert(a1,a2) = val;
				}
			}
		}
	}
}

template<size_t Nq, typename Scalar>
string MpoQ<Nq,Scalar>::
info() const
{
	stringstream ss;
	ss << label << ": " << "L=" << N_sites << ", ";
	
//	ss << "(";
//	for (size_t q=0; q<Nq; ++q)
//	{
//		ss << qlabel[q];
//		if (q != Nq-1) {ss << ",";}
//	}
//	ss << ")={";
//	for (size_t q=0; q<D; ++q)
//	{
//		ss << format(qloc[q]);
//		if (q != D-1) {ss << ",";}
//	}
//	ss << "}, ";
	
	ss << "Daux=" << Daux << ", ";
//	ss << "trunc_weight=" << truncWeight.sum() << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	ss << ", sparsity=" << sparsity();
	if (GOT_SQUARE) {ss << ", sparsity(sq)=" << sparsity(true);}
	return ss.str();
}

template<size_t Nq, typename Scalar>
double MpoQ<Nq,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	
	if (W.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		for (size_t s1=0; s1<qloc[l].size(); ++s1)
		for (size_t s2=0; s2<qloc[l].size(); ++s2)
		{
			res += calc_memory(W[l][s1][s2],memunit);
			if (GOT_SQUARE)
			{
				res += calc_memory(Wsq[l][s1][s2],memunit);
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

template<size_t Nq, typename Scalar>
double MpoQ<Nq,Scalar>::
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
		{
			N_nonZeros += (USE_SQUARE)? Wsq[l][s1][s2].nonZeros() : W[l][s1][s2].nonZeros();
			N_elements += (USE_SQUARE)? Wsq[l][s1][s2].rows() * Wsq[l][s1][s2].cols():
				                        W[l][s1][s2].rows()   * W[l][s1][s2].cols();
		
		}
	}
	
	return (PER_MATRIX)? N_nonZeros/N_matrices : N_nonZeros/N_elements;
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
setLocal (size_t loc, const MatrixType &Op)
{
	assert(Op.rows() == qloc[loc].size() and Op.cols() == qloc[loc].size());
	assert(loc < N_sites);
	
	Daux = 1;
	vector<SuperMatrix<Scalar> > M(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		M[l].setMatrix(Daux,qloc[l].size());
		(l==loc)? M[l](0,0)=Op : M[l](0,0).setIdentity();
	}
	
	construct(M, W, Gvec);
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
setLocal (vector<size_t> loc, vector<MatrixType> Op)
{
	assert(loc.size() >= 1 and Op.size() == loc.size());
	
	Daux = 1;
	vector<SuperMatrix<Scalar> > M(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		M[l].setMatrix(Daux,qloc[l].size());
		M[l](0,0).setIdentity();
	}
	
	for (size_t i=0; i<loc.size(); ++i)
	{
		assert(loc[i] < N_sites);
		assert(Op[i].rows() == qloc[loc[i]].size() and Op[i].cols() == qloc[loc[i]].size());
		M[loc[i]](0,0) = M[loc[i]](0,0) * Op[i];
	}
	
	construct(M, W, Gvec);
}

// O(1)+O(2)+...+O(L)
template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
setLocalSum (const MatrixType &Op)
{
	for (size_t l=0; l<N_sites; ++l)
	{
		assert(Op.rows() == qloc[l].size() and Op.cols() == qloc[l].size());
	}
	
	Daux = 2;
	vector<SuperMatrix<Scalar> > M(N_sites);
	
	M[0].setRowVector(Daux,qloc[0].size());
	M[0](0,0) = Op;
	M[0](0,1).setIdentity();
	
	for (size_t l=1; l<N_sites-1; ++l)
	{
		M[l].setMatrix(Daux,qloc[l].size());
		M[l](0,0).setIdentity();
		M[l](0,1).setZero();
		M[l](1,0) = Op;
		M[l](1,1).setIdentity();
	}
	
	M[N_sites-1].setColVector(Daux,qloc[N_sites-1].size());
	M[N_sites-1](0,0).setIdentity();
	M[N_sites-1](1,0) = Op;
	
	construct(M, W, Gvec);
}

// O1(1)*O2(2)+O1(2)*O1(3)+...+O1(L-1)*O2(L)
template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
setProductSum (const MatrixType &Op1, const MatrixType &Op2)
{
	for (size_t l=0; l<N_sites; ++l)
	{
		assert(Op1.rows() == qloc[l].size() and Op1.cols() == qloc[l].size() and 
		       Op2.rows() == qloc[l].size() and Op2.cols() == qloc[l].size());
	}
	
	Daux = 3;
	vector<SuperMatrix<Scalar> > M(N_sites);
	
	M[0].setRowVector(Daux,qloc[0].size());
	M[0](0,0).setIdentity();
	M[0](0,1) = Op1;
	M[0](0,2).setIdentity();
	
	for (size_t l=1; l<N_sites-1; ++l)
	{
		M[l].setMatrix(Daux,qloc[l].size());
		M[l].setZero();
		M[l](0,0).setIdentity();
		M[l](1,0) = Op1;
		M[l](2,1) = Op2;
		M[l](2,2).setIdentity();
	}
	
	M[N_sites-1].setColVector(Daux,qloc[N_sites-1].size());
	M[N_sites-1](0,0).setIdentity();
	M[N_sites-1](1,0) = Op2;
	M[N_sites-1](2,0).setIdentity();
	
	construct(M, W, Gvec);
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
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
				double val = GvecSq[l](a1,a2)(s1,s2);
				if (val != 0.)
				{
					Wsq[l][s1][s2].insert(a1,a2) = val;
				}
			}
		}
	}
}

template<size_t Nq, typename Scalar>
template<typename TimeScalar>
MpoQ<Nq,TimeScalar> MpoQ<Nq,Scalar>::
BondPropagator (TimeScalar dt, PARITY P) const
{
	string TevolLabel = label;
	stringstream ss;
	ss << ",exp(" << dt << "*H),";
	TevolLabel += ss.str();
	TevolLabel += (P==EVEN)? "evn" : "odd";
	
	MpoQ<Nq,TimeScalar> Mout(N_sites, locBasis(), qvacuum<Nq>(), qlabel, TevolLabel, format, true);
	Mout.Daux = Gvec[0].auxdim();
	
	Mout.W.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Mout.W[l].resize(qloc[l].size());
		for (size_t q=0; q<qloc[l].size(); ++q)
		{
			Mout.W[l][q].resize(qloc[l].size());
		}
	}
	
	//----------<set non-bonds to identity>----------
	vector<size_t> IdList;
	size_t l_frst, l_last;
	
	if (N_sites%2 == 0)
	{
		if (P == ODD)
		{
			IdList.push_back(0);
			IdList.push_back(N_sites-1);
			l_frst = 1;
			l_last = N_sites-3;
		}
		else
		{
			l_frst = 0;
			l_last = N_sites-2;
		}
	}
	else
	{
		if (P == ODD)
		{
			IdList.push_back(0);
			l_frst = 1;
			l_last = N_sites-2;
		}
		else
		{
			IdList.push_back(N_sites-1);
			l_frst = 0;
			l_last = N_sites-3;
		}
	}
	
	for (size_t i=0; i<IdList.size(); ++i)
	{
		size_t l = IdList[i];
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t r=0; r<qloc[l].size(); ++r)
		{
			Mout.W[l][s][r].resize(1,1);
			if (s == r)
			{
				Mout.W[l][s][r].coeffRef(0,0) = 1.;
			}
		}
	}
	//----------</set non-bonds to identity>----------
	
	for (size_t l=l_frst; l<=l_last; l+=2)
	{
		size_t D1 = qloc[l].size();
		size_t D2 = qloc[l+1].size();
		
		size_t Grow = (l==0)? 0 : Daux-1; // last row
		size_t Gcol = 0; // first column
		
		MatrixType Hbond(D1*D2,D1*D2);
		Hbond.setZero();
		
		// local part
		// variant 1: distribute local term evenly among the sites
		double locFactor1 = (l==0)? 1. : 0.5;
		double locFactor2 = (l+1==N_sites-1)? 1. : 0.5;
		// variant 2: put local term on the left site of each bond
//		double locFactor1 = 1.;
//		double locFactor2 = (l+1==N_sites-1)? 1. : 0.;
		Hbond += locFactor1 * kroneckerProduct(Gvec[l](Grow,0), MatrixType::Identity(D2,D2));
		Hbond += locFactor2 * kroneckerProduct(MatrixType::Identity(D1,D1), Gvec[l+1](Daux-1,Gcol));
		
		// tight-binding part
		for (size_t a=1; a<Daux-1; ++a)
		{
			Hbond += kroneckerProduct(Gvec[l](Grow,a), Gvec[l+1](a,Gcol));
		}
		
		SelfAdjointEigenSolver<MatrixType> Eugen(Hbond);
		Matrix<TimeScalar,Dynamic,Dynamic> Hexp = Eugen.eigenvectors() * 
		                                         (Eugen.eigenvalues()*dt).array().exp().matrix().asDiagonal() * 
		                                          Eugen.eigenvectors().adjoint();
		
		Matrix<TimeScalar,Dynamic,Dynamic> HexpPermuted(D1*D1,D2*D2);
		
		for (size_t s1=0; s1<D1; ++s1)
		for (size_t s2=0; s2<D2; ++s2)
		for (size_t r1=0; r1<D1; ++r1)
		for (size_t r2=0; r2<D2; ++r2)
		{
			size_t r = s2 + D2*s1;
			size_t c = r2 + D2*r1;
			size_t a = r1 + D1*s1;
			size_t b = r2 + D2*s2;
			
			HexpPermuted(a,b) = Hexp(r,c);
		}
		
//		#ifdef DONT_USE_LAPACK_SVD
//		JacobiSVD<Matrix<TimeScalar,Dynamic,Dynamic> > Jack;
//		#else
//		LapackSVD<TimeScalar> Jack;
//		#endif
//		
//		#ifdef DONT_USE_LAPACK_SVD
//		Jack.compute(HexpPermuted,ComputeThinU|ComputeThinV);
//		#else
//		Jack.compute(HexpPermuted);
//		#endif
		// always use Eigen for higher accuracy:
		JacobiSVD<Matrix<TimeScalar,Dynamic,Dynamic> > Jack(HexpPermuted,ComputeThinU|ComputeThinV);
		Matrix<TimeScalar,Dynamic,Dynamic> U1 = Jack.matrixU() * Jack.singularValues().cwiseSqrt().asDiagonal();
		Matrix<TimeScalar,Dynamic,Dynamic> U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixV().adjoint();
		
//		// U:
//		Matrix<TimeScalar,Dynamic,Dynamic> U1 = Jack.matrixU() * Jack.singularValues().cwiseSqrt().asDiagonal();
//		// V^T:
//		#ifdef DONT_USE_LAPACK_SVD
//		Matrix<TimeScalar,Dynamic,Dynamic> U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixV().adjoint();
//		#else
//		Matrix<TimeScalar,Dynamic,Dynamic> U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixVT();
//		#endif
		
		for (size_t s1=0; s1<D1; ++s1)
		for (size_t r1=0; r1<D1; ++r1)
		{
			Mout.W[l][s1][r1].resize(1,U1.cols());
			
			for (size_t k=0; k<U1.cols(); ++k)
			{
				if (abs(U1(r1+D1*s1,k)) > 1e-15)
				{
					Mout.W[l][s1][r1].coeffRef(0,k) = U1(r1+D1*s1,k);
				}
			}
		}
		
		for (size_t s2=0; s2<D2; ++s2)
		for (size_t r2=0; r2<D2; ++r2)
		{
			Mout.W[l+1][s2][r2].resize(U2.rows(),1);
			
			for (size_t k=0; k<U2.rows(); ++k)
			{
				if (abs(U2(k,r2+D2*s2)) > 1e-15)
				{
					Mout.W[l+1][s2][r2].coeffRef(k,0) = U2(k,r2+D2*s2);
				}
			}
		}
	}
	
	return Mout;
}

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
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

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
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

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
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

//template<size_t Nq, typename Scalar>
//string MpoQ<Nq,Scalar>::
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

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
//flatten_to_MpsQ (MpsQ<0,Scalar> &V)
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
//	V = MpsQ<0,Scalar>(N_sites, qext, qvacuum<0>());
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

template<size_t Nq, typename Scalar>
template<size_t OtherNq>
void MpoQ<Nq,Scalar>::
setHeisenbergProduct (const MpoQ<OtherNq,Scalar> Op1, const MpoQ<OtherNq,Scalar> Op2)
{
	assert(Op1.length()   == Op2.length());
	assert(Op1.locBasis() == Op2.locBasis());
	
	N_sites = Op1.length();
	Qtot = qvacuum<0>(); 
	UNITARY = true;
	label = Op1.label + " ⊗ " + Op2.label;
	
	// extended basis 
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		size_t D = Op1.locBasis(l).size();
		qloc[l].resize(D*D);
		
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			qloc[l][s2+D*s1] = qvacuum<0>();
		}
	}
	
	W.resize(N_sites);
	for (size_t l=0; l<Op1.length(); ++l)
	{
		size_t D = Op1.locBasis(l).size();
		W[l].resize(D*D);
		
		for (size_t s=0; s<D*D; ++s)
		{
			W[l][s].resize(D*D);
		}
	}
	
	for (size_t l=0; l<Op1.length(); ++l)
	{
		size_t D = Op1.locBasis(l).size();
		
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		for (size_t s3=0; s3<D; ++s3)
		for (size_t s4=0; s4<D; ++s4)
		{
			size_t r1 = s4 + D*s1;
			size_t r2 = s3 + D*s2;
			
			W[l][r1][r2] = kroneckerProduct(Op1.W_at(l)[s1][s2], Op2.W_at(l)[s3][s4]);
		}
	}
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
setFromFlattenedMpoQ (const MpsQ<0,Scalar> &Op, bool USE_SQUARE)
{
	for (size_t l=0; l<this->N_sites; ++l)
	{
		size_t D = qloc[l].size();
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			size_t s = s2 + D*s1;
			
			if (USE_SQUARE)
			{
				Wsq[l][s1][s2] = Op.A_at(l)[s].block[0].sparseView(1e-15);
			}
			else
			{
				W[l][s1][s2] = Op.A_at(l)[s].block[0].sparseView(1e-15);
			}
		}
	}
}

template<size_t Nq, typename Scalar>
void MpoQ<Nq,Scalar>::
SVDcompress (bool USE_SQUARE, double eps_svd, size_t N_halfsweeps)
{
	MpsQ<0,Scalar> PsiDummy;
	PsiDummy.setFlattenedMpoQ(*this,USE_SQUARE);
	PsiDummy.eps_svd = eps_svd;
	
	PsiDummy.sweep(0,DMRG::BROOM::SVD);
	for (int i=0; i<N_halfsweeps-1; ++i)
	{
		PsiDummy.skim(DMRG::BROOM::SVD);
	}
	
	lout << "SVD-W";
	if (USE_SQUARE) {lout << "²";}
	lout << ": " << label << ": " << PsiDummy.info() << endl;
	setFromFlattenedMpoQ(PsiDummy,USE_SQUARE);
}

template<size_t Nq, typename Scalar>
class MpoQ<Nq,Scalar>::qarrayIterator
{
public:
	
	/**
	\param qloc_input : vector of local bases
	\param l_frst : first site
	\param l_last : last site
	*/
	qarrayIterator (const vector<vector<qarray<Nq> > > &qloc_input, int l_frst, int l_last)
	:qloc(qloc_input[0])
	{
		size_t L = (l_last < 0 or l_frst >= qloc_input.size())? 0 : l_last-l_frst+1;
		
		// determine dq
		for (size_t q=0; q<Nq; ++q)
		{
			set<int> qset;
			for (size_t s=0; s<qloc.size(); ++s) {qset.insert(qloc[s][q]);}
			set<int> diffqset;
			for (auto it=qset.begin(); it!=qset.end(); ++it)
			{
				int prev;
				if (it==qset.begin()) {prev=*it;}
				else
				{
					diffqset.insert(*it-prev);
					prev = *it;
				}
			}
		
			assert(diffqset.size()==1 and 
			       "Unable to understand quantum number increments!");
			dq[q] = *diffqset.begin();
		}
		
		// determine qmin, qmax
		qmin = L * (*min_element(qloc.begin(),qloc.end()));
		qmax = L * (*max_element(qloc.begin(),qloc.end()));
		
		// setup NestedLoopIterator
		vector<size_t> ranges(Nq);
		for (size_t q=0; q<Nq; ++q)
		{
			ranges[q] = (qmax[q]-qmin[q])/dq[q]+1;
		}
		Nelly = NestedLoopIterator(Nq,ranges);
	};
	
	/**Returns the value of the quantum number.*/
	qarray<Nq> operator*() {return value;}
	
	qarrayIterator& operator= (const qarray<Nq> a) {value=a;}
	bool operator!= (const qarray<Nq> a) {return value!=a;}
	bool operator<= (const qarray<Nq> a) {return value<=a;}
	bool operator<  (const qarray<Nq> a) {return value< a;}
	
	qarray<Nq> begin()
	{
		Nelly = Nelly.begin();
		return qmin;
	}
	
	qarray<Nq> end()
	{
		qarray<Nq> qout = qmax;
		qout[0] += dq[0];
		return qout;
	}
	
	void operator++()
	{
		++Nelly;
		if (Nelly==Nelly.end())
		{
			value = qmax;
			value[0] += dq[0];
		}
		else
		{
			value = qmin;
			for (size_t q=0; q<Nq; ++q)
			{
				value[q] += Nelly(q)*dq[q];
			}
		}
	}
	
private:
	
	qarray<Nq> value;
	
	NestedLoopIterator Nelly;
	
	qarray<Nq> qmin;
	qarray<Nq> qmax;
	
	vector<qarray<Nq> > qloc;
	qarray<Nq> dq;
};

template<size_t Nq, typename Scalar>
ostream &operator<< (ostream& os, const MpoQ<Nq,Scalar> &O)
{
	assert (O.format and "Empty pointer to format function in MpoQ!");
	
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "MpoQ: L=" << O.length() << ", Daux=" << O.auxdim();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<O.length(); ++l)
	{
		for (size_t s1=0; s1<O.locBasis(l).size(); ++s1)
		for (size_t s2=0; s2<O.locBasis(l).size(); ++s2)
		{
			os << "[l=" << l << "]\t|" << O.format(O.locBasis(l)[s1]) << "><" << O.format(O.locBasis(l)[s2]) << "|:" << endl;
//			os << Matrix<Scalar,Dynamic,Dynamic>(O.W_at(l)[s1][s2]) << endl;
			os << Matrix<Scalar,Dynamic,Dynamic>(O.W_at(l)[s1][s2]) << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != O.length()-1) {os << endl;}
	}
	return os;
}

#endif
