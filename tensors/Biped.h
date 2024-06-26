#ifndef STRAWBERRY_BIPED
#define STRAWBERRY_BIPED

/// \cond
#include <unordered_map>
#include <unordered_set>
/// \endcond

#include "termcolor.hpp" // from TOOLS

#include "macros.h" // from TOOLS
#include "MemCalc.h" // from TOOLS
#include "RandomVector.h"

//include "DmrgExternal.h"
#include "symmetry/functions.h"
#include "DmrgConglutinations.h"

double xlogx (double x)
{
	if (x > 0.0)
	{
		if (x < 1e-30)
		{
			// For very small x, x * log(x) is effectively 0
			return 0.0;
		}
		else
		{
			// Direct computation for larger x
			return x * log(x);
		}
	}
	else if (x == 0.0)
	{
		return 0.0; // mathematically, limit of x*log(x) as x approaches 0 is 0
	}
	else
	{
		return NAN; // log(x) is not defined for negative x
	}
}

template<typename Symmetry>
class Qbasis;

namespace contract {
	enum MODE {UNITY,OORR,DOT};
}

// using namespace std;

/**
 * \ingroup Tensors
 *
 * Tensor with two legs and quantum number blocks.
 * One could have used a general tensor, but the special case of two legs is hardcoded to preserve the sanity of the programmer. 
 * For the general tensor see Multipede.
 * @describe_Symmetry
 * @describe_MatrixType
 */
template<typename Symmetry, typename MatrixType_>
struct Biped
{
private:
	typedef typename Symmetry::qType qType;
	typedef Eigen::Index Index;
public:
	typedef MatrixType_ MatrixType;
private:
	typedef typename MatrixType_::Scalar Scalar;
	
public:
	
	Biped(){dim=0;}
	
	///@{
	/**
	 * Convenience access to the amount of blocks.
	 * Equal to either of the following: \p in.size(), \p out.size(), \p block.size()
	 */
	std::size_t dim;
	inline std::size_t size() const {return dim;}
	inline void plusplus() {++dim;}
	
	/**Vector of all incoming quantum numbers.*/
	std::vector<qType> in;
	
	/**Vector of all outgoing quantum numbers.*/
	std::vector<qType> out;
	
	/**
	 * Vector of quantum number blocks.
	 * The matrix \p block[q] is characterized by the incoming quantum number \p in[q] and the outgoing quantum number \p out[q]
	 */
	std::vector<MatrixType_> block;
	///@}
	
	///@{
	/**
	 *Dictionary allowing one to find the index of \p block for a given array of two quantum numbers \p qin, \p qout in \f$O(1)\f$ 
	 * operations without looping over the blocks.
	 */
	std::unordered_map<std::array<qType,2>,std::size_t> dict; // key format: {qin,qout}
	
	///@{
	/**Returns an Eigen vector of size \p dim containing all Matrix rows for every block nu.*/
	Eigen::VectorXi rows(bool FULL=false) const;
	/**Returns an Eigen vector of size \p dim containing all Matrix cols for every block nu.*/
	Eigen::VectorXi cols(bool FULL=false) const;
	/**Returns the total operator norm of the Biped. This norm is 1 for Identity Bipeds, whether the following two are not.*/
	double operatorNorm(bool COLWISE=true) const;
	/**Returns the total Frobenius norm of the Biped. This is equivalent to std::sqrt(squaredNorm().sum()).*/
	double norm() const;
	/**Returns an Eigen vector of size \p dim containing all Frobenius Matrix squared norm for every block nu.*/
	Eigen::VectorXd squaredNorm() const;
	///@}
	
	/**Prints the whole tensor, formatting the quantum numbers */
	std::string formatted () const;
	
	/**
	 * Function to print the full Biped 
	 * \param SHOW_MATRICES : if true, all the block-matrices are printed.
	 * \param precision : precision for the tensor components
	 */
	std::string print (const bool SHOW_MATRICES=false , const std::size_t precision=3 ) const;
	
	/**Prints Biped<Symmetry,MatrixType>::dict into a string.*/
	std::string print_dict() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	///@}
	
	///@{
	/**Deletes the contents of \p in, \p out, \p block, \p dict.*/
	void clear();
	
	/**Sets all matrices in Biped<Symmetry,MatrixType>::block to zero, preserving the rows and columns.*/
	void setZero();
	
	/**Sets all matrices in Biped<Symmetry,MatrixType>::block to random values, preserving the rows and columns.*/
	void setRandom();
	
	/**
	 * Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to the vacuum (both \p in & \p out).
	 * Needed in for the transfer matrix to the first site in overlap calculations.
	 */
	void setVacuum();
	
	/**
	 * Creates a single block of size 1x1 containing 1 and the corresponding quantum numbers to \p Qtot (both \p in & \p out).
	 * Needed in for the transfer matrix from the last site in overlap calculations.
	 */
	void setTarget (qType Qtot);
	
	void setTarget (vector<qType> Qmulti);
	
	void setIdentity (const Qbasis<Symmetry> &base1, const Qbasis<Symmetry> &base2, qType Q = Symmetry::qvacuum());

	void setRandom (const Qbasis<Symmetry> &base1, const Qbasis<Symmetry> &base2, qType Q = Symmetry::qvacuum());

	void setZero (const Qbasis<Symmetry> &base1, const Qbasis<Symmetry> &base2, qType Q = Symmetry::qvacuum());
	///@}
	
	///@{
	
	Biped<Symmetry,MatrixType_> cleaned() const;
	
	Biped<Symmetry,MatrixType_> sorted() const;
	
	/**
	 * Returns the adjoint tensor where all the block matrices are adjoint and the quantum number arrows are flipped: 
	 * \p in \f$\to\f$ \p out and vice versa.
	 */
	Biped<Symmetry,MatrixType_> adjoint() const;
	
	Biped<Symmetry,MatrixType_> transpose() const;
	
	Biped<Symmetry,MatrixType_> conjugate() const;
	
	/**
	 * This functions transforms all quantum numbers in Biped::in and Biped::out by \f$q \rightarrow q * N_{cells}\f$.
	 * It is used for avg(Umps V, Mpo O, Umps V) in VumpsLinearAlgebra.h when O.length() > V.length(). 
	 * In this case the quantum numbers in the Bipeds are transformed in correspondence with V.length()
	 * and this is incompatible with the quantum numbers in O.length() which are transformed in correspondence to O.length().
	 * \param number_cells : \f$N_{cells}\f$
	 */
	Biped<Symmetry,MatrixType_> adjustQN (const size_t number_cells);
	
	void cholesky (Biped<Symmetry,MatrixType> &res) const;
	
	template<typename EpsScalar>
	tuple<Biped<Symmetry,MatrixType_>,Biped<Symmetry,MatrixType_>,Biped<Symmetry,MatrixType_> >
	truncateSVD (size_t minKeep, size_t maxKeep, EpsScalar eps_svd, double &truncWeight, double &entropy, map<qarray<Symmetry::Nq>,Eigen::ArrayXd> &SVspec, bool PRESERVE_MULTIPLETS=true, bool RETURN_SPEC=true) const;
	
	template<typename EpsScalar>
	tuple<Biped<Symmetry,MatrixType_>,Biped<Symmetry,MatrixType_>,Biped<Symmetry,MatrixType_> > truncateSVD (size_t minKeep, size_t maxKeep, EpsScalar eps_svd, double &truncWeight, bool PRESERVE_MULTIPLETS=true) const
	{
		double S_dumb;
		map<qarray<Symmetry::Nq>,Eigen::ArrayXd> SVspec_dumb;
		return truncateSVD(minKeep, maxKeep, eps_svd, truncWeight, S_dumb, SVspec_dumb, PRESERVE_MULTIPLETS, false); //false: Dont return singular value spectrum
	}
	
	pair<Biped<Symmetry,MatrixType_>,Biped<Symmetry,MatrixType_> >
	QR(bool RETURN_LQ=false, bool MAKE_UNIQUE=false) const;
	
	/**
	 * Adds another tensor to the current one. 
	 * If quantum numbers match, the block is updated (block rows and columns must match), otherwise a new block is created.
	 */
	Biped<Symmetry,MatrixType_>& operator+= (const Biped<Symmetry,MatrixType_> &Arhs);
	
	void addScale (const Scalar &factor, const Biped<Symmetry,MatrixType_> &Mrhs, BLOCK_POSITION BP = SAME_PLACE);
	
	void addScale_extend (const Scalar &factor, const Biped<Symmetry,MatrixType_> &Mrhs);
	
	/**
	 * This functions perform a contraction of \p this and \p A, which is a standard Matrix multiplication in this case.
	 * \param A : other Biped which is contracted together with \p this.
	 * \param MODE
	 */
	Biped<Symmetry,MatrixType_> contract(const Biped<Symmetry,MatrixType_> &A, const contract::MODE MODE = contract::MODE::UNITY) const;
	///@}
	
	/**Takes the trace of the Biped. Only useful if this Biped is really a matrix from symmetry perspective (q_in = q_out in all blocks).*/
	Scalar trace() const;

	template<typename expScalar>
	Biped<Symmetry,MatrixType_> exp( const expScalar x ) const;
		
	///@{
	/**
	 * Adds a new block to the tensor specified by the incoming quantum number \p qin and the outgoing quantum number \p qout.
	 * \warning Does not check whether the block for these quantum numbers already exists.
	 */
	void push_back (qType qin, qType qout, const MatrixType_ &M);
	
	/**
	 * Adds a new block to the tensor specified by the 2-array of quantum numbers \p quple.
	 * The ordering convention is: \p in, \p out.
	 * \warning Does not check whether the block for these quantum numbers already exists.
	 */
	void push_back (std::array<qType,2> quple, const MatrixType_ &M);
	
	void try_push_back (std::array<qType,2> quple, const MatrixType_ &M);
	void try_push_back (qType qin, qType qout, const MatrixType_ &M);
	
	void create_block (std::array<qType,2> quple);
	void try_create_block (std::array<qType,2> quple);
	///@}
	
	template<typename OtherMatrixType>
	void outerResize (const Biped<Symmetry,OtherMatrixType> Brhs)
	{
		dict = Brhs.dict;
		in = Brhs.in;
		out = Brhs.out;
		dim = Brhs.dim;
		block.resize(dim);
	}
	
	template<typename OtherMatrixType>
	Biped<Symmetry,OtherMatrixType> cast() const
	{
		Biped<Symmetry,OtherMatrixType> Vout;
		Vout.outerResize(*this);
		
		for (size_t q=0; q<dim; ++q)
		{
			Vout.block[q] = block[q].template cast<typename OtherMatrixType::Scalar>();
		}
		
		return Vout;
	}
	
	void shift_Qin (const qarray<Symmetry::Nq> &Q)
	{
		auto in_tmp = in;
		auto out_tmp = out;
		auto block_tmp = block;
		auto dim_tmp = dim;
		
		in.clear();
		out.clear();
		block.clear();
		dict.clear();
		dim = 0;
		
		for (size_t q=0; q<dim_tmp; ++q)
		{
//			cout << "q=" << ", in=" << in[q]+Q << ", out=" << out[q]+Q << endl;
			push_back({in[q]+Q, out[q]+Q}, block_tmp[q]);
		}
		
//		for (size_t q=0; q<dim_tmp; ++q)
//		{
//			auto qnews = Symmetry::reduceSilent(in[q],Q);
//			for (const auto &qnew : qnews)
//			{
//				double factor_cgc = Symmetry::coeff_rightOrtho(qnew,in[q]);
//				auto it = dict.find(qarray2<Symmetry::Nq>{qnew,qnew});
//				if (it != dict.end())
//				{
//					cout << block[it->second].rows() << "x" << block[it->second].cols() << endl;
//					cout << block_tmp[q].rows() << "x" << block_tmp[q].cols() << endl;
//					
//					block[it->second] += factor_cgc*block_tmp[q];
//				}
//				else
//				{
//					push_back({qnew,qnew}, factor_cgc*block_tmp[q]);
//				}
//			}
//		}
	}
};

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
clear()
{
	in.clear();
	out.clear();
	block.clear();
	dict.clear();
	dim = 0;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setZero()
{
	for (std::size_t q=0; q<dim; ++q) {block[q].setZero();}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setRandom()
{
	for (std::size_t q=0; q<dim; ++q) {block[q].setRandom();}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setVacuum()
{
	MatrixType_ Mtmp(1,1); Mtmp << 1.;
	push_back(Symmetry::qvacuum(), Symmetry::qvacuum(), Mtmp);
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setIdentity (const Qbasis<Symmetry> &base1, const Qbasis<Symmetry> &base2, qType Q)
{
	for (size_t q1=0; q1<base1.Nq(); ++q1)
	for (size_t q2=0; q2<base2.Nq(); ++q2)
	{
		if (Symmetry::triangle(qarray3<Symmetry::Nq>{base1[q1],Q,base2[q2]}))
		// if (base1[q1] == base2[q2])
		{
			MatrixType Mtmp(base1.inner_dim(base1[q1]), base2.inner_dim(base2[q2]));
			Mtmp.setIdentity();
			push_back(base1[q1], base2[q2], Mtmp);
		}
	}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setRandom (const Qbasis<Symmetry> &base1, const Qbasis<Symmetry> &base2, qType Q)
{
	for (size_t q1=0; q1<base1.Nq(); ++q1)
	for (size_t q2=0; q2<base2.Nq(); ++q2)
	{
		if (Symmetry::triangle(qarray3<Symmetry::Nq>{base1[q1],Q,base2[q2]}))
		{
			MatrixType Mtmp(base1.inner_dim(base1[q1]), base2.inner_dim(base2[q2]));
			for (size_t a1=0; a1<Mtmp.rows(); ++a1)
			for (size_t a2=0; a2<Mtmp.cols(); ++a2)
			{
				Mtmp(a1,a2) = threadSafeRandUniform<typename MatrixType_::Scalar>(-1.,1.);
			}
			push_back(base1[q1], base2[q2], Mtmp);
		}
	}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setZero (const Qbasis<Symmetry> &base1, const Qbasis<Symmetry> &base2, qType Q)
{
	for (size_t q1=0; q1<base1.Nq(); ++q1)
	for (size_t q2=0; q2<base2.Nq(); ++q2)
	{
		if (Symmetry::triangle(qarray3<Symmetry::Nq>{base1[q1],Q,base2[q2]}))
		{
			MatrixType Mtmp(base1.inner_dim(base1[q1]), base2.inner_dim(base2[q2]));
			Mtmp.setZero();
			// for (size_t a1=0; a1<Mtmp.rows(); ++a1)
			// for (size_t a2=0; a2<Mtmp.cols(); ++a2)
			// {
			// 	Mtmp(a1,a2) = threadSafeRandUniform<typename MatrixType_::Scalar>(-1.,1.);
			// }
			push_back(base1[q1], base2[q2], Mtmp);
		}
	}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setTarget (qType Qtot)
{
	MatrixType_ Mtmp(1,1);
	Mtmp << 1.;
//	Mtmp << Symmetry::coeff_dot(Qtot);
	push_back(Qtot, Qtot, Mtmp);
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
setTarget (vector<qType> Qmulti)
{
	MatrixType_ Mtmp(1,1);
	Mtmp << 1.;
//	Mtmp << Symmetry::coeff_dot(Qtot);
	for (const auto &Qtot:Qmulti)
	{
		push_back(Qtot, Qtot, Mtmp);
	}
}

template<typename Symmetry, typename MatrixType_>
Eigen::VectorXi Biped<Symmetry,MatrixType_>::
rows (bool FULL) const
{
	std::unordered_set<qType> uniqueControl;
	Index count=0;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		if(auto it=uniqueControl.find(in[nu]); it == uniqueControl.end()){
			uniqueControl.insert(in[nu]); count++;
		}
	}
	Eigen::VectorXi Vout(count);
	count=0;
	uniqueControl.clear();
	for (std::size_t nu=0; nu<size(); nu++)
	{
		if(auto it=uniqueControl.find(in[nu]) == uniqueControl.end()){
			uniqueControl.insert(in[nu]);
			if(FULL) { Vout[count] = Symmetry::degeneracy(in[nu])*block[nu].rows() ; }
			else { Vout[count] = block[nu].rows(); }
			count++;
		}
	}
	return Vout;
}

template<typename Symmetry, typename MatrixType_>
Eigen::VectorXi Biped<Symmetry,MatrixType_>::
cols (bool FULL) const
{
	std::unordered_set<qType> uniqueControl;
	Index count=0;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		if(auto it=uniqueControl.find(out[nu]); it == uniqueControl.end()){
			uniqueControl.insert(out[nu]); count++;
		}
	}
	Eigen::VectorXi Vout(count);
	count=0;
	uniqueControl.clear();
	for (std::size_t nu=0; nu<size(); nu++)
	{
		if(auto it=uniqueControl.find(out[nu]) == uniqueControl.end()){
			uniqueControl.insert(out[nu]);
			if(FULL) { Vout[count] = Symmetry::degeneracy(out[nu])*block[nu].cols(); }
			else { Vout[count] = block[nu].cols(); }
			count++;
		}
	}
	
	return Vout;
}

template<typename Symmetry, typename MatrixType_>
double Biped<Symmetry,MatrixType_>::
operatorNorm (bool COLWISE) const
{
	double norm = 0.;
	for (size_t q=0; q<dim; q++)
	{
		if (COLWISE)
		{
			if (block[q].cwiseAbs().colwise().sum().maxCoeff() > norm) { norm=block[q].cwiseAbs().colwise().sum().maxCoeff(); }
		}
		else { if (block[q].cwiseAbs().rowwise().sum().maxCoeff() > norm) { norm=block[q].cwiseAbs().rowwise().sum().maxCoeff(); } }
	}
	return norm;
}

template<typename Symmetry, typename MatrixType_>
double Biped<Symmetry,MatrixType_>::
norm () const
{
	return std::sqrt(squaredNorm().sum());
	// Eigen::VectorXd Vout(size());
	// for (std::size_t nu=0; nu<size(); nu++) { Vout[nu] = block[nu].norm(); }
	// return Vout;
}

template<typename Symmetry, typename MatrixType_>
Eigen::VectorXd Biped<Symmetry,MatrixType_>::
squaredNorm () const
{
	Eigen::VectorXd Vout(size());
	for (std::size_t nu=0; nu<size(); nu++) { Vout[nu] = block[nu].squaredNorm() * Symmetry::coeff_dot(in[nu]); }
	return Vout;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
push_back (qType qin, qType qout, const MatrixType_ &M)
{
	push_back({qin,qout},M);
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
push_back (std::array<qType,2> quple, const MatrixType_ &M)
{
	assert(dict.find(quple) == dict.end() and "Block already exists in Biped!");
	in.push_back(quple[0]);
	out.push_back(quple[1]);
	block.push_back(M);
	dict.insert({quple, dim});
	++dim;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
try_push_back (qType qin, qType qout, const MatrixType_ &M)
{
	try_push_back({qin,qout},M);
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
try_push_back (std::array<qType,2> quple, const MatrixType_ &M)
{
	if (dict.find(quple) == dict.end())
	{
		in.push_back(quple[0]);
		out.push_back(quple[1]);
		block.push_back(M);
		dict.insert({quple, dim});
		++dim;
	}
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
create_block (std::array<qType,2> quple)
{
	assert(dict.find(quple) == dict.end() and "Block already exists in Biped!");
	in.push_back(quple[0]);
	out.push_back(quple[1]);
	MatrixType_ Mtmp(0,0);
	block.push_back(Mtmp);
	dict.insert({quple, dim});
	++dim;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
try_create_block (std::array<qType,2> quple)
{
	if (dict.find(quple) == dict.end())
	{
		in.push_back(quple[0]);
		out.push_back(quple[1]);
		MatrixType_ Mtmp(0,0);
		block.push_back(Mtmp);
		dict.insert({quple, dim});
		++dim;
	}
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
cleaned() const
{
	Biped<Symmetry,MatrixType_> Aout;
	for (size_t q=0; q<dim; ++q)
	{
		if (block[q].size() > 0)
		{
			Aout.try_push_back(in[q], out[q], block[q]);
		}
	}
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
sorted() const
{
	Biped<Symmetry,MatrixType_> Aout;
	set<qarray2<Symmetry::Nq> > quples;
	for (size_t q=0; q<dim; ++q)
	{
		quples.insert(qarray2<Symmetry::Nq>{in[q], out[q]});
	}
	for (const auto &quple:quples)
	{
		auto it = dict.find(quple);
		Aout.push_back(quple, block[it->second]);
	}
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
adjoint() const
{
	Biped<Symmetry,MatrixType_> Aout;
	Aout.dim = dim;
	Aout.in = out;
	Aout.out = in;
	
	// new dict with reversed keys {qin,qout}->{qout,qin}
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		auto qin  = get<0>(it->first);
		auto qout = get<1>(it->first);
		Aout.dict.insert({{qout,qin}, it->second});
	}
	
	Aout.block.resize(dim);
	for (std::size_t q=0; q<dim; ++q)
	{
		Aout.block[q] = block[q].adjoint();
	}
	
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
transpose() const
{
	Biped<Symmetry,MatrixType_> Aout;
	Aout.dim = dim;
	Aout.in = out;
	Aout.out = in;
	
	// new dict with reversed keys {qin,qout}->{qout,qin}
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		auto qin  = Symmetry::flip(get<0>(it->first));
		auto qout = Symmetry::flip(get<1>(it->first));
		Aout.dict.insert({{qout,qin}, it->second});
	}
	
	Aout.block.resize(dim);
	for (std::size_t q=0; q<dim; ++q)
	{
		Aout.block[q] = block[q].transpose();
	}
	
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
conjugate() const
{
	Biped<Symmetry,MatrixType_> Aout;
	Aout.dim = dim;
	Aout.in = in;
	Aout.out = out;
//	Aout.in = out;
//	Aout.out = in;
	
	// new dict with same keys
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		auto qin  = get<0>(it->first);
		auto qout = get<1>(it->first);
		Aout.dict.insert({{qin,qout}, it->second});
//		Aout.dict.insert({{qout,qin}, it->second});
	}
	
	Aout.block.resize(dim);
	for (std::size_t q=0; q<dim; ++q)
	{
		Aout.block[q] = block[q].conjugate();
	}
	
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
adjustQN (const size_t number_cells)
{
	Biped<Symmetry,MatrixType_> Aout;
	Aout.dim = dim;
	Aout.block = block;
	Aout.in.resize(dim);
	Aout.out.resize(dim);
	for (std::size_t q=0; q<dim; ++q)
	{
		Aout.in[q] = ::adjustQN<Symmetry>(in[q],number_cells);
		Aout.out[q] = ::adjustQN<Symmetry>(out[q],number_cells);
		Aout.dict.insert({{Aout.in[q],Aout.out[q]},q});
	}
	return Aout;
}

template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
cholesky(Biped<Symmetry,MatrixType> &res) const
{
	res = *this;
	for (size_t q=0; q<res.dim; q++)
	{
		MatrixType Mtmp = res.block[q];
		Eigen::LLT Jim(Mtmp);
		res.block[q] = Jim.matrixL();
	}
	return;
}

template<typename Symmetry, typename MatrixType_>
template<typename EpsScalar>
tuple<Biped<Symmetry,MatrixType_>, Biped<Symmetry,MatrixType_>, Biped<Symmetry,MatrixType_> > Biped<Symmetry,MatrixType_>::
truncateSVD (size_t minKeep, size_t maxKeep, EpsScalar eps_truncWeight, double &truncWeight, double &entropy, map<qarray<Symmetry::Nq>,Eigen::ArrayXd> &SVspec, bool PRESERVE_MULTIPLETS, bool RETURN_SPEC) const
{
	entropy = 0.;
	truncWeight = 0;
	Biped<Symmetry,MatrixType_> U,Vdag,Sigma;
	Biped<Symmetry,MatrixType_> trunc_U,trunc_Vdag,trunc_Sigma;
	vector<pair<typename Symmetry::qType, double> > allSV;
	
	for (size_t q=0; q<dim; ++q)
	{
		#ifdef DONT_USE_BDCSVD
//		cout << "JacobiSVD" << endl;
		JacobiSVD<MatrixType> Jack; // standard SVD
		#else
//		cout << "BDCSVD" << endl;
		BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
		#endif
		
//		cout << "begin Jack.compute" << endl;
//		cout << "block[q]=" << endl << block[q] << endl;
//		cout << "begin Jack: " << block[q].rows() << "x" << block[q].cols() << endl;
		Jack.compute(block[q], ComputeThinU|ComputeThinV);
//		cout << "end Jack.compute" << endl;
//		cout << "Jack computation done!" << endl;
		for (size_t i=0; i<Jack.singularValues().size(); i++) {allSV.push_back(make_pair(in[q],std::real(Jack.singularValues()(i))));}
		// for (const auto& s:Jack.singularValues()) {allSV.push_back(make_pair(in[q],s));}
		
		U.push_back(in[q], out[q], Jack.matrixU());
		Sigma.push_back(in[q], out[q], Jack.singularValues().asDiagonal());
		Vdag.push_back(in[q], out[q], Jack.matrixV().adjoint());
		//lout << "q=" << q << ", SV=" << Jack.singularValues().transpose() << endl;
	}
	size_t numberOfStates = allSV.size();
	std::sort(allSV.begin(),allSV.end(),[](const pair<typename Symmetry::qType, double> &sv1, const pair<typename Symmetry::qType, double> &sv2) {return sv1.second > sv2.second;});
//	for (size_t i=maxKeep; i<allSV.size(); i++)
//	{
//		truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second),2.);
//	}
	
	//Use eps_svd:
	/*
	for (int i=0; i<allSV.size(); ++i)
	{
		if (allSV[i].second < eps_svd)
		{
			truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second),2.);
		}
	}
	// std::erase_if(allSV, [eps_svd](const pair<typename Symmetry::qType, Scalar> &sv) { return (sv < eps_svd); }); c++-20 version	
	allSV.erase(std::remove_if(allSV.begin(), allSV.end(), [eps_svd](const pair<typename Symmetry::qType, double> &sv) { return (sv.second < eps_svd); }), allSV.end());
	*/
	
	// Use eps_truncWeight:
	int numberOfDiscardedStates = 0;
	for (int i=allSV.size()-1; i>=0; --i)
	{
		double truncWeightIncr = Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second),2.);
		//lout << "i=" << i << ", truncWeight=" << truncWeight << ", truncWeightIncr=" << truncWeightIncr <<  ", numberOfDiscardedStates=" << numberOfDiscardedStates << endl;
		if (truncWeight+truncWeightIncr < eps_truncWeight)
		{
			truncWeight += truncWeightIncr;
			numberOfDiscardedStates += 1;
		}
	}
	//assert(numberOfDiscardedStates < allSV.size());
	if (numberOfDiscardedStates >= allSV.size())
	{
		numberOfDiscardedStates = 0;
		truncWeight = 0.;
	}
	//lout << "all=" << allSV.size() << ", discarded=" << numberOfDiscardedStates << ", truncWeight=" << truncWeight << ", eps_truncWeight=" << eps_truncWeight << endl;
	int numberOfKeptStates = allSV.size()-numberOfDiscardedStates;
	
	if (numberOfKeptStates <= maxKeep and numberOfKeptStates >= min(minKeep,numberOfStates))
	{
		allSV.resize(numberOfKeptStates);
	}
	else if (numberOfKeptStates <= maxKeep and numberOfKeptStates <= min(minKeep,numberOfStates))
	{
		numberOfKeptStates = min(minKeep,numberOfStates);
		truncWeight = 0;
		for (int i=allSV.size()-1; i>numberOfKeptStates; --i)
		{
			truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second),2.);
		}
		allSV.resize(numberOfKeptStates);
	}
	else if (numberOfKeptStates > maxKeep)
	{
		numberOfKeptStates = min(maxKeep,numberOfStates);
		truncWeight = 0;
		for (int i=allSV.size()-1; i>numberOfKeptStates; --i)
		{
			truncWeight += Symmetry::degeneracy(allSV[i].first) * std::pow(std::abs(allSV[i].second),2.);
		}
		allSV.resize(numberOfKeptStates);
	}
	else
	{
		lout << "numberOfKeptStates=" << numberOfKeptStates << ", minKeep=" << minKeep << ", maxKeep=" << maxKeep << ", numberOfStates=" << numberOfStates << endl;
		throw;
	}
	
	// cout << "saving sv for expansion to file, #sv=" << allSV.size() << endl;
	// ofstream Filer("sv_expand");
	// size_t index=0;
	// for (const auto & [q,sv]: allSV)
	// {
	// 	Filer << index << "\t" << sv << endl;
	// 	index++;
	// }
	// Filer.close();
	
	if (PRESERVE_MULTIPLETS)
	{
		//cutLastMultiplet(allSV);
		int endOfMultiplet=-1;
		for (int i=allSV.size()-1; i>0; i--)
		{
			EpsScalar rel_diff = 2*(allSV[i-1].second-allSV[i].second)/(allSV[i-1].second+allSV[i].second);
			if (rel_diff > 0.1) {endOfMultiplet = i; break;}
		}
		if (endOfMultiplet != -1)
		{
			cout << termcolor::red << "Cutting of the last " << allSV.size()-endOfMultiplet << " singular values to preserve the multiplet" << termcolor::reset << endl;
			allSV.resize(endOfMultiplet);
		}
	}
	
	//cout << "Adding " << allSV.size() << " states from " << numberOfStates << " states" << endl;
	map<typename Symmetry::qType, vector<Scalar> > qn_orderedSV;
	for (const auto &[q,s] : allSV)
	{
		qn_orderedSV[q].push_back(s);
		//entropy += -Symmetry::degeneracy(q) * s*s * std::log(s*s);
		entropy += -Symmetry::degeneracy(q) * xlogx(s*s);
	}
	for (const auto & [q,vec_sv]: qn_orderedSV)
	{
		size_t Nret = vec_sv.size();
		// cout << "q=" << q << ", Nret=" << Nret << endl;
		auto itSigma = Sigma.dict.find({q,q});
		trunc_Sigma.push_back(q,q,Sigma.block[itSigma->second].diagonal().head(Nret).asDiagonal());
		if (RETURN_SPEC)
		{
			SVspec.insert(make_pair(q, Sigma.block[itSigma->second].diagonal().head(Nret).real()));
		}
		auto itU = U.dict.find({q,q});
		trunc_U.push_back(q, q, U.block[itU->second].leftCols(Nret));
		auto itVdag = Vdag.dict.find({q,q});
		trunc_Vdag.push_back(q, q, Vdag.block[itVdag->second].topRows(Nret));
	}
	return make_tuple(trunc_U,trunc_Sigma,trunc_Vdag);
}

template<typename Symmetry, typename MatrixType_>
pair<Biped<Symmetry,MatrixType_>,Biped<Symmetry,MatrixType_> > Biped<Symmetry,MatrixType_>::
QR(bool RETURN_LQ, bool MAKE_UNIQUE) const
{
	Biped<Symmetry,MatrixType> Q,R;
	for (size_t q=0; q<dim; ++q)
	{
		HouseholderQR<MatrixType> Quirinus;
		Quirinus.compute(block[q]);

		MatrixType Qmat, Rmat;
		if (RETURN_LQ)
		{
			Qmat = (Quirinus.householderQ() * MatrixType::Identity(block[q].rows(),block[q].cols())).adjoint();
			Rmat = (MatrixType::Identity(block[q].cols(),block[q].rows()) * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
		}
		else
		{
			Qmat = Quirinus.householderQ() * MatrixType::Identity(block[q].rows(),block[q].cols());
			Rmat = MatrixType::Identity(block[q].cols(),block[q].rows()) * Quirinus.matrixQR().template triangularView<Upper>();
		}
		if (MAKE_UNIQUE)
		{
			//make the QR decomposition unique by enforcing the diagonal of R to be positive.
			DiagonalMatrix<Scalar,Dynamic> Sign = Rmat.diagonal().cwiseSign().matrix().asDiagonal();
			Rmat = Sign*Rmat;
			Qmat = Qmat*Sign;
		}

		Q.push_back(in[q], out[q], Qmat);
		R.push_back(in[q], out[q], Rmat);
	}
	return make_pair(Q,R);
}

template<typename Symmetry, typename MatrixType_>
typename MatrixType_::Scalar Biped<Symmetry,MatrixType_>::
trace() const
{
	typename MatrixType_::Scalar res=0.;
	for (std::size_t nu=0; nu<size(); nu++)
	{
		assert(in[nu] == out[nu] and "A trace can only be taken from a matrix");
		res += block[nu].trace()*Symmetry::coeff_dot(in[nu]);
	}
	return res;
}

template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_>& Biped<Symmetry,MatrixType_>::operator+= (const Biped<Symmetry,MatrixType_> &Arhs)
{
	std::vector<std::size_t> addenda;
	
	for (std::size_t q=0; q<Arhs.dim; ++q)
	{
		std::array<qType,2> quple = {Arhs.in[q], Arhs.out[q]};
		auto it = dict.find(quple);
		
		if (it != dict.end())
		{
			block[it->second] += Arhs.block[q];
		}
		else
		{
			addenda.push_back(q);
		}
	}
	
	for (size_t q=0; q<addenda.size(); ++q)
	{
		push_back(Arhs.in[addenda[q]], Arhs.out[addenda[q]], Arhs.block[addenda[q]]);
	}
	
	return *this;
}
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
contract (const Biped<Symmetry,MatrixType_> &A, const contract::MODE MODE) const
{
	Biped<Symmetry,MatrixType_> Ares;
	Scalar factor_cgc;
	for (std::size_t q1=0; q1<this->size(); ++q1)
	for (std::size_t q2=0; q2<A.size(); ++q2)
	{
		if (this->out[q1] == A.in[q2])
		{
			if (this->in[q1] == A.out[q2])
			{
				if (this->block[q1].size() != 0 and A.block[q2].size() != 0)
				{
					factor_cgc = Scalar(1);
					if (MODE == contract::MODE::OORR)
					{
						// factor_cgc = Symmetry::coeff_rightOrtho(this->out[q1],this->in[q2]);
						factor_cgc = Symmetry::coeff_rightOrtho(this->out[q1],this->in[q1]);
					}
					else if (MODE == contract::MODE::DOT)
					{
						factor_cgc = Symmetry::coeff_dot(this->out[q1]);
					}
					if (auto it = Ares.dict.find({{this->in[q1],A.out[q2]}}); it == Ares.dict.end())
					{
						Ares.push_back(this->in[q1], A.out[q2], factor_cgc*this->block[q1]*A.block[q2]);
					}
					else
					{
						Ares.block[it->second] += factor_cgc*this->block[q1]*A.block[q2];
					}
				}
			}
		}
	}
	return Ares;
}

// Note: multiplication of Bipes which are not neccesarily singlets. So on does not have qin = qout.
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> operator* (const Biped<Symmetry,MatrixType_> &A1, const Biped<Symmetry,MatrixType_> &A2)
{
	Biped<Symmetry,MatrixType_> Ares;
	for (std::size_t q1=0; q1<A1.dim; ++q1)
	for (std::size_t q2=0; q2<A2.dim; ++q2)
	{
		if (A1.out[q1] == A2.in[q2] and A1.block[q1].size() != 0 and A2.block[q2].size() != 0)
		{
			auto it = Ares.dict.find(qarray2<Symmetry::Nq>{A1.in[q1],A2.out[q2]});
			if (it != Ares.dict.end())
			{
// 				cout << "adding" << endl;
// 				cout << Ares.block[it->second].rows() << "x" << Ares.block[it->second].cols() << endl;
// 				cout << A1.block[q1].rows() << "x" << A1.block[q1].cols() << endl;
// 				cout << A2.block[q2].rows() << "x" << A2.block[q2].cols() << endl;
// 				cout << endl;
				Ares.block[it->second] += A1.block[q1]*A2.block[q2];
			}
			else
			{
// 				cout << "pushing" << endl;
// 				cout << A1.block[q1].rows() << "x" << A1.block[q1].cols() << endl;
// 				cout << A2.block[q2].rows() << "x" << A2.block[q2].cols() << endl;
// 				cout << endl;
				Ares.push_back(A1.in[q1], A2.out[q2], A1.block[q1]*A2.block[q2]);
			}
		}
	}
	return Ares;
}

template<typename Symmetry, typename MatrixType_, typename Scalar>
Biped<Symmetry,MatrixType_> operator* (const Scalar &alpha, const Biped<Symmetry,MatrixType_> &A)
{
	Biped<Symmetry,MatrixType_> Ares = A;
	for (std::size_t q=0; q<Ares.dim; ++q)
	{
		Ares.block[q] *= alpha;
	}
	return Ares;
}

// template<typename Symmetry, typename MatrixType_>
// Biped<Symmetry,MatrixType_> operator+ (const Biped<Symmetry,MatrixType_> &A1, const Biped<Symmetry,MatrixType_> &A2)
// {
// 	Biped<Symmetry,MatrixType_> Ares = A1;
// 	Ares += A2;
// 	return Ares;
// }

template<typename Symmetry, typename MatrixType_>
template<typename expScalar>
Biped<Symmetry,MatrixType_> Biped<Symmetry,MatrixType_>::
exp( const expScalar x ) const
{
	// assert( this->legs[0].getDir() == dir::in and this->legs[1].getDir() == dir::out and "We need a regular matrix for exponentials.");
	Biped<Symmetry,MatrixType_> Mout;

	for (std::size_t nu=0; nu<size(); nu++)
	{
		MatrixType_ A;
		A = block[nu] * x;
		MatrixType_ Aexp = A.exp();
		Mout.push_back(this->in[nu], this->out[nu], Aexp);			
	}
	return Mout;
}

template<typename Symmetry, typename MatrixType_>
string Biped<Symmetry,MatrixType_>::
print_dict() const
{
	std::stringstream ss;
	for (auto it=dict.begin(); it!=dict.end(); ++it)
	{
		ss << "in:" << get<0>(it->first) << "\tout:" << get<1>(it->first) << "\t→\t" << it->second << endl;
	}
	return ss.str();
}

template<typename Symmetry, typename MatrixType_>
double Biped<Symmetry,MatrixType_>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (std::size_t q=0; q<dim; ++q)
	{
		res += calc_memory(block[q], memunit);
	}
	return res;
}

template<typename Symmetry, typename MatrixType_>
double Biped<Symmetry,MatrixType_>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	res += 2. * 2. * Symmetry::Nq * calc_memory<int>(dim, memunit); // in,out; dict.keys
	res += Symmetry::Nq * calc_memory<std::size_t>(dim, memunit); // dict.vals
	return res;
}

template<typename Symmetry, typename MatrixType_>
std::string Biped<Symmetry,MatrixType_>::
formatted () const
{
	std::stringstream ss;
	ss << "•Biped(" << dim << "):" << endl;
	for (std::size_t q=0; q<dim; ++q)
	{
		ss << "  [" << q << "]: " << Sym::format<Symmetry>(in[q]) << "→" << Sym::format<Symmetry>(out[q]);
		ss << ":" << endl;
		ss << "   " << block[q];
		if (q!=dim-1) {ss << endl;}
	}
	return ss.str();
}

template <typename Symmetry, typename MatrixType_>
std::string
Biped<Symmetry, MatrixType_>::print(const bool SHOW_MATRICES,
                                    const std::size_t precision) const {
#ifdef TOOLS_IO_TABLE
  std::stringstream out_string;

  TextTable t('-', '|', '+');
  t.add("ν");
  t.add("Q_ν");
  t.add("A_ν");
  t.endOfRow();
  for (std::size_t nu = 0; nu < size(); nu++) {
    std::stringstream ss, tt, uu;
    ss << nu;
    tt << "(" << Sym::format<Symmetry>(in[nu]) << ","
       << Sym::format<Symmetry>(out[nu]) << ")";
    uu << block[nu].rows() << "x" << block[nu].cols();
    t.add(ss.str());
    t.add(tt.str());
    t.add(uu.str());
    t.endOfRow();
  }
  t.setAlignment(0, TextTable::Alignment::RIGHT);
  out_string << t;

  if (SHOW_MATRICES) {
    out_string << termcolor::blue << termcolor::underline
               << "A-tensors:" << termcolor::reset << std::endl;
    for (std::size_t nu = 0; nu < dim; nu++) {
      out_string << termcolor::blue << "ν=" << nu << termcolor::reset
                 << std::endl
                 << std::setprecision(precision) << std::fixed
                 << termcolor::green << block[nu] << termcolor::reset
                 << std::endl;
    }
  }
  return out_string.str();
#else
  return "Can't print. Table Library is missing.";
#endif
}

/**Adds two Bipeds block- and coefficient-wise.*/
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> operator+ (const Biped<Symmetry,MatrixType_> &M1, const Biped<Symmetry,MatrixType_> &M2)
{
	if (M1.size() < M2.size()) {return M2+M1;}
	
	std::vector<std::size_t> blocks_in_2nd_biped;
	Biped<Symmetry,MatrixType_> Mout;
	
	for (std::size_t nu=0; nu<M1.size(); nu++)
	{
		auto it = M2.dict.find({{M1.in[nu], M1.out[nu]}});
		if (it != M2.dict.end())
		{
			blocks_in_2nd_biped.push_back(it->second);
		}
		
		MatrixType_ Mtmp;
		if (it != M2.dict.end() and M1.block[nu].size() > 0 and M2.block[it->second].size() > 0)
		{
			Mtmp = M1.block[nu] + M2.block[it->second]; // M1+M2
		}
		else if (it != M2.dict.end() and M1.block[nu].size() == 0)
		{
			Mtmp = M2.block[it->second]; // 0+M2
		}
		else
		{
			Mtmp = M1.block[nu]; // M1+0
		}
		
		if (Mtmp.size() != 0)
		{
			Mout.push_back({{M1.in[nu], M1.out[nu]}}, Mtmp);
		}
	}
	
	if (blocks_in_2nd_biped.size() != M2.size())
	{
		for (std::size_t nu=0; nu<M2.size(); nu++)
		{
			auto it = std::find(blocks_in_2nd_biped.begin(),blocks_in_2nd_biped.end(),nu);
			if (it == blocks_in_2nd_biped.end())
			{
				if (M2.block[nu].size() != 0)
				{
					Mout.push_back({{M2.in[nu], M2.out[nu]}}, M2.block[nu]); // 0+M2
				}
			}
		}
	}
	return Mout;
}

/**Subtracts two Bipeds block- and coefficient-wise.*/
template<typename Symmetry, typename MatrixType_>
Biped<Symmetry,MatrixType_> operator- (const Biped<Symmetry,MatrixType_> &M1, const Biped<Symmetry,MatrixType_> &M2)
{
	std::vector<std::size_t> blocks_in_2nd_biped;
	Biped<Symmetry,MatrixType_> Mout;
	
	for (std::size_t nu=0; nu<M1.size(); nu++)
	{
		auto it = M2.dict.find({{M1.in[nu], M1.out[nu]}});
		if (it != M2.dict.end())
		{
			blocks_in_2nd_biped.push_back(it->second);
		}
		
		MatrixType_ Mtmp;
		if (it != M2.dict.end() and M1.block[nu].size() != 0 and M2.block[it->second].size() != 0)
		{
//			cout << "nu=" << nu << ", M1-M2" << endl;
			Mtmp = M1.block[nu] - M2.block[it->second]; // M1-M2
		}
		else if (it != M2.dict.end() and M1.block[nu].size() == 0)
		{
//			cout << "nu=" << nu << ", 0-M2" << endl;
			Mtmp = -M2.block[it->second]; // 0-M2
		}
		else
		{
//			cout << "nu=" << nu << ", M1-0" << endl;
			Mtmp = M1.block[nu]; // M1-0
		}
		
		if (Mtmp.size() != 0)
		{
			Mout.push_back({{M1.in[nu], M1.out[nu]}}, Mtmp);
		}
	}
	
	if (blocks_in_2nd_biped.size() != M2.size())
	{
		for (std::size_t nu=0; nu<M2.size(); nu++)
		{
			auto it = std::find(blocks_in_2nd_biped.begin(),blocks_in_2nd_biped.end(),nu);
			if (it == blocks_in_2nd_biped.end())
			{
				if (M2.block[nu].size() != 0)
				{
//					cout << "nu=" << nu << ", 0-M2 and blocks_in_2nd_biped.size() != M2.size()" << endl;
					Mout.push_back({{M2.in[nu], M2.out[nu]}}, -M2.block[nu]); // 0-M2
				}
			}
		}
	}
	
//	cout << "M1:" << endl << M1 << endl;
//	cout << "M2:" << endl << M2 << endl;
//	cout << "Mout:" << endl << Mout << endl;
	
	return Mout;
}

/**Adds two Bipeds block- and coefficient-wise.*/
template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
addScale (const Scalar &factor, const Biped<Symmetry,MatrixType_> &Mrhs, BLOCK_POSITION POS)
{
	vector<size_t> matching_blocks;
	Biped<Symmetry,MatrixType_> Mout;
	
	for (size_t q=0; q<dim; ++q)
	{
		auto it = Mrhs.dict.find({{in[q], out[q]}});
		if (it != Mrhs.dict.end())
		{
			matching_blocks.push_back(it->second);
		}
		
		MatrixType_ Mtmp;
		if (it != Mrhs.dict.end())
		{
			if (block[q].size() != 0 and Mrhs.block[it->second].size() != 0)
			{
				if (POS == SAME_PLACE)
				{
					Mtmp = block[q] + factor * Mrhs.block[it->second]; // M1+factor*Mrhs
				}
				else
				{
					Mtmp = block[q];
					addPos(factor * Mrhs.block[it->second], Mtmp, POS);
				}
			}
			else if (block[q].size() == 0 and Mrhs.block[it->second].size() != 0)
			{
				Mtmp = factor * Mrhs.block[it->second]; // 0+factor*Mrhs
			}
			else if (block[q].size() != 0 and Mrhs.block[it->second].size() == 0)
			{
				Mtmp = block[q]; // M1+0
			}
			// else: block[q].size() == 0 and Mrhs.block[it->second].size() == 0 -> do nothing -> Mtmp.size() = 0
		}
		else
		{
			Mtmp = block[q]; // M1+0
		}
		
		if (Mtmp.size() != 0)
		{
			Mout.push_back({{in[q], out[q]}}, Mtmp);
		}
	}
	
	if (matching_blocks.size() != Mrhs.dim)
	{
		for (size_t q=0; q<Mrhs.size(); ++q)
		{
			auto it = find(matching_blocks.begin(), matching_blocks.end(), q);
			if (it == matching_blocks.end())
			{
				if (Mrhs.block[q].size() != 0)
				{
					Mout.push_back({{Mrhs.in[q], Mrhs.out[q]}}, factor * Mrhs.block[q]); // 0+factor*Mrhs
				}
			}
		}
	}
	
	*this = Mout;
}

/**Adds two Bipeds block- and coefficient-wise.
Extends the result if the block sizes don't match.
*/
template<typename Symmetry, typename MatrixType_>
void Biped<Symmetry,MatrixType_>::
addScale_extend (const Scalar &factor, const Biped<Symmetry,MatrixType_> &Mrhs)
{
	vector<size_t> matching_blocks;
	Biped<Symmetry,MatrixType_> Mout;
	
	for (size_t q=0; q<dim; ++q)
	{
		auto it = Mrhs.dict.find({{in[q], out[q]}});
		if (it != Mrhs.dict.end())
		{
			matching_blocks.push_back(it->second);
		}
		
		MatrixType_ Mtmp;
		if (it != Mrhs.dict.end())
		{
			if (block[q].size() != 0 and Mrhs.block[it->second].size() != 0)
			{
				if (block[q].rows() == Mrhs.block[it->second].rows() and block[q].cols()==Mrhs.block[it->second].cols())
				{
					Mtmp = block[q] + factor * Mrhs.block[it->second]; // M1+factor*Mrhs
				}
				else
				{
					int maxrows = max(block[q].rows(),Mrhs.block[it->second].rows());
					int maxcols = max(block[q].cols(),Mrhs.block[it->second].cols());
					Mtmp.resize(maxrows,maxcols);
					Mtmp.setZero();
					Mtmp.topLeftCorner(block[q].rows(),block[q].cols()) = block[q];
					Mtmp.topLeftCorner(Mrhs.block[it->second].rows(),Mrhs.block[it->second].cols()) += factor * Mrhs.block[it->second];
				}
			}
			else if (block[q].size() == 0 and Mrhs.block[it->second].size() != 0)
			{
				Mtmp = factor * Mrhs.block[it->second]; // 0+factor*Mrhs
			}
			else if (block[q].size() != 0 and Mrhs.block[it->second].size() == 0)
			{
				Mtmp = block[q]; // M1+0
			}
			// else: block[q].size() == 0 and Mrhs.block[it->second].size() == 0 -> do nothing -> Mtmp.size() = 0
		}
		else
		{
			Mtmp = block[q]; // M1+0
		}
		
		if (Mtmp.size() != 0)
		{
			Mout.push_back({{in[q], out[q]}}, Mtmp);
		}
	}
	
	if (matching_blocks.size() != Mrhs.dim)
	{
		for (size_t q=0; q<Mrhs.size(); ++q)
		{
			auto it = find(matching_blocks.begin(), matching_blocks.end(), q);
			if (it == matching_blocks.end())
			{
				if (Mrhs.block[q].size() != 0)
				{
					Mout.push_back({{Mrhs.in[q], Mrhs.out[q]}}, factor * Mrhs.block[q]); // 0+factor*Mrhs
				}
			}
		}
	}
	
	*this = Mout;
}

template<typename Symmetry, typename MatrixType_>
std::ostream& operator<< (std::ostream& os, const Biped<Symmetry,MatrixType_> &V)
{
	os << V.print(true,4);
	return os;
}

#endif
