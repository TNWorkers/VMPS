#ifndef ZN_H_
#define ZN_H_

//include <array>
//include <cstddef>
/// \cond
#include <unordered_set>
/// \endcond

#include "DmrgTypedefs.h"
#include "DmrgExternal.h"
//include "qarray.h"

namespace Sym{

/** \class ZN
  * \ingroup Symmetry
  *
  * Class for handling a Z(N) symmetry of a Hamiltonian.
  *
  * \describe_Scalar
  */
template<typename Kind, int N, typename Scalar=double>
class ZN
{
public:
	typedef Scalar Scalar_;
	
	static constexpr int Nq=1;
	
	static constexpr bool HAS_CGC = false;
	static constexpr bool NON_ABELIAN = false;
	static constexpr bool ABELIAN = true;
	static constexpr bool IS_TRIVIAL = false;
	static constexpr bool IS_MODULAR = true;
	static constexpr int MOD_N = N;

	static constexpr bool IS_CHARGE_SU2() { return false; }
	static constexpr bool IS_SPIN_SU2() { return false; }
	
	typedef qarray<Nq> qType;
	
	ZN() {};
	
	inline static qType qvacuum() { return {0}; }
	inline static std::string name()
	{
		stringstream ss;
		ss << "Z" << N;
		return ss.str();
	}
	inline static constexpr std::array<KIND,Nq> kind() { return {Kind::name}; }
	
	inline static qType flip( const qType& q ) { return {posmod<N>(-q[0])}; }
	inline static int degeneracy( const qType& q ) { return 1; }

	inline static int spinorFactor( const qType& q ) { return +1; }
	///@{
	/**
	 * Calculate the irreps of the tensor product of \p ql and \p qr.
	 */
	static std::vector<qType> reduceSilent( const qType& ql, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of \p ql, \p qm and \p qr.
	 * \note This is independent of the order the quantumnumbers.
	 */
	static std::vector<qType> reduceSilent( const qType& ql, const qType& qm, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of all entries of \p ql and \p qr.
	 */
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of all entries of \p ql with all entries of \p qr.
	 */
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE = false);
	
	static vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr );
	///@}

	///@{
	/**
	 * Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
	 * \note All coefficients are trivial for Z(N) and could be represented by a bunch of Kronecker deltas.
	 *       Here we return simply 1, because the algorithm only allows valid combinations of quantumnumbers,
	 *       for which the Kronecker deltas are not necessary.  
	 */
	inline static Scalar coeff_unity();
	inline static Scalar coeff_dot(const qType& q1);
	inline static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
	inline static Scalar coeff_leftSweep(const qType& q1, const qType& q2);
	
	inline static Scalar coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_sign(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_sign2(const qType& q1, const qType& q2, const qType& q3) {return 1.;};
	inline static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);

	inline static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3,
								  int        q1_z, int        q2_z,        int q3_z);
	inline static Scalar coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
								   int        q1_z, int        q2_z,        int q3_z);

	inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
									 const qType& q4, const qType& q5, const qType& q6);

	inline static Scalar coeff_9j(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6,
								  const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
									  const qType& q4, const qType& q5, const qType& q6,
									  const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
									  const qType& q4, const qType& q5, const qType& q6,
									  const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
									  const qType& q4, const qType& q5, const qType& q6,
									  const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
									const qType& q4, const qType& q5, const qType& q6,
									const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_AW(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6,
								  const qType& q7, const qType& q8, const qType& q9);

	inline static Scalar coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
									 const qType& q4, const qType& q5, const qType& q6,
									 const qType& q7, const qType& q8, const qType& q9,
									 const qType& q10, const qType& q11, const qType& q12);
	///@}

	/** 
	 * This function defines a strict order for arrays of quantum-numbers.
	 * \note The implementation is arbritary, as long as it defines a strict order.
	 */
	template<std::size_t M>
	static bool compare ( const std::array<qType,M>& q1, const std::array<qType,M>& q2 );

	/** 
	 * This function checks if the array \p qs contains quantum-numbers which match together, with respect to the flow equations.
	 * \todo2 Write multiple functions, for different sizes of the array and rename them, to have a more clear interface.
	 *        Example: For 3-array: triangular(...) or something similar.
	 */
	template<std::size_t M>
	static bool validate( const std::array<qType,M>& qs );

	static bool triangle( const std::array<qType,3>& qs );
	static bool pair( const std::array<qType,2>& qs );

};

template<typename Kind, int N, typename Scalar>
std::vector<typename ZN<Kind,N,Scalar>::qType> ZN<Kind,N,Scalar>::
reduceSilent( const qType& ql, const qType& qr )
{
	std::vector<qType> vout;
	vout.push_back({posmod<N>(ql[0]+qr[0])});
	return vout;
}

template<typename Kind, int N, typename Scalar>
std::vector<typename ZN<Kind,N,Scalar>::qType> ZN<Kind,N,Scalar>::
reduceSilent( const qType& ql, const qType& qm, const qType& qr )
{
	std::vector<qType> vout;
	vout.push_back({posmod<N>(ql[0]+qm[0]+qr[0])});
	return vout;
}

template<typename Kind, int N, typename Scalar>
std::vector<typename ZN<Kind,N,Scalar>::qType> ZN<Kind,N,Scalar>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<typename ZN<Kind,N,Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		vout.push_back({posmod<N>(ql[q][0]+qr[0])});
	}
	return vout;
}

template<typename Kind, int N, typename Scalar>
std::vector<typename ZN<Kind,N,Scalar>::qType> ZN<Kind,N,Scalar>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE )
{
	if (UNIQUE)
	{
		std::unordered_set<qType> uniqueControl;
		std::vector<qType> vout;
		for (std::size_t q=0; q<ql.size(); q++)
		for (std::size_t p=0; p<qr.size(); p++)
		{
			int i = posmod<N>(ql[q][0]+qr[p][0]);
			if (auto it = uniqueControl.find({i}) == uniqueControl.end()) {uniqueControl.insert({i}); vout.push_back({i});}
		}
		return vout;
	}
	else
	{
		std::vector<qType> vout;

		for (std::size_t q=0; q<ql.size(); q++)
		for (std::size_t p=0; p<qr.size(); p++)
		{
			vout.push_back({posmod<N>(ql[q][0]+qr[p][0])});
		}
		return vout;
	}
}

template<typename Kind, int N, typename Scalar>
vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > ZN<Kind,N,Scalar>::
tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
	vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > out;
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		out.push_back(make_tuple(ql[q], q, qr[p], p, qarray<1>{posmod<N>(ql[q][0]+qr[p][0])}));
	}
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_dot(const qType& q1)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_3j(const qType& q1, const qType& q2, const qType& q3,
		 int        q1_z, int        q2_z,        int q3_z)
{
	return Scalar(1.);
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
		  int        q1_z, int        q2_z,        int q3_z)
{
	return Scalar(1.);
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	return Scalar(1.);
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	return Scalar(1.);
}
	
template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	return Scalar(1.);
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_AW(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
Scalar ZN<Kind,N,Scalar>::
coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6,
			const qType& q7, const qType& q8, const qType& q9,
			const qType& q10, const qType& q11, const qType& q12)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, int N, typename Scalar>
template<std::size_t M>
bool ZN<Kind,N,Scalar>::
compare ( const std::array<ZN<Kind,N,Scalar>::qType,M>& q1, const std::array<ZN<Kind,N,Scalar>::qType,M>& q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if      (q1[m][0] > q2[m][0]) {return false;}
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	return false;
}

template<typename Kind, int N, typename Scalar>
bool ZN<Kind,N,Scalar>::
triangle ( const std::array<ZN<Kind,N,Scalar>::qType,3>& qs )
{
	//check the triangle rule for ZN quantum numbers
	if (posmod<N>(qs[0][0]+qs[1][0]) == qs[2][0]) {return true;}
	return false;
}

template<typename Kind, int N, typename Scalar>
bool ZN<Kind,N,Scalar>::
pair ( const std::array<ZN<Kind,N,Scalar>::qType,2>& qs )
{
	//check if two quantum numbers fulfill the flow equations: simply qin = qout
	if (qs[0] == qs[1]) {return true;}
	return false;
}

template<typename Kind, int N, typename Scalar>
template<std::size_t M>
bool ZN<Kind,N,Scalar>::
validate ( const std::array<ZN<Kind,N,Scalar>::qType,M>& qs )
{
	if constexpr( M == 1 ) { return true; }
	else if constexpr( M == 2 ) { return ZN<Kind,N,Scalar>::pair(qs); }
	else if constexpr( M==3 ) { return ZN<Kind,N,Scalar>::triangle(qs); }
	else { cout << "This should not be printed out!" << endl; return true; }
}

} //end namespace Sym

#ifndef STREAM_OPERATOR_ARR_1_INT
#define STREAM_OPERATOR_ARR_1_INT
std::ostream& operator<< (std::ostream& os, const typename Sym::ZN<double>::qType &q)
{
	os << q[0];
	return os;
}
#endif

#endif
