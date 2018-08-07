#ifndef U1_H_
#define U1_H_

#include <array>
#include <cstddef>

#include "DmrgTypedefs.h"
#include "qarray.h"

namespace Sym{

/** \class U1
  * \ingroup Symmetry
  *
  * Class for handling a U(1) symmetry of a Hamiltonian.
  *
  * \describe_Scalar
  */
template<typename Kind, typename Scalar=double>
class U1 // : SymmetryBase<SymSUN<N,Scalar> >
{
public:
	typedef Scalar Scalar_;
	
	static constexpr size_t Nq=1;
	static constexpr bool HAS_CGC = false;
	static constexpr bool NON_ABELIAN = false;
	static constexpr bool IS_TRIVIAL = false;

	typedef qarray<Nq> qType;

	U1() {};

	inline static qType qvacuum() { return {0}; }
	inline static std::string name() { return "U(1)"; }
	inline static constexpr std::array<KIND,Nq> kind() { return {Kind::name}; }

	inline static qType flip( const qType& q ) { return {-q[0]}; }
	inline static int degeneracy( const qType& q ) { return 1; }

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
	 * \note All coefficients are trivial for U(1) and could be represented by a bunch of Kronecker deltas.
	 *       Here we return simply 1, because the algorythm only allows valid combinations of quantumnumbers,
	 *       for which the Kronecker deltas are not necessary.  
	 */
	inline static Scalar coeff_unity();
	inline static Scalar coeff_dot(const qType& q1);
	inline static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
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
	 * \todo Write multiple functions, for different sizes of the array and rename them, to have a more clear interface.
	 *       Example: For 3-array: triangular(...) or something similar.
	 */
	template<std::size_t M>
	static bool validate( const std::array<qType,M>& qs );
};

// template<typename Scalar> bool operator== (const typename U1<Scalar>::qType& lhs, const typename U1<Scalar>::qType& rhs) {return lhs == rhs;}
	
template<typename Kind, typename Scalar>
std::vector<typename U1<Kind,Scalar>::qType> U1<Kind,Scalar>::
reduceSilent( const qType& ql, const qType& qr )
{
	std::vector<qType> vout;
	vout.push_back({ql[0]+qr[0]});
	return vout;
}

template<typename Kind, typename Scalar>
std::vector<typename U1<Kind,Scalar>::qType> U1<Kind,Scalar>::
reduceSilent( const qType& ql, const qType& qm, const qType& qr )
{
	std::vector<qType> vout;
	vout.push_back({ql[0]+qm[0]+qr[0]});
	return vout;
}

template<typename Kind, typename Scalar>
std::vector<typename U1<Kind,Scalar>::qType> U1<Kind,Scalar>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<typename U1<Kind,Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		vout.push_back({ql[q][0]+qr[0]});
	}
	return vout;
}

template<typename Kind, typename Scalar>
std::vector<typename U1<Kind,Scalar>::qType> U1<Kind,Scalar>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE )
{
	if (UNIQUE)
	{
		std::unordered_set<qType> uniqueControl;
		std::vector<qType> vout;
		for (std::size_t q=0; q<ql.size(); q++)
		for (std::size_t p=0; p<qr.size(); p++)
		{
			int i = ql[q][0]+qr[p][0];
			if( auto it = uniqueControl.find({i}) == uniqueControl.end() ) { uniqueControl.insert({i}); vout.push_back({i}); }
		}
		return vout;
	}
	else
	{
		std::vector<qType> vout;

		for (std::size_t q=0; q<ql.size(); q++)
		for (std::size_t p=0; p<qr.size(); p++)
		{
			vout.push_back({ql[q][0]+qr[p][0]});
		}
		return vout;
	}
}

template<typename Kind, typename Scalar>
vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > U1<Kind,Scalar>::
tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
//	std::unordered_map<qarray3<1>,std::size_t> dout;
//	size_t i=0;
//	for (std::size_t q=0; q<ql.size(); q++)
//	for (std::size_t p=0; p<qr.size(); p++)
//	{
//		dout.insert(make_pair(qarray3<1>{ql[q], qr[p], qarray<1>{ql[q][0]+qr[p][0]}}, i));
//		++i;
//	}
//	return dout;
	
	vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > out;
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		out.push_back(make_tuple(ql[q], q, qr[p], p, qarray<1>{ql[q][0]+qr[p][0]}));
	}
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_dot(const qType& q1)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_3j(const qType& q1, const qType& q2, const qType& q3,
		 int        q1_z, int        q2_z,        int q3_z)
{
	return Scalar(1.);
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
		  int        q1_z, int        q2_z,        int q3_z)
{
	return Scalar(1.);
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	// std::cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << " q4=" << q4 << " q5=" << q5 << " q6=" << q6 << std::endl;
	// assert(-q1[0] + q2[0] + q3[0] == 0 and "ERROR in U1-symmetry flow equations (6j symbol).");
	// assert(-q1[0] + q5[0] + q6[0] == 0 and "ERROR in U1-symmetry flow equations (6j symbol).");
	// assert(+q4[0] + q2[0] - q6[0] == 0 and "ERROR in U1-symmetry flow equations (6j symbol).");
	// assert(+q4[0] + q5[0] - q3[0] == 0 and "ERROR in U1-symmetry flow equations (6j symbol).");

	return Scalar(1.);
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	// std::cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << " q4=" << q4 << " q5=" << q5 << " q6=" << q6 << " q7=" << q7 << " q8=" << q8 << " q9=" << q9 << std::endl;
	// if (q1[0] + q4[0] - q7[0] != 0) {return 0.;}
	// if (q2[0] + q5[0] - q8[0] != 0) {return 0.;}
	// if (q3[0] + q6[0] - q9[0] != 0) {return 0.;}
	// if (q4[0] + q5[0] - q6[0] != 0) {return 0.;}
	// if (q7[0] + q8[0] - q9[0] != 0) {return 0.;}
	return Scalar(1.);
}
	
template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	// std::cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << " q4=" << q4 << " q5=" << q5 << " q6=" << q6 << " q7=" << q7 << " q8=" << q8 << " q9=" << q9 << std::endl;
	// assert(-q1[0] + q4[0] + q7[0] == 0 and "ERROR in U1-symmetry flow equations (9j-symbol).");
	// assert(-q2[0] + q5[0] + q8[0] == 0 and "ERROR in U1-symmetry flow equations (9j-symbol).");
	// assert(-q3[0] + q6[0] + q9[0] == 0 and "ERROR in U1-symmetry flow equations (9j-symbol).");
	// assert(+q4[0] + q5[0] - q6[0] == 0 and "ERROR in U1-symmetry flow equations (9j-symbol).");
	// assert(-q7[0] + q8[0] + q9[0] == 0 and "ERROR in U1-symmetry flow equations (9j-symbol).");
	return Scalar(1.);
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_AW(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar U1<Kind,Scalar>::
coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6,
			const qType& q7, const qType& q8, const qType& q9,
			const qType& q10, const qType& q11, const qType& q12)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
template<std::size_t M>
bool U1<Kind,Scalar>::
compare ( const std::array<U1<Kind,Scalar>::qType,M>& q1, const std::array<U1<Kind,Scalar>::qType,M>& q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] > q2[m][0]) { return false; }
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	return false;
}

template<typename Kind, typename Scalar>
template<std::size_t M>
bool U1<Kind,Scalar>::
validate ( const std::array<U1<Kind,Scalar>::qType,M>& qs )
{
	if constexpr( M == 1 or M > 3 ) { return true; }
	else if constexpr( M == 2 )
				{
					std::vector<U1<Kind,Scalar>::qType> decomp = U1<Kind,Scalar>::reduceSilent(qs[0],U1<Kind,Scalar>::flip(qs[1]));
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == U1<Kind,Scalar>::qvacuum() ) { return true; }
					}
					return false;
				}
	else if constexpr( M==3 )
					 {
						 //todo: check here triangle rule
						 std::vector<U1<Kind,Scalar>::qType> qTarget = U1<Kind,Scalar>::reduceSilent(qs[0],qs[1]);
						 if( qTarget[0] == qs[2] ) { return true; }
						 else { return false; }
						 // for (std::size_t i=2; i<M; i++)
						 // {
						 // 	 decomp = SU2xU1<Kind,Scalar>::reduceSilent(decomp,qs[i]);
						 // }						 
					 }
}

} //end namespace Sym

#ifndef STREAM_OPERATOR_ARR_1_INT
#define STREAM_OPERATOR_ARR_1_INT
std::ostream& operator<< (std::ostream& os, const typename Sym::U1<double>::qType &q)
{
	os << q[0];
	return os;
}
#endif

#endif
