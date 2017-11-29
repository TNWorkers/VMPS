#ifndef U1_H_
#define U1_H_

#include <array>
#include <cstddef>

#include "qarray.h"

namespace Sym{

/** \class U1
  * \ingroup Symmetry
  *
  * Class for handling a U(1) symmetry of a Hamiltonian.
  *
  * \describe_Scalar
  */
template<typename Scalar>
class U1 // : SymmetryBase<SymSUN<N,Scalar> >
{
public:
	static constexpr bool HAS_CGC = false;
	static constexpr size_t Nq=1;
	static constexpr bool NON_ABELIAN = false;
	static constexpr bool IS_TRIVIAL = false;

	// typedef std::array<int,1> qType;
	typedef qarray<Nq> qType;

	U1() {};

	
	inline static qType qvacuum() { return {0}; }

	inline static std::string name() { return "U(1)"; }

	inline static qType flip( const qType& q ) { return {-q[0]}; }
	inline static int degeneracy( const qType& q ) { return 1; }

	static std::vector<qType> reduceSilent( const qType& ql, const qType& qr);
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr);

	inline static Scalar coeff_unity();
	inline static Scalar coeff_dot(const qType& q1);
	inline static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
	inline static Scalar coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_sign(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);

	inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
									 const qType& q4, const qType& q5, const qType& q6);

	inline static Scalar coeff_9j(const qType& q1, const qType& q2, const qType& q3,
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

	inline static Scalar coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
									 const qType& q4, const qType& q5, const qType& q6,
									 const qType& q7, const qType& q8, const qType& q9,
									 const qType& q10, const qType& q11, const qType& q12);

	template<std::size_t M>
	static bool compare ( const std::array<qType,M>& q1, const std::array<qType,M>& q2 );
	
	template<std::size_t M>
	static bool validate( const std::array<qType,M>& qs );
};

template<typename Scalar> bool operator== (const typename U1<Scalar>::qType& lhs, const typename U1<Scalar>::qType& rhs) {return lhs == rhs;}
	
template<typename Scalar>
std::vector<typename U1<Scalar>::qType> U1<Scalar>::
reduceSilent( const U1<Scalar>::qType& ql, const U1<Scalar>::qType& qr )
{
	std::vector<typename U1<Scalar>::qType> vout;
	vout.push_back({ql[0]+qr[0]});
	return vout;
}

template<typename Scalar>
std::vector<typename U1<Scalar>::qType> U1<Scalar>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<typename U1<Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		vout.push_back({ql[q][0]+qr[0]});
	}
	return vout;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_dot(const qType& q1)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
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

template<typename Scalar>
Scalar U1<Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
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
	
template<typename Scalar>
Scalar U1<Scalar>::
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

template<typename Scalar>
Scalar U1<Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar U1<Scalar>::
coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6,
			const qType& q7, const qType& q8, const qType& q9,
			const qType& q10, const qType& q11, const qType& q12)
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
template<std::size_t M>
bool U1<Scalar>::
compare ( const std::array<U1<Scalar>::qType,M>& q1, const std::array<U1<Scalar>::qType,M>& q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] > q2[m][0]) { return false; }
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	return false;
}

template<typename Scalar>
template<std::size_t M>
bool U1<Scalar>::
validate ( const std::array<U1<Scalar>::qType,M>& qs )
{
	if constexpr( M == 1 or M > 3 ) { return true; }
	else if constexpr( M == 2 )
				{
					std::vector<U1<Scalar>::qType> decomp = U1<Scalar>::reduceSilent(qs[0],U1<Scalar>::flip(qs[1]));
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == U1<Scalar>::qvacuum() ) { return true; }
					}
					return false;
				}
	else if constexpr( M==3 )
					 {
						 //todo: check here triangle rule
						 std::vector<U1<Scalar>::qType> qTarget = U1<Scalar>::reduceSilent(qs[0],qs[1]);
						 if( qTarget[0] == qs[2] ) { return true; }
						 else { return false; }
						 // for (std::size_t i=2; i<M; i++)
						 // {
						 // 	 decomp = SU2xU1<Scalar>::reduceSilent(decomp,qs[i]);
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
