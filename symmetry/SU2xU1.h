#ifndef SU2XU1_H_
#define SU2XU1_H_

#include <array>
#include <cstddef>
#include <unordered_set>

#include <gsl/gsl_sf_coupling.h>

#include <boost/rational.hpp>

#include "qarray.h"
#include "symmetry/functions.h"

namespace Sym{
	
/** \class SU2xU1
 * \ingroup Symmetry
 *
 * Class for handling a SU(2)\f$\otimes\f$U(1) symmetry of a Hamiltonian without explicitly store the Clebsch-Gordon coefficients 
 * but with computing (3n)j-symbols.
 *
 * \describe_Scalar
 * \warning Use the gsl library sf_coupling.
 */
template<typename Scalar>
class SU2xU1 // : SymmetryBase<SymSUN<N,Scalar> >
{
public:
	static constexpr bool HAS_CGC = false;
	static constexpr std::size_t Nq=2;
	static constexpr bool NON_ABELIAN = true;

	// typedef std::array<int,2> qType;
	typedef qarray<Nq> qType;

	SU2xU1() {};

	
	inline static constexpr qType qvacuum() { return {1,0}; }

	inline static std::string name() { return "SU(2)⊗U(1)"; }

	inline static qType flip( const qType& q ) { return {q[0],-q[1]}; }
	inline static int degeneracy( const qType& q ) { return q[0]; }

	///@{
	/** 
		Calculate the irreps of the tensor product of \p ql and \p qr.
	*/
	static std::vector<qType> reduceSilent(const qType& ql, const qType& qr);
	/** 
		Calculate the irreps of the tensor product of all entries of \p ql with \p qr.
		\warning : Returns not only unique irreps.
		           Not sure, if we should return only the unique values here. Probably, that should be at least added as an option.
	*/
	static std::vector<qType> reduceSilent(const std::vector<qType>& ql, const qType& qr);
	/** 
		Calculate the irreps of the tensor product of all entries of \p ql with all entries of \p qr.
		\warning : Returns only unique irreps.
		           Better: Put an option for unique or non-unique irreps in the return vector.
	*/
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr);
	///@}

	///@{
	/**
	   Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
	*/
	inline static Scalar coeff_unity();
	static Scalar coeff_dot(const qType& q1);
	static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
	static Scalar coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3);
	static Scalar coeff_sign(const qType& q1, const qType& q2, const qType& q3);
	static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);

	static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
						   const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
							  const qType& q4, const qType& q5, const qType& q6);
	
	static Scalar coeff_9j(const qType& q1, const qType& q2, const qType& q3,
						   const qType& q4, const qType& q5, const qType& q6,
						   const qType& q7, const qType& q8, const qType& q9);
	static Scalar coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
							   const qType& q4, const qType& q5, const qType& q6,
							   const qType& q7, const qType& q8, const qType& q9);
	static Scalar coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
							   const qType& q4, const qType& q5, const qType& q6,
							   const qType& q7, const qType& q8, const qType& q9);
	static Scalar coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
							 const qType& q4, const qType& q5, const qType& q6,
							 const qType& q7, const qType& q8, const qType& q9);

	static Scalar coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
							  const qType& q4, const qType& q5, const qType& q6,
							  const qType& q7, const qType& q8, const qType& q9,
							  const qType& q10, const qType& q11, const qType& q12);
	///@}

	/** 
		This function defines a strict order for arrays of quantum-numbers.
		\note The implementation is arbritary, as long as it defines a strict order.
	*/
	template<std::size_t M>
	static bool compare ( const std::array<qType,M>& q1, const std::array<qType,M>& q2 );

	/** 
		This function checks if the array \p qs contains quantum-numbers which match together, with respect to the flow equations.
		\todo Write multiple functions, for different sizes of the array and rename them, to have a more clear interface.
		      Example: For 3-array: triangular(...) or something similar.
	*/
	template<std::size_t M>
	static bool validate( const std::array<qType,M>& qs );
};

template<typename Scalar> bool operator== (const typename SU2xU1<Scalar>::qType& lhs, const typename SU2xU1<Scalar>::qType& rhs) {
	return (lhs[0] == rhs[0] and lhs[1] == rhs[1]);}

template<typename Scalar>
std::vector<typename SU2xU1<Scalar>::qType> SU2xU1<Scalar>::
reduceSilent( const SU2xU1<Scalar>::qType& ql, const SU2xU1<Scalar>::qType& qr )
{
	std::vector<typename SU2xU1<Scalar>::qType> vout;
	int smin = std::abs(ql[0]-qr[0]) +1;
	int smax = std::abs(ql[0]+qr[0]) -1;
	for ( int i=smin; i<=smax; i+=2 ) {  vout.push_back({i,ql[1]+qr[1]}); }
	return vout;
}

template<typename Scalar>
std::vector<typename SU2xU1<Scalar>::qType> SU2xU1<Scalar>::
reduceSilent( const std::vector<SU2xU1<Scalar>::qType>& ql, const SU2xU1<Scalar>::qType& qr )
{
	std::vector<typename SU2xU1<Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		int smin = std::abs(ql[q][0]-qr[0]) +1;
		int smax = std::abs(ql[q][0]+qr[0]) -1;
		for ( int i=smin; i<=smax; i+=2 ) { vout.push_back({i,ql[q][1]+qr[1]}); }
	}
	return vout;
}

template<typename Scalar>
std::vector<typename SU2xU1<Scalar>::qType> SU2xU1<Scalar>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
	std::unordered_set<qType> uniqueControl;
	std::vector<qType> vout;
	for (std::size_t q1=0; q1<ql.size(); q1++)
	for (std::size_t q2=0; q2<qr.size(); q2++)
	{
		int qmin = std::abs(ql[q1][0]-qr[q2][0]) +1;
		int qmax = std::abs(ql[q1][0]+qr[q2][0]) -1;
		for ( int i=qmin; i<=qmax; i+=2 )
		{
			if( auto it = uniqueControl.find({i}) == uniqueControl.end() ) {
				uniqueControl.insert({i,ql[q1][1]+qr[q2][1]}); vout.push_back({i,ql[q1][1]+qr[q2][1]});}
		}
	}
	return vout;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_dot(const qType& q1)
{
	Scalar out = static_cast<Scalar>(q1[0]);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = static_cast<Scalar>(q1[0]) * std::pow(static_cast<Scalar>(q2[0]),Scalar(-1.));
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]))*
	             Scalar(-1.)*phase<Scalar>((q3[0]+q1[0]-q2[0]-1) / 2);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::sqrt(static_cast<Scalar>(q2[0])) / std::sqrt(static_cast<Scalar>(q1[0]))*
		Scalar(-1.)*phase<Scalar>((q3[0]+q1[0]-q2[0]-1) /2);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]))*
		phase<Scalar>((q3[0]+q1[0]-q2[0]-1) /2);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1)*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
		phase<Scalar>((q1[0]+q5[0]+q6[0]-3) /2);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	// std::cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 <<
	//              " q4=" << q4 << " q5=" << q5 << " q6=" << q6 <<
	//              " q7=" << q7 << " q8=" << q8 << " q9=" << q9 << std::endl;
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1);
	return out;
}
	
template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]))*
		static_cast<Scalar>(q9[0]) / static_cast<Scalar>(q7[0]);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]))*
		static_cast<Scalar>(q9[0]) / static_cast<Scalar>(q7[0]);
	return out;
}

template<typename Scalar>
Scalar SU2xU1<Scalar>::
coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6,
			const qType& q7, const qType& q8, const qType& q9,
			const qType& q10, const qType& q11, const qType& q12)
{
	Scalar out = gsl_sf_coupling_9j(q4[0] -1,q5[0] -1,q6[0] -1,
									q10[0]-1,q11[0]-1,q12[0]-1,
									q7[0] -1,q8[0] -1,q9[0] -1)*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q6[0]*q12[0]))*
		gsl_sf_coupling_6j(q2[0] -1,q10[0]-1,q3[0] -1,
						   q11[0]-1,q1[0] -1,q12[0]-1)*
		std::sqrt(static_cast<Scalar>(q3[0]*q12[0]))*
		phase<Scalar>((q1[0]+q2[0]+q12[0]-3) /2);
	return out;
}

template<typename Scalar>
template<std::size_t M>
bool SU2xU1<Scalar>::
compare ( const std::array<SU2xU1<Scalar>::qType,M>& q1, const std::array<SU2xU1<Scalar>::qType,M>& q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] > q2[m][0]) { return false; }
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][1] > q2[m][1]) { return false; }
		else if (q1[m][1] < q2[m][1]) {return true; }
	}
	return false;
}

template<typename Scalar>
template<std::size_t M>
bool SU2xU1<Scalar>::
validate ( const std::array<SU2xU1<Scalar>::qType,M>& qs )
{
	if constexpr( M == 1 or M > 3 ) { return true; }
	else if constexpr( M == 2 )
				{
					std::vector<SU2xU1<Scalar>::qType> decomp = SU2xU1<Scalar>::reduceSilent(qs[0],SU2xU1<Scalar>::flip(qs[1]));
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == SU2xU1<Scalar>::qvacuum() ) { return true; }
					}
					return false;
				}
	else if constexpr( M == 3 )
					 {
						 //todo: check here triangle rule
						 std::vector<SU2xU1<Scalar>::qType> qTarget = SU2xU1<Scalar>::reduceSilent(qs[0],qs[1]);
						 bool CHECK=false;
						 for( const auto& q : qTarget )
						 {
							 if(q == qs[2]) {CHECK = true;}
						 }
						 return CHECK;
					 }
}

}//end namespace Sym

//std::ostream& operator<< (std::ostream& os, const typename Sym::SU2xU1<double>::qType &q)
//{	
//	boost::rational<int> s = boost::rational<int>(q[0]-1,2);
//	os << "[";
//	if      (s.numerator()   == 0) {os << 0;}
//	else if (s.denominator() == 1) {os << s.numerator();}
//	else {os << s.numerator() << "|" << s.denominator();}
//	// else {os << s;}
//	os << "," << q[1] << "]";
//	return os;
//}

#endif
