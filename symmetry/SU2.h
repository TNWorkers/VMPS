#ifndef SU2_H_
#define SU2_H_

#include <array>
#include <cstddef>
#include <unordered_set>

#include <gsl/gsl_sf_coupling.h>

#include <boost/rational.hpp>

#include "qarray.h"
#include "symmetry/phase.h"
namespace Sym{

/** \class SU2
  * \ingroup Symmetry
  *
  * Class for handling a SU(2) symmetry of a Hamiltonian without explicitly store the Clebsch-Gordon coefficients but with computing (3n)j-symbols.
  *
  * \describe_Scalar
  * \warning Use the gsl library sf_coupling.
  */
template<typename Scalar>
class SU2 // : SymmetryBase<SymSUN<N,Scalar> >
{
public:
	static constexpr std::size_t Nq=1;
	static constexpr bool HAS_CGC = false;
	static constexpr bool SPECIAL = true;

	// typedef std::array<int,1> qType;
	typedef qarray<Nq> qType;

	SU2() {};

	
	inline static qType qvacuum() { return {1}; }

	inline static std::string name() { return "SU(2)"; }

	inline static qType flip( const qType& q ) { return q; }
	inline static int degeneracy( const qType& q ) { return q[0]; }
		
	static std::vector<qType> reduceSilent(const qType& ql, const qType& qr);
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr);
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr);

	/** 
		Splits the quantum number \p Q into pairs q1,q2 with q1 \otimes q2 = Q.
		q1 and q2 can take all values from the given parameters \p q1 and \q2.
		\note : Without specifying \q1 and \q2 there exist infinity solutions.
	*/
	static std::vector<std::pair<qType,qType> > split(const qType Q, const std::vector<qType>& ql, const std::vector<qType> qr);
	static std::vector<std::pair<std::size_t,std::size_t> > split(const qType Q, const std::vector<qType>& ql, const std::vector<qType> qr, bool INDEX);
		
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
	static Scalar coeff_temp(const qType& q1, const qType& q2, const qType& q3,
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
	static Scalar coeff_test(const qType& q1, const qType& q2, const qType& q3,
							 const qType& q4, const qType& q5, const qType& q6,
							 const qType& q7, const qType& q8, const qType& q9);
	static Scalar coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
							 const qType& q4, const qType& q5, const qType& q6,
							 const qType& q7, const qType& q8, const qType& q9);

	static Scalar coeff_temp2(const qType& q1, const qType& q2, const qType& q3,
							 const qType& q4, const qType& q5, const qType& q6,
							 const qType& q7, const qType& q8, const qType& q9);

	static Scalar coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
							  const qType& q4, const qType& q5, const qType& q6,
							  const qType& q7, const qType& q8, const qType& q9,
							  const qType& q10, const qType& q11, const qType& q12);

	template<std::size_t M>
	static bool compare ( const std::array<qType,M>& q1, const std::array<qType,M>& q2 );
	
	template<std::size_t M>
	static bool validate( const std::array<qType,M>& qs );
};

// template<typename Scalar> bool operator== (const typename SU2<Scalar>::qType& lhs, const typename SU2<Scalar>::qType& rhs) {return lhs == rhs;}
	
template<typename Scalar>
std::vector<typename SU2<Scalar>::qType> SU2<Scalar>::
reduceSilent( const SU2<Scalar>::qType& ql, const SU2<Scalar>::qType& qr )
{
	std::vector<typename SU2<Scalar>::qType> vout;
	int qmin = std::abs(ql[0]-qr[0]) +1;
	int qmax = std::abs(ql[0]+qr[0]) -1;
	for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
	return vout;
}

template<typename Scalar>
std::vector<typename SU2<Scalar>::qType> SU2<Scalar>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<typename SU2<Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		int qmin = std::abs(ql[q][0]-qr[0]) +1;
		int qmax = std::abs(ql[q][0]+qr[0]) -1;
		for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
	}
	return vout;
}

template<typename Scalar>
std::vector<typename SU2<Scalar>::qType> SU2<Scalar>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
	std::unordered_set<qType> uniqueControl;
	std::vector<typename SU2<Scalar>::qType> vout;
	for (std::size_t q1=0; q1<ql.size(); q1++)
	for (std::size_t q2=0; q2<qr.size(); q2++)
	{
		int qmin = std::abs(ql[q1][0]-qr[q2][0]) +1;
		int qmax = std::abs(ql[q1][0]+qr[q2][0]) -1;
		for ( int i=qmin; i<=qmax; i+=2 )
		{
			if( auto it = uniqueControl.find({i}) == uniqueControl.end() ) {uniqueControl.insert({i}); vout.push_back({i});}
		}
	}
	return vout;
}

template<typename Scalar>
std::vector<std::pair<typename SU2<Scalar>::qType,typename SU2<Scalar>::qType> > SU2<Scalar>::
split(const qType Q, const std::vector<qType>& ql, const std::vector<qType> qr)
{
	std::vector<std::pair<typename SU2<Scalar>::qType,typename SU2<Scalar>::qType> > vout;
	for (std::size_t q1=0; q1<ql.size(); q1++)
	for (std::size_t q2=0; q2<qr.size(); q2++)
	{
		auto Qs = SU2<Scalar>::reduceSilent(ql[q1],qr[q2]);
		if(auto it = std::find(Qs.begin(),Qs.end(),Q) != Qs.end()) {vout.push_back({ql[q1],qr[q2]});}
	}
	return vout;
}

template<typename Scalar>
std::vector<std::pair<std::size_t,std::size_t> > SU2<Scalar>::
split(const qType Q, const std::vector<qType>& ql, const std::vector<qType> qr, bool INDEX)
{
	std::vector<std::pair<std::size_t,std::size_t> > vout;
	for (std::size_t q1=0; q1<ql.size(); q1++)
	for (std::size_t q2=0; q2<qr.size(); q2++)
	{
		auto Qs = SU2<Scalar>::reduceSilent(ql[q1],qr[q2]);
		if(auto it = std::find(Qs.begin(),Qs.end(),Q) != Qs.end()) {vout.push_back({q1,q2});}
	}
	return vout;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_dot(const qType& q1)
{
	Scalar out = static_cast<Scalar>(q1[0]);
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = static_cast<Scalar>(q1[0]) * std::pow(static_cast<Scalar>(q2[0]),Scalar(-1.));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::pow(static_cast<Scalar>(q1[0]),Scalar(0.5)) * std::pow(static_cast<Scalar>(q2[0]),Scalar(-0.5))*
		Scalar(-1.)*std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::pow(static_cast<Scalar>(q2[0]),Scalar(0.5)) * std::pow(static_cast<Scalar>(q1[0]),Scalar(-0.5))*
		Scalar(-1.)*std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1)) *
		std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1);
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1)*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
		std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q5[0]+q6[0]-3));
	// Scalar out = gsl_sf_coupling_6j(q2[0]-1,q4[0]-1,q3[0]-1,
	// 								q5[0]-1,q1[0]-1,q6[0]-1)*
	// 	std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
	// 	std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q2[0]+q6[0]-3));

	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_temp(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6)
{
	// Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
	// 								q4[0]-1,q5[0]-1,q6[0]-1)*
	// 	std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
	// 	std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q5[0]+q6[0]-3));
	Scalar out = gsl_sf_coupling_6j(q2[0]-1,q4[0]-1,q3[0]-1,
									q5[0]-1,q1[0]-1,q6[0]-1)*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
		phase((q1[0]+q2[0]+q6[0]-3)/2);
		// std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q2[0]+q6[0]-3));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	// std::cout << "q1=" << q1 << " q2=" << q2 << " q3=" << q3 << " q4=" << q4 << " q5=" << q5 << " q6=" << q6 << " q7=" << q7 << " q8=" << q8 << " q9=" << q9 << std::endl;
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1);
	return out;
}
	
template<typename Scalar>
Scalar SU2<Scalar>::
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
Scalar SU2<Scalar>::
coeff_test(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::pow(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]),Scalar(0.5))*
		static_cast<Scalar>(q9[0])*std::pow(static_cast<Scalar>(q7[0]),Scalar(-1.));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::pow(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]),Scalar(0.5))*
		static_cast<Scalar>(q9[0])*std::pow(static_cast<Scalar>(q7[0]),Scalar(-1.));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::pow(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]),Scalar(0.5))*
		static_cast<Scalar>(q9[0])*std::pow(static_cast<Scalar>(q7[0]),Scalar(-1.));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
coeff_temp2(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6,
			const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1)*
		std::sqrt(static_cast<Scalar>(q4[0]*q5[0]*q3[0]*q9[0]));
	return out;
}

template<typename Scalar>
Scalar SU2<Scalar>::
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
		std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q2[0]+q12[0]-3));
	return out;
}

template<typename Scalar>
template<std::size_t M>
bool SU2<Scalar>::
compare ( const std::array<SU2<Scalar>::qType,M>& q1, const std::array<SU2<Scalar>::qType,M>& q2 )
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
bool SU2<Scalar>::
validate ( const std::array<SU2<Scalar>::qType,M>& qs )
{
	if constexpr( M > 1 )
				{
					std::vector<SU2<Scalar>::qType> decomp = SU2<Scalar>::reduceSilent(qs[0],qs[1]);
					for (std::size_t i=2; i<M; i++)
					{
						decomp = SU2<Scalar>::reduceSilent(decomp,qs[i]);
					}
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == SU2<Scalar>::qvacuum() ) { return true; }
					}
					return false;
				}
	else { return true; }
}

} //end namespace Sym

// #ifndef STREAM_OPERATOR_ARR_1_INT
// #define STREAM_OPERATOR_ARR_1_INT
// std::ostream& operator<< (std::ostream& os, const typename Sym::SU2<double>::qType &q)
// {
// 	boost::rational<int> s = boost::rational<int>(q[0]-1,2);
// 	if      (s.numerator()   == 0) {os << " " << 0 << " ";}
// 	else if (s.denominator() == 1) {os << " " << s.numerator() << " ";}
// 	else {os << s;}
// 	return os;
// }
// #endif

#endif
