#ifndef SU2_H_
#define SU2_H_

#include <array>
#include <cstddef>
#include <unordered_set>

#include <gsl/gsl_sf_coupling.h>

#include <boost/rational.hpp>

#include "DmrgTypedefs.h"
#include "DmrgExternal.h"
#include "qarray.h"
#include "symmetry/functions.h"

#include <unordered_map>
#include <functional>

namespace std
{
	template<>
	struct hash<std::array<int,9> >
	{
		inline size_t operator()(const std::array<int,9> &a) const
		{
			size_t seed = 0;
			boost::hash_combine(seed, a[0]);
			boost::hash_combine(seed, a[1]);
			boost::hash_combine(seed, a[2]);
			boost::hash_combine(seed, a[3]);
			boost::hash_combine(seed, a[4]);
			boost::hash_combine(seed, a[5]);
			boost::hash_combine(seed, a[6]);
			boost::hash_combine(seed, a[7]);
			boost::hash_combine(seed, a[8]);
			return seed;
		}
	};
	
	template<>
	struct hash<std::array<int,6> >
	{
		inline size_t operator()(const std::array<int,6> &a) const
		{
			size_t seed = 0;
			boost::hash_combine(seed, a[0]);
			boost::hash_combine(seed, a[1]);
			boost::hash_combine(seed, a[2]);
			boost::hash_combine(seed, a[3]);
			boost::hash_combine(seed, a[4]);
			boost::hash_combine(seed, a[5]);
			return seed;
		}
	};
}

std::unordered_map<std::array<int,9>,double > Table9j;
std::unordered_map<std::array<int,6>,double > Table6j;

double coupling_9j (const int &q1, const int &q2, const int &q3, 
                    const int &q4, const int &q5, const int &q6, 
                    const int &q7, const int &q8, const int &q9)
{
	auto it = Table9j.find(std::array<int,9>{q1,q2,q3,q4,q5,q6,q7,q8,q9});
	
	if (it != Table9j.end())
	{
		return Table9j[std::array<int,9>{q1,q2,q3,q4,q5,q6,q7,q8,q9}];
	}
	else
	{
		double out = gsl_sf_coupling_9j(q1-1,q2-1,q3-1,
		                                q4-1,q5-1,q6-1,
		                                q7-1,q8-1,q9-1);
		Table9j[std::array<int,9>{q1,q2,q3,q4,q5,q6,q7,q8,q9}] = out;
		return out;
	}
}

double coupling_6j (const int &q1, const int &q2, const int &q3, 
                    const int &q4, const int &q5, const int &q6)
{
	auto it = Table6j.find(std::array<int,6>{q1,q2,q3,q4,q5,q6});
	
	if (it != Table6j.end())
	{
		return Table6j[std::array<int,6>{q1,q2,q3,q4,q5,q6}];
	}
	else
	{
		double out = gsl_sf_coupling_6j(q1-1,q2-1,q3-1,
		                                q4-1,q5-1,q6-1);
		Table6j[std::array<int,6>{q1,q2,q3,q4,q5,q6}] = out;
		return out;
	}
}

namespace Sym{

/** 
 * \class SU2
 * \ingroup Symmetry
 *
 * Class for handling a SU(2) symmetry of a Hamiltonian without explicitly store the Clebsch-Gordon coefficients but with computing (3n)j-symbols.
 *
 * \describe_Scalar
 * \warning Use the gsl library sf_coupling.
 */
template<typename Kind, typename Scalar=double>
class SU2 // : SymmetryBase<SymSUN<N,Scalar> >
{
public:
	typedef Scalar Scalar_;

	static constexpr std::size_t Nq=1;
	static constexpr bool HAS_CGC = false;
	static constexpr bool NON_ABELIAN = true;
	static constexpr bool IS_TRIVIAL = false;

	// typedef std::array<int,1> qType;
	typedef qarray<Nq> qType;
	
	SU2() {};

	inline static std::string name() { return "SU(2)"; }
	inline static constexpr std::array<KIND,Nq> kind() { return {Kind::name}; }
	
	inline static qType qvacuum() { return {1}; }
	inline static qType flip( const qType& q ) { return q; }
	inline static int degeneracy( const qType& q ) { return q[0]; }

	///@{
	/** 
	 * Calculate the irreps of the tensor product of \p ql and \p qr.
	 */
	static std::vector<qType> reduceSilent(const qType& ql, const qType& qr);
	
	static std::vector<qType> reduceSilent(const qType& ql, const qType& qm, const qType& qr);
	
	/** 
	 * Calculate the irreps of the tensor product of all entries of \p ql with \p qr.
	 * \warning : Returns not only unique irreps.
	 *            Not sure, if we should return only the unique values here. Probably, that should be at least added as an option.
	 */
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr);
	/** 
	 * Calculate the irreps of the tensor product of all entries of \p ql with all entries of \p qr.
	 * \warning : Returns only unique irreps.
	 *            Better: Put an option for unique or non-unique irreps in the return vector.
	 */
	static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE=false);

	static vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr );
	///@}
	
	///@{
	/**
	 * Various coeffecients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
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
	
	static Scalar coeff_test(const qType& q1, const qType& q2, const qType& q3,
						   const qType& q4, const qType& q5, const qType& q6,
						   const qType& q7, const qType& q8, const qType& q9);
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

template<typename Kind, typename Scalar>
std::vector<typename SU2<Kind,Scalar>::qType> SU2<Kind,Scalar>::
reduceSilent( const qType& ql, const qType& qr )
{
	std::vector<qType> vout;
	int qmin = std::abs(ql[0]-qr[0]) +1;
	int qmax = std::abs(ql[0]+qr[0]) -1;
	for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
	return vout;
}

template<typename Kind, typename Scalar>
std::vector<typename SU2<Kind,Scalar>::qType> SU2<Kind,Scalar>::
reduceSilent( const qType& ql, const qType& qm, const qType& qr )
{
	auto qtmp = reduceSilent(ql,qm);
	return reduceSilent(qtmp,qr);
}

template<typename Kind, typename Scalar>
std::vector<typename SU2<Kind,Scalar>::qType> SU2<Kind,Scalar>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<typename SU2<Kind,Scalar>::qType> vout;
	for (std::size_t q=0; q<ql.size(); q++)
	{
		int qmin = std::abs(ql[q][0]-qr[0]) +1;
		int qmax = std::abs(ql[q][0]+qr[0]) -1;
		for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
	}
	return vout;
}

template<typename Kind, typename Scalar>
std::vector<typename SU2<Kind,Scalar>::qType> SU2<Kind,Scalar>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE )
{
	if (UNIQUE)
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
				if( auto it = uniqueControl.find({i}) == uniqueControl.end() ) {uniqueControl.insert({i}); vout.push_back({i});}
			}
		}
		return vout;
	}
	else
	{
		std::vector<qType> vout;
		for (std::size_t q1=0; q1<ql.size(); q1++)
		for (std::size_t q2=0; q2<qr.size(); q2++)
		{
			int qmin = std::abs(ql[q1][0]-qr[q2][0]) +1;
			int qmax = std::abs(ql[q1][0]+qr[q2][0]) -1;
			for ( int i=qmin; i<=qmax; i+=2 ) { vout.push_back({i}); }
		}
		return vout;
	}
}

template<typename Kind, typename Scalar>
vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > SU2<Kind,Scalar>::
tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
//	std::unordered_map<qarray3<1>,std::size_t> dout;
//	size_t j=0;
//	for (std::size_t q1=0; q1<ql.size(); q1++)
//	for (std::size_t q2=0; q2<qr.size(); q2++)
//	{
//		int qmin = std::abs(ql[q1][0]-qr[q2][0]) +1;
//		int qmax = std::abs(ql[q1][0]+qr[q2][0]) -1;
//		for ( int i=qmin; i<=qmax; i+=2 )
//		{
//			dout.insert(make_pair(qarray3<1>{ql[q1], qr[q2], qarray<1>{i}}, j));
//			++j;
//			
//		}
//	}
//	return dout;
	
	vector<tuple<qarray<1>,size_t,qarray<1>,size_t,qarray<1> > > out;
	
	for (std::size_t q1=0; q1<ql.size(); q1++)
	for (std::size_t q2=0; q2<qr.size(); q2++)
	{
		int qmin = std::abs(ql[q1][0]-qr[q2][0]) + 1;
		int qmax = std::abs(ql[q1][0]+qr[q2][0]) - 1;
		for (int i=qmin; i<=qmax; i+=2)
		{
			out.push_back(make_tuple(ql[q1], q1, qr[q2], q2, qarray<1>{i}));
		}
	}
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_dot(const qType& q1)
{
	Scalar out = static_cast<Scalar>(q1[0]);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = static_cast<Scalar>(q1[0]) / static_cast<Scalar>(q2[0]); //* std::pow(static_cast<Scalar>(q2[0]),Scalar(-1.));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]))*
		Scalar(-1.)*phase<Scalar>((q3[0]+q1[0]-q2[0]-1) / 2);
		// Scalar(-1.)*std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = std::sqrt(static_cast<Scalar>(q2[0])) / std::sqrt(static_cast<Scalar>(q1[0]))*
		Scalar(-1.)*phase<Scalar>((q3[0]+q1[0]-q2[0]-1) /2);
		// Scalar(-1.)*std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = phase<Scalar>((q3[0]+q1[0]-q2[0]-1) / 2) * //std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q3[0]+q1[0]-q2[0]-1)) *
		std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
//	Scalar out = gsl_sf_coupling_6j(q1[0]-1,q2[0]-1,q3[0]-1,
//									q4[0]-1,q5[0]-1,q6[0]-1)*
//		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
//		phase<Scalar>((q1[0]+q5[0]+q6[0]-3)/2);
//	return out;
	
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0])*
	std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
	phase<Scalar>((q1[0]+q5[0]+q6[0]-3)/2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
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
	
template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	return coupling_9j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0],q7[0],q8[0],q9[0]) * std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
	
//	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
//									q4[0]-1,q5[0]-1,q6[0]-1,
//									q7[0]-1,q8[0]-1,q9[0]-1)*
//		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
//	return out;
}

 template<typename Kind, typename Scalar>
 Scalar SU2<Kind,Scalar>::
 coeff_test(const qType& q1, const qType& q2, const qType& q3,
 		   const qType& q4, const qType& q5, const qType& q6,
 		   const qType& q7, const qType& q8, const qType& q9)
 {
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]))*
		static_cast<Scalar>(q7[0]) / static_cast<Scalar>(q9[0]);
	return out;
 }

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
									q4[0]-1,q5[0]-1,q6[0]-1,
									q7[0]-1,q8[0]-1,q9[0]-1)*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]))*
		static_cast<Scalar>(q9[0]) / static_cast<Scalar>(q7[0]); //std::pow(static_cast<Scalar>(q7[0]),Scalar(-1.));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	return coupling_9j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0],q7[0],q8[0],q9[0])*
	       std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]))*
	       static_cast<Scalar>(q9[0]) / static_cast<Scalar>(q7[0]);

//	Scalar out = gsl_sf_coupling_9j(q1[0]-1,q2[0]-1,q3[0]-1,
//									q4[0]-1,q5[0]-1,q6[0]-1,
//									q7[0]-1,q8[0]-1,q9[0]-1)*
//		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]))*
//		static_cast<Scalar>(q9[0]) / static_cast<Scalar>(q7[0]); //*std::pow(static_cast<Scalar>(q7[0]),Scalar(-1.));
//	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
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
		// std::pow(Scalar(-1.),Scalar(0.5)*static_cast<Scalar>(q1[0]+q2[0]+q12[0]-3));
	return out;
}

template<typename Kind, typename Scalar>
template<std::size_t M>
bool SU2<Kind,Scalar>::
compare ( const std::array<SU2<Kind,Scalar>::qType,M>& q1, const std::array<SU2<Kind,Scalar>::qType,M>& q2 )
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
bool SU2<Kind,Scalar>::
validate ( const std::array<SU2<Kind,Scalar>::qType,M>& qs )
{
	if constexpr( M > 3 )
				{
					std::vector<SU2<Kind,Scalar>::qType> decomp = SU2<Kind,Scalar>::reduceSilent(qs[0],qs[1]);
					for (std::size_t i=2; i<M; i++)
					{
						decomp = SU2<Kind,Scalar>::reduceSilent(decomp,qs[i]);
					}
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == SU2<Kind,Scalar>::qvacuum() ) { return true; }
					}
					return false;
				}
	else if constexpr( M==3 )
					 {
						 // triangle rule
						 std::vector<SU2<Kind,Scalar>::qType> qTarget = SU2<Kind,Scalar>::reduceSilent(qs[0],qs[1]);
						 bool CHECK=false;
						 for( const auto& q : qTarget )
						 {
							 if(q == qs[2]) {CHECK = true;}
						 }
						 return CHECK;
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
