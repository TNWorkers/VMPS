#ifndef SU2_H_
#define SU2_H_

/// \cond
#include <array>
#include <cstddef>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#include <boost/rational.hpp>
/// \endcond


#include "DmrgTypedefs.h"
// #include "DmrgExternal.h"
//include "qarray.h"
#include "symmetry/functions.h"
#include "symmetry/SU2Wrappers.h"

namespace Sym{

/** 
 * \class SU2
 * \ingroup Symmetry
 *
 * Class for handling a SU(2) symmetry of a Hamiltonian without explicitly store the Clebsch-Gordon coefficients but with computing \f$(3n)j\f$-symbols.
 *
 * \describe_Scalar
 * \note An implementation for the basic \f$(3n)j\f$ symbols is used from SU2Wrappers.h.
 *       Currently, only the gsl-implementation can be used, but any library which calculates the symbols can be included.
 *       Just add a wrapper in SU2Wrappers.h.
 */
template<typename Kind, typename Scalar=double>
class SU2
{
public:
	typedef Scalar Scalar_;

	static constexpr std::size_t Nq=1;
	
	static constexpr bool HAS_CGC = false;
	static constexpr bool NON_ABELIAN = true;
	static constexpr bool ABELIAN = false;
	static constexpr bool IS_TRIVIAL = false;
	static constexpr bool IS_MODULAR = false;
	static constexpr int MOD_N = 0;

	static constexpr bool IS_CHARGE_SU2() { if constexpr (SU2<Kind,Scalar>::kind()[0] == KIND::T) {return true;} return false; }
	static constexpr bool IS_SPIN_SU2() { if constexpr (SU2<Kind,Scalar>::kind()[0] == KIND::S) {return true;} return false; }

	static constexpr bool IS_SPIN_U1() { return false; }
	
	static constexpr bool NO_SPIN_SYM() { if (SU2<Kind,Scalar>::kind()[0] != KIND::S) {return true;} return false;}
	static constexpr bool NO_CHARGE_SYM() { if (SU2<Kind,Scalar>::kind()[0] != KIND::T) {return true;} return false;}
	
	typedef qarray<Nq> qType;
	
	SU2() {};

	inline static std::string name() { return "SU2"; }
	inline static constexpr std::array<KIND,Nq> kind() { return {Kind::name}; }
	
	inline static qType qvacuum() { return {1}; }
	inline static qType flip( const qType& q ) { return q; }
	inline static int degeneracy( const qType& q ) { return q[0]; }

	inline static int spinorFactor() { return -1; }
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
	static Scalar coeff_leftSweep(const qType& q1, const qType& q2);
	static Scalar coeff_swapPhase(const qType& q1, const qType& q2);

	static Scalar coeff_leftSweep2(const qType& q1, const qType& q2, const qType& q3);
	static Scalar coeff_leftSweep3(const qType& q1, const qType& q2, const qType& q3);
	static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);
	static Scalar coeff_splitAA(const qType& q1, const qType& q2, const qType& q3);
	static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3,
						   int        q1_z, int        q2_z,        int q3_z);
	static Scalar coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
							int        q1_z, int        q2_z,        int q3_z);

	static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
						   const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
							  const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_splitAA(const qType& q1, const qType& q2, const qType& q3,
								const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_prod(const qType& q1, const qType& q2, const qType& q3,
							 const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_MPOprod6(const qType& q1, const qType& q2, const qType& q3,
								 const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_twoSiteGate(const qType& q1, const qType& q2, const qType& q3,
									const qType& q4, const qType& q5, const qType& q6);
	
	static Scalar coeff_9j(const qType& q1, const qType& q2, const qType& q3,
						   const qType& q4, const qType& q5, const qType& q6,
						   const qType& q7, const qType& q8, const qType& q9);
	static Scalar coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
								   const qType& q4, const qType& q5, const qType& q6,
								   const qType& q7, const qType& q8, const qType& q9);
	static Scalar coeff_MPOprod9(const qType& q1, const qType& q2, const qType& q3,
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
	static Scalar coeff_AW(const qType& q1, const qType& q2, const qType& q3,
						   const qType& q4, const qType& q5, const qType& q6,
						   const qType& q7, const qType& q8, const qType& q9);
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
	Scalar out = static_cast<Scalar>(q1[0]) / static_cast<Scalar>(q2[0]);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_leftSweep(const qType& q1, const qType& q2)
{
	Scalar out = std::sqrt(static_cast<Scalar>(q1[0]) / static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_leftSweep2(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = phase<Scalar>((q1[0]-q2[0]+q3[0]-1) / 2) *
		std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_leftSweep3(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = phase<Scalar>((q1[0]-q2[0]-q3[0]-1) / 2) *
		std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_swapPhase(const qType& q1, const qType& q2)
{
	Scalar out = phase<Scalar>((q1[0]+q2[0]-2) /2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = phase<Scalar>((q3[0]+q1[0]-q2[0]-1) / 2) *
		std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_splitAA(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = phase<Scalar>((q1[0]-q2[0]-q3[0]-3) / 2) *
		std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	// Scalar out = std::sqrt(static_cast<Scalar>(q1[0])) / std::sqrt(static_cast<Scalar>(q2[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_3j(const qType& q1, const qType& q2, const qType& q3,
		 int        q1_z, int        q2_z,        int q3_z)
{
	Scalar out = coupling_3j(q1[0],q2[0],q3[0],
							 q1_z ,q2_z ,q3_z);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
		  int        q1_z, int        q2_z,        int q3_z)
{
	// Scalar out = coupling_3j(q1[0], q2[0], q3[0],
	// 						 q1_z , q2_z , -q3_z) *
	// 	phase<Scalar>((-q1[0]+q2[0]-q3_z-2)/2) * sqrt(q3[0]);
	Scalar out = coupling_3j(q1[0], q2[0], q3[0],
							 q1_z , q2_z , -q3_z) *
		phase<Scalar>((q1[0]-q2[0]+q3_z)/2) * sqrt(q3[0]);

	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0]);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{	
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0])*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))
		*phase<Scalar>((q1[0]+q2[0]+q4[0]+q5[0]-4)/2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_splitAA(const qType& q1, const qType& q2, const qType& q3,
			  const qType& q4, const qType& q5, const qType& q6)
{	
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0])*
		std::sqrt(static_cast<Scalar>(q2[0]*q3[0]))
		*phase<Scalar>((q1[0]+q5[0]+q6[0]-3)/2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_prod(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6)
{	
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0])*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
		phase<Scalar>((q1[0]+q5[0]+q6[0]-3)/2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_MPOprod6(const qType& q1, const qType& q2, const qType& q3,
			   const qType& q4, const qType& q5, const qType& q6)
{	
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0])*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))*
		phase<Scalar>((q1[0]+q2[0]+q4[0]+q5[0]-4)/2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_twoSiteGate(const qType& q1, const qType& q2, const qType& q3,
			   const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out = coupling_6j(q1[0],q2[0],q3[0],q4[0],q5[0],q6[0])*
		std::sqrt(static_cast<Scalar>(q3[0]*q6[0]))
		*phase<Scalar>((q1[0]+q2[0]+q4[0]+q5[0]-4)/2);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = coupling_9j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0],
							 q7[0],q8[0],q9[0]);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
                 const qType& q4, const qType& q5, const qType& q6,
                 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = coupling_9j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0],
							 q7[0],q8[0],q9[0])*
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_MPOprod9(const qType& q1, const qType& q2, const qType& q3,
			   const qType& q4, const qType& q5, const qType& q6,
			   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = coupling_9j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0],
							 q7[0],q8[0],q9[0]) *
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = coupling_9j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0],
							 q7[0],q8[0],q9[0]) *
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
   	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = coupling_9j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0],
							 q7[0],q8[0],q9[0]) *
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0])) *
		static_cast<Scalar>(q9[0]) / static_cast<Scalar>(q7[0]);
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out = coupling_9j(q1[0],q2[0],q3[0],
							 q4[0],q5[0],q6[0],
							 q7[0],q8[0],q9[0]) *
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
	return out;
}

template<typename Kind, typename Scalar>
Scalar SU2<Kind,Scalar>::
coeff_AW(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out =  coupling_9j(q1[0],q2[0],q3[0],
							  q4[0],q5[0],q6[0],
							  q7[0],q8[0],q9[0]) *
		std::sqrt(static_cast<Scalar>(q7[0]*q8[0]*q3[0]*q6[0]));
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
bool SU2<Kind,Scalar>::
triangle ( const std::array<SU2<Kind,Scalar>::qType,3>& qs )
{
	//check the triangle rule for angular momenta, but remark that we use the convention q=2S+1
	if (qs[2][0]-1 >= abs(qs[0][0]-qs[1][0]) and qs[2][0]-1 <= qs[0][0]+qs[1][0]-2) { return true;}
	return false;
}

template<typename Kind, typename Scalar>
bool SU2<Kind,Scalar>::
pair ( const std::array<SU2<Kind,Scalar>::qType,2>& qs )
{
	//check if two quantum numbers fulfill the flow equations: simply qin = qout
	if (qs[0] == qs[1]) {return true;}
	return false;
}

template<typename Kind, typename Scalar>
template<std::size_t M>
bool SU2<Kind,Scalar>::
validate ( const std::array<SU2<Kind,Scalar>::qType,M>& qs )
{
	if constexpr( M == 1 ) { return true; }
	else if constexpr( M == 2 ) { return SU2<Kind,Scalar>::pair(qs); }
	else if constexpr( M==3 ) { return SU2<Kind,Scalar>::triangle(qs); }
	else { cout << "This should not be printed out!" << endl; return true; }
		// std::vector<SU2<Kind,Scalar>::qType> decomp = SU2<Kind,Scalar>::reduceSilent(qs[0],qs[1]);
		// for (std::size_t i=2; i<M; i++)
		// {
		// 	decomp = SU2<Kind,Scalar>::reduceSilent(decomp,qs[i]);
		// }
		// for (std::size_t i=0; i<decomp.size(); i++)
		// {
		// 	if ( decomp[i] == SU2<Kind,Scalar>::qvacuum() ) { return true; }
		// }
		// return false;
	// }
}

} //end namespace Sym

#endif
