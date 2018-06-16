#ifndef S1xS2_H_
#define S1xS2_H_

#include <utility>
#include <unordered_map>
#include <iostream>

#include "symmetry/qarray.h"
#include "DmrgTypedefs.h"
#include "DmrgExternal.h"
#include "symmetry/kind_dummies.h"

#include "symmetry/U0.h"
#include "symmetry/U1.h"
#include "symmetry/SU2.h"

namespace Sym{
	
/** 
 * \class S1xS2
 * \ingroup Symmetry
 *
 * This class combines two symmetries and puts a label to each of them.
 *
 */
template<typename S1, typename S2>
class S1xS2
{
public:
	typedef typename S1::Scalar_ Scalar;
	
	S1xS2() {};
	
	static std::string name() { return S1::name()+"⊗"+S2::name(); }
	
	static constexpr bool HAS_CGC = false;
	static constexpr std::size_t Nq=S1::Nq+S2::Nq;
	static constexpr bool NON_ABELIAN = S1::NON_ABELIAN or S2::NON_ABELIAN;
	static constexpr bool IS_TRIVIAL = S1::IS_TRIVIAL and S2::IS_TRIVIAL;
	
	typedef qarray<Nq> qType;
	
	inline static constexpr std::array<KIND,Nq> kind() { return {S1::kind()[0],S2::kind()[0]}; }
	
	inline static qType qvacuum() { return {S1::qvacuum()[0],S2::qvacuum()[0]}; }
	inline static qType flip( const qType& q ) { return {S1::flip({q[0]})[0],S2::flip({q[1]})[0]}; }
	inline static int degeneracy( const qType& q ) { return S1::degeneracy({q[0]})*S2::degeneracy({q[1]}); }
	
	///@{
	/** 
	 * Calculate the irreps of the tensor product of \p ql and \p qr.
	 */
	static std::vector<qType> reduceSilent(const qType& ql, const qType& qr);
	/**
	 * Calculate the irreps of the tensor product of \p ql, \p qm and \p qr.
	 * \note This is independent of the order the quantumnumbers.
	 */
	static std::vector<qType> reduceSilent( const qType& ql, const qType& qm, const qType& qr);
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
	
	static vector<tuple<qarray<S1::Nq+S2::Nq>,size_t,qarray<S1::Nq+S2::Nq>,size_t,qarray<S1::Nq+S2::Nq> > > tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr );
	///@}

	///@{
	/**
	 * Various coefficients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
	 */
	inline static Scalar coeff_unity();
	inline static Scalar coeff_dot(const qType& q1);
	inline static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
	inline static Scalar coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_sign(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);

	static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3,
						   int        q1_z, int        q2_z,        int q3_z) {return 1.;}
	static Scalar coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
							int        q1_z, int        q2_z,        int q3_z) {return 1.;}
	
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

template<typename S1, typename S2>
std::vector<typename S1xS2<S1,S2>::qType> S1xS2<S1,S2>::
reduceSilent( const qType& ql, const qType& qr )
{
	std::vector<typename S1::qType> firstSym = S1::reduceSilent(qarray<1>{ql[0]},qarray<1>{qr[0]});
	std::vector<typename S2::qType> secondSym = S2::reduceSilent(qarray<1>{ql[1]},qarray<1>{qr[1]});

	std::vector<qType> vout;
	for(const auto& q1:firstSym)
	for(const auto& q2:secondSym)
	{
		vout.push_back({q1[0],q2[0]});	
	}
	return vout;
}

template<typename S1, typename S2>
std::vector<typename S1xS2<S1,S2>::qType> S1xS2<S1,S2>::
reduceSilent( const qType& ql, const qType& qm, const qType& qr )
{
	return reduceSilent(reduceSilent(ql,qm),qr);
}

template<typename S1, typename S2>
std::vector<typename S1xS2<S1,S2>::qType> S1xS2<S1,S2>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<qType> vout;

	for (std::size_t q=0; q<ql.size(); q++)
	{
		std::vector<typename S1::qType> firstSym = S1::reduceSilent(qarray<1>{ql[q][0]},qarray<1>{qr[0]});
		std::vector<typename S2::qType> secondSym = S2::reduceSilent(qarray<1>{ql[q][1]},qarray<1>{qr[1]});

		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			vout.push_back({q1[0],q2[0]});	
		}
	}

	return vout;
}

template<typename S1, typename S2>
std::vector<typename S1xS2<S1,S2>::qType> S1xS2<S1,S2>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE )
{
	std::vector<qType> vout;
	std::unordered_set<qType> uniqueControl;
	
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		std::vector<typename S1::qType> firstSym  = S1::reduceSilent(qarray<1>{ql[q][0]},qarray<1>{qr[p][0]});
		std::vector<typename S2::qType> secondSym = S2::reduceSilent(qarray<1>{ql[q][1]},qarray<1>{qr[p][1]});
	
		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			if (UNIQUE)
			{
				if( auto it = uniqueControl.find({q1[0],q2[0]}) == uniqueControl.end() )
				{
					uniqueControl.insert({q1[0],q2[0]});
					vout.push_back({q1[0],q2[0]});
				}
			}
			else
			{
				vout.push_back({q1[0],q2[0]});
			}
		}
	}
	return vout;
}

template<typename S1, typename S2>
vector<tuple<qarray<S1::Nq+S2::Nq>,size_t,qarray<S1::Nq+S2::Nq>,size_t,qarray<S1::Nq+S2::Nq> > > S1xS2<S1,S2>::
tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
	vector<tuple<qarray<Nq>,size_t,qarray<Nq>,size_t,qarray<Nq> > > out;
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		std::vector<typename S1::qType> firstSym  = S1::reduceSilent(qarray<1>{ql[q][0]},qarray<1>{qr[p][0]});
		std::vector<typename S2::qType> secondSym = S2::reduceSilent(qarray<1>{ql[q][1]},qarray<1>{qr[p][1]});
		
		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			out.push_back(make_tuple(ql[q], q, qr[p], p, qarray<2>{q1[0],q2[0]}));
		}
	}
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_dot(const qType& q1)
{
	Scalar out = S1::coeff_dot({q1[0]})*S2::coeff_dot({q1[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = S1::coeff_rightOrtho({q1[0]},{q2[0]})*S2::coeff_rightOrtho({q1[1]},{q2[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = S1::coeff_leftSweep({q1[0]},{q2[0]},{q3[0]})*S2::coeff_leftSweep({q1[1]},{q2[1]},{q3[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_sign(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = S1::coeff_sign({q1[0]},{q2[0]},{q3[0]})*S2::coeff_sign({q1[1]},{q2[1]},{q3[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = S1::coeff_adjoint({q1[0]},{q2[0]},{q3[0]})*S2::coeff_adjoint({q1[1]},{q2[1]},{q3[1]});
	return out;
}

// template<typename S1, typename S2>
// typename S1::Scalar S1xS2<S1,S2>::
// coeff_3j(const qType& q1, const qType& q2, const qType& q3,
// 		 int        q1_z, int        q2_z,        int q3_z)
// {
// 	Scalar out = S1::coeff_3j({q1[0]}, {q2[0]}, {q3[0]}, q1_z, q2_z, q3_z) * S2::coeff_3j({q1[1]}, {q2[1]}, {q3[1]}, q1_z, q2_z, q3_z);
// 	return out;
// }

// template<typename S1, typename S2>
// typename S1::Scalar S1xS2<S1,S2>::
// coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
// 		  int        q1_z, int        q2_z,        int q3_z)
// {
// 	Scalar out = S1::coeff_CGC({q1[0]}, {q2[0]}, {q3[0]}, q1_z, q2_z, q3_z) * S2::coeff_CGC({q1[1]}, {q2[1]}, {q3[1]}, q1_z, q2_z, q3_z);
// 	return out;
// }

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out=S1::coeff_6j({q1[0]},{q2[0]},{q3[0]},
							{q4[0]},{q5[0]},{q6[0]})*
		       S2::coeff_6j({q1[1]},{q2[1]},{q3[1]},
							{q4[1]},{q5[1]},{q6[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out=S1::coeff_Apair({q1[0]},{q2[0]},{q3[0]},
							   {q4[0]},{q5[0]},{q6[0]})*
		       S2::coeff_Apair({q1[1]},{q2[1]},{q3[1]},
							   {q4[1]},{q5[1]},{q6[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1::coeff_9j({q1[0]},{q2[0]},{q3[0]},
							{q4[0]},{q5[0]},{q6[0]},
							{q7[0]},{q8[0]},{q9[0]})*
		       S2::coeff_9j({q1[1]},{q2[1]},{q3[1]},
							{q4[1]},{q5[1]},{q6[1]},
							{q7[1]},{q8[1]},{q9[1]});
	return out;
}
	
template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1::coeff_buildR({q1[0]},{q2[0]},{q3[0]},
	                            {q4[0]},{q5[0]},{q6[0]},
	                            {q7[0]},{q8[0]},{q9[0]})*
		       S2::coeff_buildR({q1[1]},{q2[1]},{q3[1]},
	                            {q4[1]},{q5[1]},{q6[1]},
	                            {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1::coeff_buildL({q1[0]},{q2[0]},{q3[0]},
	                            {q4[0]},{q5[0]},{q6[0]},
	                            {q7[0]},{q8[0]},{q9[0]})*
		       S2::coeff_buildL({q1[1]},{q2[1]},{q3[1]},
	                            {q4[1]},{q5[1]},{q6[1]},
	                            {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1::coeff_HPsi({q1[0]},{q2[0]},{q3[0]},
							  {q4[0]},{q5[0]},{q6[0]},
							  {q7[0]},{q8[0]},{q9[0]})*
		       S2::coeff_HPsi({q1[1]},{q2[1]},{q3[1]},
							  {q4[1]},{q5[1]},{q6[1]},
							  {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_AW(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1::coeff_AW({q1[0]},{q2[0]},{q3[0]},
							{q4[0]},{q5[0]},{q6[0]},
							{q7[0]},{q8[0]},{q9[0]})*
		       S2::coeff_AW({q1[1]},{q2[1]},{q3[1]},
							{q4[1]},{q5[1]},{q6[1]},
							{q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1, typename S2>
typename S1::Scalar_ S1xS2<S1,S2>::
coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6,
			const qType& q7, const qType& q8, const qType& q9,
			const qType& q10, const qType& q11, const qType& q12)
{
	Scalar out=S1::coeff_Wpair({q1[0]},{q2[0]},{q3[0]},
							   {q4[0]},{q5[0]},{q6[0]},
							   {q7[0]},{q8[0]},{q9[0]},
							   {q10[0]},{q11[0]},{q12[0]})*
		       S2::coeff_Wpair({q1[1]},{q2[1]},{q3[1]},
							   {q4[1]},{q5[1]},{q6[1]},
							   {q7[1]},{q8[1]},{q9[1]},
							   {q10[0]},{q11[0]},{q12[0]});
	return out;
}

template<typename S1, typename S2>
template<std::size_t M>
bool S1xS2<S1,S2>::
compare ( const std::array<S1xS2<S1,S2>::qType,M>& q1, const std::array<S1xS2<S1,S2>::qType,M>& q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] > q2[m][0]) { return false; }
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	return false;
}

template<typename S1, typename S2>
template<std::size_t M>
bool S1xS2<S1,S2>::
validate ( const std::array<S1xS2<S1,S2>::qType,M>& qs )
{
	if constexpr( M == 1 or M > 3 ) { return true; }
	else if constexpr( M == 2 )
				{
					std::vector<S1xS2<S1,S2>::qType> decomp = S1xS2<S1,S2>::reduceSilent(qs[0],S1xS2<S1,S2>::flip(qs[1]));
					for (std::size_t i=0; i<decomp.size(); i++)
					{
						if ( decomp[i] == S1xS2<S1,S2>::qvacuum() ) { return true; }
					}
					return false;
				}
	else if constexpr( M==3 )
					 {
						 //todo: check here triangle rule
						 std::vector<S1xS2<S1,S2>::qType> qTarget = S1xS2<S1,S2>::reduceSilent(qs[0],qs[1]);
						 bool CHECK=false;
						 for( const auto& q : qTarget )
						 {
							 if(q == qs[2]) {CHECK = true;}
						 }
						 return CHECK;
					 }
}

} //end namespace Sym

#endif //end S1xS2_H_
