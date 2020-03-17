#ifndef S1xS2_H_
#define S1xS2_H_

/// \cond
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
/// \endcond

#include "DmrgTypedefs.h"
#include "DmrgExternal.h"

#include "qarray.h"

//include "symmetry/kind_dummies.h"
//include "symmetry/qarray.h"
//include "symmetry/U0.h"
//include "symmetry/U1.h"
//include "symmetry/SU2.h"

namespace Sym{
	
/** 
 * \class S1xS2
 * \ingroup Symmetry
 *
 * This class combines two symmetries.
 *
 */
template<typename S1_, typename S2_>
class S1xS2
{
public:
	typedef typename S1_::Scalar_ Scalar;

	typedef S1_ S1;
	typedef S2_ S2;
	
	S1xS2() {};
	
	static std::string name() { return S1_::name()+"⊗"+S2_::name(); }

	static constexpr std::size_t Nq=S1_::Nq+S2_::Nq;

	static constexpr bool HAS_CGC = false;
	static constexpr bool NON_ABELIAN = S1_::NON_ABELIAN or S2_::NON_ABELIAN;
	static constexpr bool ABELIAN = S1_::ABELIAN and S2_::ABELIAN;
	static constexpr bool IS_TRIVIAL = S1_::IS_TRIVIAL and S2_::IS_TRIVIAL;
	static constexpr bool IS_MODULAR = S1_::IS_MODULAR and S2_::IS_MODULAR;
	static constexpr int MOD_N = S1_::MOD_N * S2_::MOD_N;

	static constexpr bool IS_CHARGE_SU2() { return S1_::IS_CHARGE_SU2() or S2_::IS_CHARGE_SU2(); }
	static constexpr bool IS_SPIN_SU2() { return S1_::IS_SPIN_SU2() or S2_::IS_SPIN_SU2(); }

	static constexpr bool IS_SPIN_U1() { return S1_::IS_SPIN_U1() or S2_::IS_SPIN_U1(); }
	
	static constexpr bool NO_SPIN_SYM() { return S1_::NO_SPIN_SYM() and S2_::NO_SPIN_SYM(); }
	static constexpr bool NO_CHARGE_SYM() { return S1_::NO_CHARGE_SYM() and S2_::NO_CHARGE_SYM(); }
	
	typedef qarray<Nq> qType;
	
	inline static constexpr std::array<KIND,Nq> kind() { return {S1_::kind()[0],S2_::kind()[0]}; }
	
	inline static qType qvacuum() { return join(S1_::qvacuum(),S2_::qvacuum()); }
	inline static qType flip( const qType& q ) { return {S1_::flip({q[0]})[0],S2_::flip({q[1]})[0]}; }
	inline static int degeneracy( const qType& q ) { return S1_::degeneracy({q[0]})*S2_::degeneracy({q[1]}); }

	inline static int spinorFactor() { return S1_::spinorFactor() * S2_::spinorFactor(); }
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
	
	static vector<tuple<qarray<S1_::Nq+S2_::Nq>,size_t,qarray<S1_::Nq+S2_::Nq>,size_t,qarray<S1_::Nq+S2_::Nq> > > tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr );
	///@}

	///@{
	/**
	 * Various coefficients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
	 */
	inline static Scalar coeff_unity();
	inline static Scalar coeff_dot(const qType& q1);
	inline static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
	inline static Scalar coeff_leftSweep(const qType& q1, const qType& q2);

	inline static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);

	static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3,
						   int        q1_z, int        q2_z,        int q3_z) {return 1.;}
	static Scalar coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
							int        q1_z, int        q2_z,        int q3_z) {return 1.;}
	
	inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
									 const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_prod(const qType& q1, const qType& q2, const qType& q3,
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

template<typename S1_, typename S2_>
std::vector<typename S1xS2<S1_,S2_>::qType> S1xS2<S1_,S2_>::
reduceSilent( const qType& ql, const qType& qr )
{
	std::vector<typename S1_::qType> firstSym = S1_::reduceSilent(qarray<1>{ql[0]},qarray<1>{qr[0]});
	std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(qarray<1>{ql[1]},qarray<1>{qr[1]});

	std::vector<qType> vout;
	for(const auto& q1:firstSym)
	for(const auto& q2:secondSym)
	{
		vout.push_back({q1[0],q2[0]});	
	}
	return vout;
}

template<typename S1_, typename S2_>
std::vector<typename S1xS2<S1_,S2_>::qType> S1xS2<S1_,S2_>::
reduceSilent( const qType& ql, const qType& qm, const qType& qr )
{
	return reduceSilent(reduceSilent(ql,qm),qr);
}

template<typename S1_, typename S2_>
std::vector<typename S1xS2<S1_,S2_>::qType> S1xS2<S1_,S2_>::
reduceSilent( const std::vector<qType>& ql, const qType& qr )
{
	std::vector<qType> vout;

	for (std::size_t q=0; q<ql.size(); q++)
	{
		std::vector<typename S1_::qType> firstSym = S1_::reduceSilent(qarray<1>{ql[q][0]},qarray<1>{qr[0]});
		std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(qarray<1>{ql[q][1]},qarray<1>{qr[1]});

		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			vout.push_back({q1[0],q2[0]});	
		}
	}

	return vout;
}

template<typename S1_, typename S2_>
std::vector<typename S1xS2<S1_,S2_>::qType> S1xS2<S1_,S2_>::
reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr, bool UNIQUE )
{
	std::vector<qType> vout;
	std::unordered_set<qType> uniqueControl;
	
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		std::vector<typename S1_::qType> firstSym  = S1_::reduceSilent(qarray<1>{ql[q][0]},qarray<1>{qr[p][0]});
		std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(qarray<1>{ql[q][1]},qarray<1>{qr[p][1]});
	
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

template<typename S1_, typename S2_>
vector<tuple<qarray<S1_::Nq+S2_::Nq>,size_t,qarray<S1_::Nq+S2_::Nq>,size_t,qarray<S1_::Nq+S2_::Nq> > > S1xS2<S1_,S2_>::
tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
	vector<tuple<qarray<Nq>,size_t,qarray<Nq>,size_t,qarray<Nq> > > out;
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		std::vector<typename S1_::qType> firstSym  = S1_::reduceSilent(qarray<1>{ql[q][0]},qarray<1>{qr[p][0]});
		std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(qarray<1>{ql[q][1]},qarray<1>{qr[p][1]});
		
		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			out.push_back(make_tuple(ql[q], q, qr[p], p, qarray<2>{q1[0],q2[0]}));
		}
	}
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_unity()
{
	Scalar out = Scalar(1.);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_dot(const qType& q1)
{
	Scalar out = S1_::coeff_dot({q1[0]})*S2_::coeff_dot({q1[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	Scalar out = S1_::coeff_rightOrtho({q1[0]},{q2[0]})*S2_::coeff_rightOrtho({q1[1]},{q2[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_leftSweep(const qType& q1, const qType& q2)
{
	Scalar out = S1_::coeff_leftSweep({q1[0]},{q2[0]})*S2_::coeff_leftSweep({q1[1]},{q2[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	Scalar out = S1_::coeff_adjoint({q1[0]},{q2[0]},{q3[0]})*S2_::coeff_adjoint({q1[1]},{q2[1]},{q3[1]});
	return out;
}

// template<typename S1_, typename S2_>
// typename S1_::Scalar S1xS2<S1_,S2_>::
// coeff_3j(const qType& q1, const qType& q2, const qType& q3,
// 		 int        q1_z, int        q2_z,        int q3_z)
// {
// 	Scalar out = S1_::coeff_3j({q1[0]}, {q2[0]}, {q3[0]}, q1_z, q2_z, q3_z) * S2_::coeff_3j({q1[1]}, {q2[1]}, {q3[1]}, q1_z, q2_z, q3_z);
// 	return out;
// }

// template<typename S1_, typename S2_>
// typename S1_::Scalar S1xS2<S1_,S2_>::
// coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
// 		  int        q1_z, int        q2_z,        int q3_z)
// {
// 	Scalar out = S1_::coeff_CGC({q1[0]}, {q2[0]}, {q3[0]}, q1_z, q2_z, q3_z) * S2_::coeff_CGC({q1[1]}, {q2[1]}, {q3[1]}, q1_z, q2_z, q3_z);
// 	return out;
// }

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_6j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out=S1_::coeff_6j({q1[0]},{q2[0]},{q3[0]},
							{q4[0]},{q5[0]},{q6[0]})*
		       S2_::coeff_6j({q1[1]},{q2[1]},{q3[1]},
							{q4[1]},{q5[1]},{q6[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out=S1_::coeff_Apair({q1[0]},{q2[0]},{q3[0]},
							   {q4[0]},{q5[0]},{q6[0]})*
		       S2_::coeff_Apair({q1[1]},{q2[1]},{q3[1]},
							   {q4[1]},{q5[1]},{q6[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_prod(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6)
{
	Scalar out=S1_::coeff_prod({q1[0]},{q2[0]},{q3[0]},
							  {q4[0]},{q5[0]},{q6[0]})*
		       S2_::coeff_prod({q1[1]},{q2[1]},{q3[1]},
							  {q4[1]},{q5[1]},{q6[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1_::coeff_9j({q1[0]},{q2[0]},{q3[0]},
							{q4[0]},{q5[0]},{q6[0]},
							{q7[0]},{q8[0]},{q9[0]})*
		       S2_::coeff_9j({q1[1]},{q2[1]},{q3[1]},
							{q4[1]},{q5[1]},{q6[1]},
							{q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1_::coeff_tensorProd({q1[0]},{q2[0]},{q3[0]},
	                            {q4[0]},{q5[0]},{q6[0]},
	                            {q7[0]},{q8[0]},{q9[0]})*
		       S2_::coeff_tensorProd({q1[1]},{q2[1]},{q3[1]},
	                            {q4[1]},{q5[1]},{q6[1]},
	                            {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1_::coeff_buildL({q1[0]},{q2[0]},{q3[0]},
	                            {q4[0]},{q5[0]},{q6[0]},
	                            {q7[0]},{q8[0]},{q9[0]})*
		       S2_::coeff_buildL({q1[1]},{q2[1]},{q3[1]},
	                            {q4[1]},{q5[1]},{q6[1]},
	                            {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1_::coeff_buildR({q1[0]},{q2[0]},{q3[0]},
	                            {q4[0]},{q5[0]},{q6[0]},
	                            {q7[0]},{q8[0]},{q9[0]})*
		       S2_::coeff_buildR({q1[1]},{q2[1]},{q3[1]},
	                            {q4[1]},{q5[1]},{q6[1]},
	                            {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1_::coeff_HPsi({q1[0]},{q2[0]},{q3[0]},
							  {q4[0]},{q5[0]},{q6[0]},
							  {q7[0]},{q8[0]},{q9[0]})*
		       S2_::coeff_HPsi({q1[1]},{q2[1]},{q3[1]},
							  {q4[1]},{q5[1]},{q6[1]},
							  {q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_AW(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	Scalar out=S1_::coeff_AW({q1[0]},{q2[0]},{q3[0]},
							{q4[0]},{q5[0]},{q6[0]},
							{q7[0]},{q8[0]},{q9[0]})*
		       S2_::coeff_AW({q1[1]},{q2[1]},{q3[1]},
							{q4[1]},{q5[1]},{q6[1]},
							{q7[1]},{q8[1]},{q9[1]});
	return out;
}

template<typename S1_, typename S2_>
template<std::size_t M>
bool S1xS2<S1_,S2_>::
compare ( const std::array<S1xS2<S1_,S2_>::qType,M>& q1, const std::array<S1xS2<S1_,S2_>::qType,M>& q2 )
{
	for (std::size_t m=0; m<M; m++)
	{
		if (q1[m][0] > q2[m][0]) { return false; }
		else if (q1[m][0] < q2[m][0]) {return true; }
	}
	return false;
}

template<typename S1_, typename S2_>
bool S1xS2<S1_,S2_>::
triangle ( const std::array<S1xS2<S1_,S2_>::qType,3>& qs )
{
	qarray3<S1_::Nq> q_frstSym; q_frstSym[0][0] = qs[0][0]; q_frstSym[1][0] = qs[1][0]; q_frstSym[2][0] = qs[2][0];
	qarray3<S2_::Nq> q_secdSym; q_secdSym[0][0] = qs[0][1]; q_secdSym[1][0] = qs[1][1]; q_secdSym[2][0] = qs[2][1];

	return (S1_::triangle(q_frstSym) and S2_::triangle(q_secdSym));
}

template<typename S1_, typename S2_>
bool S1xS2<S1_,S2_>::
pair ( const std::array<S1xS2<S1_,S2_>::qType,2>& qs )
{
	qarray2<S1_::Nq> q_frstSym; q_frstSym[0][0] = qs[0][0]; q_frstSym[1][0] = qs[1][0];
	qarray2<S1_::Nq> q_secdSym; q_secdSym[0][0] = qs[0][1]; q_secdSym[1][0] = qs[1][1];

	return (S1_::pair(q_frstSym) and S2_::pair(q_secdSym));
}

template<typename S1_, typename S2_>
template<std::size_t M>
bool S1xS2<S1_,S2_>::
validate ( const std::array<S1xS2<S1_,S2_>::qType,M>& qs )
{
	if constexpr( M == 1 ) { return true; }
	else if constexpr( M == 2 ) { return S1xS2<S1_,S2_>::pair(qs); }
	else if constexpr( M==3 ) { return S1xS2<S1_,S2_>::triangle(qs); }
	else { cout << "This should not be printed out!" << endl; return true; }
}

} //end namespace Sym

#endif //end S1xS2_H_
