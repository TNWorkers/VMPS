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

#include "JoinArray.h" //from TOOLS

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
	
	inline static constexpr std::array<KIND,Nq> kind() { return thirdparty::join(S1_::kind(),S2_::kind()); }
	
	inline static constexpr qType qvacuum() { return join(S1_::qvacuum(),S2_::qvacuum()); }
	inline static constexpr std::array<qType,S1::lowest_qs().size()*S2::lowest_qs().size()> lowest_qs()
	{
		std::array<qType,S1::lowest_qs().size()*S2::lowest_qs().size()> out;
		size_t index = 0;
		for (const auto &q1 : S1::lowest_qs())
		for (const auto &q2 : S2::lowest_qs())
		{
			out[index] = join(q1,q2);
			index++;
		}
		return out;
	}
	
	inline static qType flip( const qType& q ) { auto [ql,qr] = disjoin<S1_::Nq,S2_::Nq>(q); return join(S1_::flip(ql),S2_::flip(qr)); }
	inline static int degeneracy( const qType& q ) { auto [ql,qr] = disjoin<S1_::Nq,S2_::Nq>(q); return S1_::degeneracy(ql)*S2_::degeneracy(qr); }

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
	
	static vector<tuple<qarray<Nq>,size_t,qarray<Nq>,size_t,qarray<Nq> > > tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr );
	///@}

	///@{
	/**
	 * Various coefficients, all resulting from contractions or traces of the Clebsch-Gordon coefficients.
	 */
	inline static Scalar coeff_unity();
	inline static Scalar coeff_dot(const qType& q1);
	inline static Scalar coeff_rightOrtho(const qType& q1, const qType& q2);
	inline static Scalar coeff_leftSweep(const qType& q1, const qType& q2);
	
	inline static Scalar coeff_swapPhase(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_adjoint(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_splitAA(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_leftSweep2(const qType& q1, const qType& q2, const qType& q3);
	inline static Scalar coeff_leftSweep3(const qType& q1, const qType& q2, const qType& q3);

	static Scalar coeff_3j(const qType& q1, const qType& q2, const qType& q3,
						   int        q1_z, int        q2_z,        int q3_z) {return 1.;}
	static Scalar coeff_CGC(const qType& q1, const qType& q2, const qType& q3,
							int        q1_z, int        q2_z,        int q3_z) {return 1.;}
	
	inline static Scalar coeff_6j(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
									 const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_splitAA(const qType& q1, const qType& q2, const qType& q3,
								const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_prod(const qType& q1, const qType& q2, const qType& q3,
									const qType& q4, const qType& q5, const qType& q6);
	inline static Scalar coeff_MPOprod6(const qType& q1, const qType& q2, const qType& q3,
										const qType& q4, const qType& q5, const qType& q6);
	static Scalar coeff_twoSiteGate(const qType& q1, const qType& q2, const qType& q3,
									const qType& q4, const qType& q5, const qType& q6);
	
	inline static Scalar coeff_9j(const qType& q1, const qType& q2, const qType& q3,
								  const qType& q4, const qType& q5, const qType& q6,
								  const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
										  const qType& q4, const qType& q5, const qType& q6,
										  const qType& q7, const qType& q8, const qType& q9);
	inline static Scalar coeff_MPOprod9(const qType& q1, const qType& q2, const qType& q3,
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
	auto [ql1,ql2] = disjoin<S1_::Nq,S2_::Nq>(ql);
	auto [qr1,qr2] = disjoin<S1_::Nq,S2_::Nq>(qr);
	std::vector<typename S1_::qType> firstSym = S1_::reduceSilent(ql1,qr1);
	std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(ql2,qr2);

	std::vector<qType> vout;
	for(const auto& q1:firstSym)
	for(const auto& q2:secondSym)
	{
		vout.push_back(join(q1,q2));	
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
		auto [ql1,ql2] = disjoin<S1_::Nq,S2_::Nq>(ql[q]);
		auto [qr1,qr2] = disjoin<S1_::Nq,S2_::Nq>(qr);

		std::vector<typename S1_::qType> firstSym = S1_::reduceSilent(ql1,qr1);
		std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(ql2,qr2);

		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			vout.push_back(join(q1,q2));	
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
		auto [ql1,ql2] = disjoin<S1_::Nq,S2_::Nq>(ql[q]);
		auto [qr1,qr2] = disjoin<S1_::Nq,S2_::Nq>(qr[p]);
		std::vector<typename S1_::qType> firstSym = S1_::reduceSilent(ql1,qr1);
		std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(ql2,qr2);
	
		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			if (UNIQUE)
			{
				if( auto it = uniqueControl.find(join(q1,q2)) == uniqueControl.end() )
				{
					uniqueControl.insert(join(q1,q2));
					vout.push_back(join(q1,q2));
				}
			}
			else
			{
				vout.push_back(join(q1,q2));
			}
		}
	}
	return vout;
}

template<typename S1_, typename S2_>
vector<tuple<qarray<S1xS2<S1_,S2_>::Nq>,size_t,qarray<S1xS2<S1_,S2_>::Nq>,size_t,qarray<S1xS2<S1_,S2_>::Nq> > > S1xS2<S1_,S2_>::
tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
{
	vector<tuple<qarray<Nq>,size_t,qarray<Nq>,size_t,qarray<Nq> > > out;
	for (std::size_t q=0; q<ql.size(); q++)
	for (std::size_t p=0; p<qr.size(); p++)
	{
		auto [ql1,ql2] = disjoin<S1_::Nq,S2_::Nq>(ql[q]);
		auto [qr1,qr2] = disjoin<S1_::Nq,S2_::Nq>(qr[p]);
		std::vector<typename S1_::qType> firstSym = S1_::reduceSilent(ql1,qr1);
		std::vector<typename S2_::qType> secondSym = S2_::reduceSilent(ql2,qr2);
		
		
		for(const auto& q1:firstSym)
		for(const auto& q2:secondSym)
		{
			out.push_back(make_tuple(ql[q], q, qr[p], p, join(q1,q2)));
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
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	Scalar out = S1_::coeff_dot(q1l)*S2_::coeff_dot(q1r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_rightOrtho(const qType& q1, const qType& q2)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	Scalar out = S1_::coeff_rightOrtho(q1l,q2l)*S2_::coeff_rightOrtho(q1r,q2r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_leftSweep(const qType& q1, const qType& q2)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	Scalar out = S1_::coeff_leftSweep(q1l,q2l)*S2_::coeff_leftSweep(q1r,q2r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_swapPhase(const qType& q1, const qType& q2, const qType& q3)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	Scalar out = S1_::coeff_swapPhase(q1l,q2l,q3l)*S2_::coeff_swapPhase(q1r,q2r,q3r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_adjoint(const qType& q1, const qType& q2, const qType& q3)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	Scalar out = S1_::coeff_adjoint(q1l,q2l,q3l)*S2_::coeff_adjoint(q1r,q2r,q3r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_splitAA(const qType& q1, const qType& q2, const qType& q3)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	Scalar out = S1_::coeff_splitAA(q1l,q2l,q3l)*S2_::coeff_splitAA(q1r,q2r,q3r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_leftSweep2(const qType& q1, const qType& q2, const qType& q3)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	Scalar out = S1_::coeff_leftSweep2(q1l,q2l,q3l)*S2_::coeff_leftSweep2(q1r,q2r,q3r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_leftSweep3(const qType& q1, const qType& q2, const qType& q3)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	Scalar out = S1_::coeff_leftSweep3(q1l,q2l,q3l)*S2_::coeff_leftSweep3(q1r,q2r,q3r);
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
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	
	Scalar out=S1_::coeff_6j(q1l,q2l,q3l,
		                     q4l,q5l,q6l)*
		       S2_::coeff_6j(q1r,q2r,q3r,
							 q4r,q5r,q6r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
			const qType& q4, const qType& q5, const qType& q6)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	
	Scalar out=S1_::coeff_Apair(q1l,q2l,q3l,
								q4l,q5l,q6l)*
		       S2_::coeff_Apair(q1r,q2r,q3r,
								q4r,q5r,q6r);	
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_splitAA(const qType& q1, const qType& q2, const qType& q3,
			  const qType& q4, const qType& q5, const qType& q6)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	
	Scalar out=S1_::coeff_splitAA(q1l,q2l,q3l,
								  q4l,q5l,q6l)*
		       S2_::coeff_splitAA(q1r,q2r,q3r,
								  q4r,q5r,q6r);	
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_prod(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	
	Scalar out=S1_::coeff_prod(q1l,q2l,q3l,
							   q4l,q5l,q6l)*
		       S2_::coeff_prod(q1r,q2r,q3r,
							   q4r,q5r,q6r);	
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_MPOprod6(const qType& q1, const qType& q2, const qType& q3,
			   const qType& q4, const qType& q5, const qType& q6)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	
	Scalar out=S1_::coeff_MPOprod6(q1l,q2l,q3l,
								   q4l,q5l,q6l)*
		       S2_::coeff_MPOprod6(q1r,q2r,q3r,
								   q4r,q5r,q6r);	
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_twoSiteGate(const qType& q1, const qType& q2, const qType& q3,
				  const qType& q4, const qType& q5, const qType& q6)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	
	Scalar out=S1_::coeff_twoSiteGate(q1l,q2l,q3l,
									  q4l,q5l,q6l)*
		       S2_::coeff_twoSiteGate(q1r,q2r,q3r,
									  q4r,q5r,q6r);	
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_9j(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_9j(q1l,q2l,q3l,
							 q4l,q5l,q6l,
							 q7l,q8l,q9l)*
		       S2_::coeff_9j(q1r,q2r,q3r,
							 q4r,q5r,q6r,
							 q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_tensorProd(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_tensorProd(q1l,q2l,q3l,
									 q4l,q5l,q6l,
									 q7l,q8l,q9l)*
		       S2_::coeff_tensorProd(q1r,q2r,q3r,
									 q4r,q5r,q6r,
									 q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_MPOprod9(const qType& q1, const qType& q2, const qType& q3,
			   const qType& q4, const qType& q5, const qType& q6,
			   const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_MPOprod9(q1l,q2l,q3l,
								   q4l,q5l,q6l,
								   q7l,q8l,q9l)*
		       S2_::coeff_MPOprod9(q1r,q2r,q3r,
								   q4r,q5r,q6r,
								   q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_buildL(q1l,q2l,q3l,
								 q4l,q5l,q6l,
								 q7l,q8l,q9l)*
		       S2_::coeff_buildL(q1r,q2r,q3r,
								 q4r,q5r,q6r,
								 q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
			 const qType& q4, const qType& q5, const qType& q6,
			 const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_buildR(q1l,q2l,q3l,
								 q4l,q5l,q6l,
								 q7l,q8l,q9l)*
		       S2_::coeff_buildR(q1r,q2r,q3r,
								 q4r,q5r,q6r,
								 q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
		   const qType& q4, const qType& q5, const qType& q6,
		   const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_HPsi(q1l,q2l,q3l,
							   q4l,q5l,q6l,
							   q7l,q8l,q9l)*
		       S2_::coeff_HPsi(q1r,q2r,q3r,
							   q4r,q5r,q6r,
							   q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
typename S1_::Scalar_ S1xS2<S1_,S2_>::
coeff_AW(const qType& q1, const qType& q2, const qType& q3,
		 const qType& q4, const qType& q5, const qType& q6,
		 const qType& q7, const qType& q8, const qType& q9)
{
	auto [q1l,q1r] = disjoin<S1_::Nq,S2_::Nq>(q1);
	auto [q2l,q2r] = disjoin<S1_::Nq,S2_::Nq>(q2);
	auto [q3l,q3r] = disjoin<S1_::Nq,S2_::Nq>(q3);
	auto [q4l,q4r] = disjoin<S1_::Nq,S2_::Nq>(q4);
	auto [q5l,q5r] = disjoin<S1_::Nq,S2_::Nq>(q5);
	auto [q6l,q6r] = disjoin<S1_::Nq,S2_::Nq>(q6);
	auto [q7l,q7r] = disjoin<S1_::Nq,S2_::Nq>(q7);
	auto [q8l,q8r] = disjoin<S1_::Nq,S2_::Nq>(q8);
	auto [q9l,q9r] = disjoin<S1_::Nq,S2_::Nq>(q9);
	
	Scalar out=S1_::coeff_AW(q1l,q2l,q3l,
							 q4l,q5l,q6l,
							 q7l,q8l,q9l)*
		       S2_::coeff_AW(q1r,q2r,q3r,
							 q4r,q5r,q6r,
							 q7r,q8r,q9r);
	return out;
}

template<typename S1_, typename S2_>
template<std::size_t M>
bool S1xS2<S1_,S2_>::
compare ( const std::array<S1xS2<S1_,S2_>::qType,M>& q1, const std::array<S1xS2<S1_,S2_>::qType,M>& q2 )
{
	 for (std::size_t m=0; m<M; m++)
        {
                if (q1[m] > q2[m]) { return false; }
                else if (q1[m] < q2[m]) {return true; }
        }
	 return false;

	// std::array<typename S1_::qType,M> q1l;
	// std::array<typename S1_::qType,M> q2l;
	// std::array<typename S2_::qType,M> q1r;
	// std::array<typename S2_::qType,M> q2r;
	
	// for (std::size_t m=0; m<M; m++)
	// {
	// 	auto [q1ll,q1rr] = disjoin<S1_::Nq,S2_::Nq>(q1[m]);
	// 	auto [q2ll,q2rr] = disjoin<S1_::Nq,S2_::Nq>(q2[m]);
	// 	q1l[m] = q1ll;
	// 	q2l[m] = q2ll;
	// 	q1r[m] = q1rr;
	// 	q2r[m] = q2rr;
	// }
	// bool b1 = S1_::compare(q1l,q2l);
	// if (b1) {return b1;}
	// return S2_::compare(q1r,q2r);
}

template<typename S1_, typename S2_>
bool S1xS2<S1_,S2_>::
triangle ( const std::array<S1xS2<S1_,S2_>::qType,3>& qs )
{
	qarray3<S1_::Nq> q_frstSym;
	qarray3<S2_::Nq> q_secdSym;

	for (size_t q=0; q<3; q++)
	{
		auto [q1,q2] = disjoin<S1_::Nq,S2_::Nq>(qs[q]);
		q_frstSym[q] = q1;
		q_secdSym[q] = q2;
	}
	return (S1_::triangle(q_frstSym) and S2_::triangle(q_secdSym));
}

template<typename S1_, typename S2_>
bool S1xS2<S1_,S2_>::
pair ( const std::array<S1xS2<S1_,S2_>::qType,2>& qs )
{
	qarray2<S1_::Nq> q_frstSym;
	qarray2<S1_::Nq> q_secdSym;

	for (size_t q=0; q<2; q++)
	{
		auto [q1,q2] = disjoin<S1_::Nq,S2_::Nq>(qs[q]);
		q_frstSym[q] = q1;
		q_secdSym[q] = q2;
	}
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
