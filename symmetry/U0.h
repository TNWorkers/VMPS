#ifndef U0_H_
#define U0_H_

#include <cstddef>

namespace Sym{
	
/** \class U0
 * \ingroup Symmetry
 *
 * Dummy class for no symmetry.
 *
 */
	class U0 // : SymmetryBase<SymSUN<N,Scalar> >
	{
	public:
		typedef qarray<0> qType;
		
		U0() {};

		static std::string name() { return "U(0) (no Symmetry)"; }

		static constexpr bool HAS_CGC = false;
		static constexpr std::size_t Nq=0;
		static constexpr bool NON_ABELIAN = false;
		static constexpr bool IS_TRIVIAL = true;
		
		inline static qType qvacuum() {return {};}
		inline static qType flip( const qType& q ) { return {}; }
		inline static int degeneracy( const qType& q ) { return 1; }

		inline static std::vector<qType> reduceSilent( const qType& ql, const qType& qr) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr) { return {{}}; }

		inline static double coeff_unity() { return 1.; }
		inline static double coeff_dot(const qType& q1) { return 1.; }
		inline static double coeff_rightOrtho(const qType& q1, const qType& q2) { return 1.; }
		inline static double coeff_leftSweep(const qType& q1, const qType& q2, const qType& q3) { return 1.; }
		inline static double coeff_sign(const qType& q1, const qType& q2, const qType& q3) { return 1.; }
		inline static double coeff_adjoint(const qType& q1, const qType& q2, const qType& q3) { return 1.; }

		inline static double coeff_6j(const qType& q1, const qType& q2, const qType& q3,
									  const qType& q4, const qType& q5, const qType& q6) { return 1.; }
		inline static double coeff_Apair(const qType& q1, const qType& q2, const qType& q3,
										 const qType& q4, const qType& q5, const qType& q6) { return 1.; }

		inline static double coeff_9j(const qType& q1, const qType& q2, const qType& q3,
									  const qType& q4, const qType& q5, const qType& q6,
									  const qType& q7, const qType& q8, const qType& q9) { return 1.; }
		inline static double coeff_buildL(const qType& q1, const qType& q2, const qType& q3,
										  const qType& q4, const qType& q5, const qType& q6,
										  const qType& q7, const qType& q8, const qType& q9) { return 1.; }
		inline static double coeff_buildR(const qType& q1, const qType& q2, const qType& q3,
										  const qType& q4, const qType& q5, const qType& q6,
										  const qType& q7, const qType& q8, const qType& q9) { return 1.; }
		inline static double coeff_HPsi(const qType& q1, const qType& q2, const qType& q3,
										const qType& q4, const qType& q5, const qType& q6,
										const qType& q7, const qType& q8, const qType& q9) { return 1.; }

		inline static double coeff_Wpair(const qType& q1, const qType& q2, const qType& q3,
										 const qType& q4, const qType& q5, const qType& q6,
										 const qType& q7, const qType& q8, const qType& q9,
										 const qType& q10, const qType& q11, const qType& q12) { return 1.; }

		template<std::size_t M> inline static bool validate( const std::array<qType,M>& qs ) { return true; }

	};

} //end namespace Sym
#endif

