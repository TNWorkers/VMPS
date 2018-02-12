#ifndef U0_H_
#define U0_H_

#include <cstddef>

#include "DmrgTypedefs.h"
#include "symmetry/qarray.h"

/**Dummies for models without symmetries.*/
const std::array<qarray<0>,1> qloc1dummy {qarray<0>{}};
const std::array<qarray<0>,2> qloc2dummy {qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,3> qloc3dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,4> qloc4dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<qarray<0>,8> qloc8dummy {qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}, qarray<0>{}};
const std::array<string,0>    labeldummy{};

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

		inline static constexpr std::array<KIND,Nq> kind() { return {}; }

		inline static constexpr qType qvacuum() {return {};}
		inline static qType flip( const qType& q ) { return {}; }
		inline static int degeneracy( const qType& q ) { return 1; }

		inline static std::vector<qType> reduceSilent( const qType& ql, const qType& qr) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const qType& ql, const qType& qm, const qType& qr) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const qType& qr) { return {{}}; }
		inline static std::vector<qType> reduceSilent( const std::vector<qType>& ql, const std::vector<qType>& qr) { return {{}}; }
		
		inline static std::unordered_map<qarray3<0>,std::size_t> tensorProd ( const std::vector<qType>& ql, const std::vector<qType>& qr )
		{
			std::unordered_map<qarray3<0>,std::size_t> out;
			out.insert(make_pair(qarray3<0>{qarray<0>{},qarray<0>{},qarray<0>{}},0));
			return out;
		};

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

