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
		typedef int qType;
		
		U0() {};

		static std::string name() { return "U(0) (no Symmetry)"; }

		static constexpr bool HAS_CGC = false;
		static constexpr std::size_t Nq=0;
		static constexpr bool SPECIAL = false;
		
	};

} //end namespace Sym
#endif

