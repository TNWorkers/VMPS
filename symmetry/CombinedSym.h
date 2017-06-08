#ifndef COMBINED_SYM_H
#define COMBINED_SYM_H

#include <utility>
#include <unordered_map>
#include <iostream>

#include "Tuples.h"
#include "qarray.h"

#include "DmrgExternalQ.h"

namespace Sym{
	
/** \class Combined
 * \ingroup Symmetry
 *
 * This class combines several Symmetries.
 *
 */
	template<typename Symmetries...>
	class CombinedSym
	{
	public:
		typedef int qType;
		
		CombinedSym() {};

		static std::string name() { return "U(0) (no Symmetry)"; }

		static constexpr bool HAS_CGC = false;
		static constexpr std::size_t Nq=0;
		static constexpr bool SPECIAL = false;
		
	};

} //end namespace Sym

#endif //end COMBINED_SYM_H

