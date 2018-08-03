#ifndef DMRG_MACROS
#define DMRG_MACROS

#define MAKE_TYPEDEFS(MODEL) \
typedef DmrgSolver<MODEL::Symmetry,MODEL,MODEL::Scalar_>  Solver; \
typedef VumpsSolver<MODEL::Symmetry,MODEL,MODEL::Scalar_> uSolver;

#endif
