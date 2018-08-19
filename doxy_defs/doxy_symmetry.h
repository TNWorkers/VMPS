/** 
 * @defgroup Symmetry Symmetry
 *
 * The symmetry module provides classes which encapsulate all relevant operations for building symmetry invariant Mps and Mpo 
 * and algorithms for them (e.g. DmrgSolver).
 * The classes enter as template arguments of the most of the other classes, e.g. for the Mps class:
 * \code{.cpp}
 * template<typename Symmetry, typename Scalar>
 * class Mps{}
 * \endcode
 * The classes have to satisfie different requirements:
 * - only static members
 * - the number of independent symmetries Nq, defined as:
 *   \code{.cpp}
 *      constexpr int Nq=...
 *   \endcode
 * - a typedef to the type of the irreducible representations (irreps), this is currently always a typedef to ::qarray:
 *   \code{.cpp}
 *      typedef qarray<Nq> qType
 *   \endcode
 * - a function which has as its input two irreps and returns all irreps of the tensor product of them.
 *   Additionally similiar functions for vectors of irreps are necessary:
 *   \code{.cpp}
 *      std::vector<Symmetry::qType> Symmetry::reduceSilent( const Symmetry::qType &ql, const Symmetry::qType &qr )
 *      std::vector<Symmetry::qType> Symmetry::reduceSilent( const Symmetry::qType &ql, const Symmetry::qType &qm, const Symmetry::qType &qr )
 *      std::vector<Symmetry::qType> Symmetry::reduceSilent( const std::vector<Symmetry::qType> &ql, const Symmetry::qType &qr )
 *      std::vector<Symmetry::qType> Symmetry::reduceSilent( const std::vector<Symmetry::qType> &ql, const std::vector<Symmetry::qType> &qr )
 *   \endcode
 * - a compare function, which defines a strict order for different irreps.
 * - a bunch of coefficients for the different algorithms resulting from contractions or traces over the Clebsch-Gordon spaces.
 *   For abelian symmetries, these coefficients are all trivially equal to unity, but nevertheless they need to be defined! 
 *   For a more detailed description of the different coeffiecients see the class info from Sym::SU2.
 *
 * Currently, there exists implementations for the groups U(1) (Sym::U1) and SU(2) (Sym::SU2) and for a dummy class, 
 * which represents no symmetry (Sym::U0).
 * These classes can be combined with the Sym::S1xS2 class, which takes two arbitrary symmetries as template arguments.
 * By chaining this, it is in principle possible, to get a class for three or more symmetries, 
 * but until now there is no physical model for this szenario (see \ref Models).
 *
 * \todo2
 * 1. The compare function, can be defined outside of the classes, since it is the same for all symmetries.
 * 2. The reduceSilent() functions for vectors of irreps or three or more irreps can be defined outside the classes.
 *   Only the basic function, taking to irreps, is dependent on the symmetry. 
 *          
 */
