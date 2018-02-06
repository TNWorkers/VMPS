/** 
 * \mainpage VMPS++
 *
 * # Introduction
 *
 * The VMPS library (*variational matrix product states*) is a c++ library for different algorithms, 
 * all based on the structure of matrix product states (MPS) and matrix product operators (MPO).
 * All the datastructures (MPS or MPO) can respect abelian and non abelian symmetries for more efficiency of the algorithms. 
 *
 * # Algorithms
 * All algorithms are collected here: \ref Algorithm
 *
 * # Supported models
 * All models are collected here: \ref Models
 *
 * # Symmetries
 * The different possible symmetries can be found here: \ref Symmetry
 *
 */

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
 * \todo  
 * 1. The compare function, can be defined outside of the classes, since it is the same for all symmetries.
 * 2. The reduceSilent() functions for vectors of irreps or three or more irreps can be defined outside the classes.
 *   Only the basic function, taking to irreps, is dependent on the symmetry. 
 *          
 */

/** 
 * @defgroup Algorithm Algorithm
 *
 * Different algorithms that can be applied, bla bla bla
 *
 */

/** 
 * @defgroup Models Models
 *
 * All models are defined within this group.
 *
 */

/** 
 * @defgroup Heisenberg Heisenberg Models
 * @ingroup Models
 * Different variants of the Heisenberg model.
 *
 */

/** 
 * @defgroup Hubbard Hubbard Models
 * @ingroup Models
 * Different variants of the Hubbard model.
 *
 */

/** 
 * @defgroup Kondo Kondo Models
 * @ingroup Models
 * Different variants of the Kondo model.
 *
 */

/** 
 * @defgroup Bases Local Operators for spins and fermions
 *
 * This group defines the local fermion and spin operators in different block diagonal forms, depending on the used symmetry.
 *
 * \todo Implement a base class to get rid of duplicated code.
 */
 
 /** 
 * @defgroup VUMPS VUMPS
 * 
 * Implementation of the _Variational Uniform Matrix Product States_ without symmetries. All equation references are according to the follwing paper:
 * arXiv:1701.07035 "Variational optimization algorithms for uniform matrix product states" (V. Zauner-Stauber, L. Vanderstraeten, M.T. Fishman, F. Verstraete, J. Haegeman, 2017). 
 * Much of the earlier code can be actually reused, the only major additions are:
 * - The calculation of the environment involves infinite contractions that are carried out by solving linear systems eq. 14 or eq. C25ab. This is handled in VUMPS/VumpsTransferMatrix.h\.
 * - The solution of the linear systems requires an GMRES solver. A simple version GMResSolver is implemented in the %LANCZOS git.
 * - It is possible to run the algorithm with a 2-site Hamiltonian (without an MPO), see algorithm 2. This involves the local updates eq. 11 and eq. 16 which are handled in VUMPS/VumpsPivotStuff.h\.
 * - The algorithm requires the calculation of \f$A_L\f$ and \f$A_R\f$ from \f$A_C\f$ and \f$C\f$ (in standard %DMRG, \f$A_C\f$ is decomposed into \f$A_L\f$ or \f$A_R\f$; and \f$C\f$ is multiplied to the next site). This requires the modified sweeping algorithms Umps::svdDecompose (eq. 19, 20) and Umps::polarDecompose (eq. 21ab, 22).
 * - For the overlap between two UMPS, one needs to calculate the dominant eigenvalue of the transfer matrix eq. A7. This is achieved with the ArnoldiSolver from the %LANCZOS git.

 * Otherwise:
 * - The local update with an MPO involve the contractions eq. C28 and eq. C29. These are easily handled with the already existing pivot classes (1-site and 0-site).
 
 * Unit cells of size 2 and 4 are implemented (simply create an MPO of this size). Always remember to set \p OPEN_BC=false for VUMPS.
 * 
 */
