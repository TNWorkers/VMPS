/** 
 * \mainpage VMPS++
 *
 * \section intro_sec Introduction
 *
 * The VMPS library (variational matrix product states) is a c++ library for different algorithms, 
 * all based on the structure of matrix product states (MPS) and matrix product operators (MPO).
 * All the datastructures (MPS or MPO) can respect abelian and non abelian symmetries for more efficiency of the algorithms. 
 *
 * \section algorithms Algorithms
 * All algorithms are collected here: \ref Algorithm
 *
 * \section models Supported models
 * All models are collected here: \ref Models
 *
 * \section symmetries Symmetries
 * The different possible symmetries can be found here: \ref Symmetry
 *
 */

/** 
 * @defgroup Symmetry Symmetry
 *
 * This group holds all relevant code for the used symmetries.
 *
 * \todo Implement a base class to simplifie the use of several symmetries.
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
