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
 * 
 */
