/*! \mainpage VMPS++
 *
 * \section intro_sec Introduction
 *
 * This is the introduction.
 *
 * \section install_sec Supported models
 *
 * \subsection heisenberg Heisenberg
 *
 * etc...
 */

/** @defgroup Symmetry Symmetry
 *
 * This group holds all relevant code for the used symmetries.
 *
 * \todo Implement a base class to simplifie the use of several symmetries.
 */

/** @defgroup Models Models
 *
 * All models are defined within this group.
 *
 */

/** @defgroup Heisenberg Heisenberg Models
 * @ingroup Models
 * Different variants of the Heisenberg model.
 *
 */

/** @defgroup Hubbard Hubbard Models
 * @ingroup Models
 * Different variants of the Hubbard model.
 *
 */

/** @defgroup Kondo Kondo Models
 * @ingroup Models
 * Different variants of the Kondo model.
 *
 */

/** @defgroup Fermions Base for fermions
 *
 * This group defines the local fermion operators in different block diagonal forms, depending on the used symmetry.
 *
 * \todo Implement a base class to get rid of duplicated code.
 */

/** @defgroup Spins Base for spins
 *
 * This group defines the local operators for quantum spins (quantum number \f$S\f$) in different block diagonal forms, depending on the used symmetry.
 *
 * \todo Implement a base class to get rid of duplicated code.
 */
