/** 
 * @defgroup Tensors Tensors
 *
 * Data structures which store symmetric tensors. Only symmetry-allowed non zero blocks are stored.
 * Abelian and non Abelian symmetries are possible.
 * The module consists of the following parts:
 * - Local physical operators are stored as SiteOperator or SiteOperatorQ.
 * - MPS tensors and environments are stored as Biped or Multipede.
 * - Basis and Qbasis are elementary classes to manage the Hilbert space basis.
 * - Several contract-methods are available for operations on these tensors.
 */

