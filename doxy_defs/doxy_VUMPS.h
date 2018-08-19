 /** 
 * @defgroup VUMPS VUMPS
 * 
 * Implementation of the _Variational Uniform Matrix Product States_ for Abelian and non Abelian symmetries. All equation references are according to the follwing paper:
 * arXiv:1701.07035 "Variational optimization algorithms for uniform matrix product states" (V. Zauner-Stauber, L. Vanderstraeten, M.T. Fishman, F. Verstraete, J. Haegeman, 2017). 
 * Much of the earlier code can be actually reused, the only major additions are:
 * - The calculation of the environment involves infinite contractions that are carried out by solving linear systems eq. 14 or eq. C25ab. This is handled in VUMPS/VumpsTransferMatrix.h\.
 * - The solution of the linear systems requires an GMRES solver. A simple version GMResSolver is implemented in the %ALGS git.
 * - It is possible to run the algorithm with a 2-site Hamiltonian (without an MPO), see algorithm 2. This involves the local updates eq. 11 and eq. 16 which are handled in VUMPS/VumpsPivotStuff.h\. Note that this version is only implemented for Abelian symmetries.
 * - The algorithm requires the calculation of \f$A_L\f$ and \f$A_R\f$ from \f$A_C\f$ and \f$C\f$ (in standard %DMRG, \f$A_C\f$ is decomposed into \f$A_L\f$ or \f$A_R\f$; and \f$C\f$ is multiplied to the next site). This requires the modified sweeping algorithms Umps::svdDecompose (eq. 19, 20) and Umps::polarDecompose (eq. 21ab, 22).
 * - For the overlap between two UMPS, one needs to calculate the dominant eigenvalue of the transfer matrix eq. A7. This is achieved with the ArnoldiSolver from the %ALGS git.

 * Otherwise:
 * - The local update with an MPO involve the contractions eq. C28 and eq. C29. These are easily handled with the already existing pivot classes (1-site and 0-site).
 
 * Arbitrary sizes of unit cells are implemented. (simply create an MPO of this size). Always remember to set \p OPEN_BC=false for VUMPS.
 * 
 */

