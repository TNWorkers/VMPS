#ifndef STRAWBERRY_MPS_WITH_Q
#define STRAWBERRY_MPS_WITH_Q

/// \cond
#include <set>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <type_traits>
#include <iostream>
#include <fstream>
/// \endcond

#ifdef USE_HDF5_STORAGE
	#include <HDF5Interface.h>
	static double dump_Mps; // dump variable if energy value not saved
#endif

#include "pivot/DmrgPivotMatrix1.h"
#include "DmrgJanitor.h"
#include "MpsBoundaries.h"
#include "Blocker.h"

// Forward Declaration
template<typename Symmetry, typename Scalar> class Mpo;
template<typename Symmetry, typename Scalar> class TwoSiteGate;

/**
 * Matrix Product State with conserved quantum numbers (Abelian and non abelian symmetries).
 * \describe_Symmetry
 * \describe_Scalar
 * \note We define the quantum number flow for the \f$A\f$-tensors as follows: 
 * left auxiliary leg \f$i\f$ is combined with the physical index \f$\sigma\f$ to obtain the right auxiliary leg \f$j\f$.
 * For U(1) this means that \f$i+\sigma=j \f$. 
 * For SU(2) this means that the \f$A\f$-tensor decompose with the CGC \f$C^{i,\sigma\rightarrow j}_{m_i,m_\sigma\rightarrow m_j}\f$.
 */
template<typename Symmetry, typename Scalar=double>
class Mps : public DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	static constexpr size_t Nq = Symmetry::Nq;
	typedef typename Symmetry::qType qType;
	
	// Note: Cannot partially specialize template friends
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class DmrgSolver;
	template<typename Symmetry_, typename S1, typename S2> friend class MpsCompressor;
	template<typename H, typename Symmetry_, typename S1, typename S2, typename V> friend class TDVPPropagator;
	template<typename Symmetry_, typename S1, typename S2> friend
	void HxV (const Mpo<Symmetry_,S1> &H, const Mps<Symmetry_,S2> &Vin, Mps<Symmetry_,S2> &Vout, DMRG::VERBOSITY::OPTION VERBOSITY);
	template<typename Symmetry_, typename S1, typename S2> friend 
	void OxV (const Mpo<Symmetry_,S1> &H, const Mps<Symmetry_,S2> &Vin, Mps<Symmetry_,S2> &Vout, DMRG::BROOM::OPTION TOOL);
	template<typename Symmetry_, typename S1, typename S2> friend 
	void OxV_exact (const Mpo<Symmetry_,S1> &H, const Mps<Symmetry_,S2> &Vin, Mps<Symmetry_,S2> &Vout, double tol_compr, DMRG::VERBOSITY::OPTION VERBOSITY, int max_halfsweeps, int min_halfsweeps, int Minit);
	
	template<typename Symmetry_, typename S_> friend class Mps; // in order to exchange data between real & complex Mps
	
public:
	
	//---constructors---
	
	/**Does nothing.*/
	Mps();
	
	/**
	 * Construct by setting all the relevant parameters.
	 * \param L_input : chain length
	 * \param qloc_input : local basis
	 * \param Qtot_input : target quantum number
	 * \param N_phys_input : the volume of the system (normally (chain length) * (chain width))
	 * \param Nqmax_input : maximal initial number of symmetry blocks per site in the Mps
	 */
	Mps (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input, int Nqmax_input, bool TRIVIAL_BOUNDARIES=true);
	
	/** 
	 * Construct by pulling info from an Mpo.
	 * \param H : chain length and local basis will be retrieved from this Mpo
	 * \param Mmax : size cutoff
	 * \param Qtot_input : target quantum number
	 * \param Nqmax_input : maximal initial number of symmetry blocks per site in the Mps
	 */
	template<typename Hamiltonian> Mps (const Hamiltonian &H, size_t Mmax, qarray<Nq> Qtot_input, int Nqmax_input);
	
	/** 
	 * Construct by explicitly providing the A-matrices. Basically only for testing purposes.
	 * \param L_input : chain length
	 * \param As : vector of vectors of A matrices ([site][local basis index])
	 * \param qloc_input : vector of local bases for all sites
	 * \param Qtot_input : target quantum number
	 * \param N_phys_input : the volume of the system (normally (chain length) * (chain width))
	 */
	Mps (size_t L_input, const vector<vector<Biped<Symmetry,MatrixType> > > &As,
	     const vector<vector<qarray<Nq> > > &qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input);
	
	#ifdef USE_HDF5_STORAGE
	/**
	 * Construct from an external HDF5 file named <FILENAME>.h5.
	 * \param filename : The format is fixed to .h5, just enter the name without the format.
	 * \warning This method requires hdf5. For more information see https://www.hdfgroup.org/.
	 */
	Mps (string filename) {load(filename);}
	#endif //USE_HDF5_STORAGE
	
	//---set and modify---
	
	///\{
	/**
	 * Sets all matrices to random using boost's uniform distribution from -1 to 1.
	 * \warning Watch for overflow in large chains where one gets exponentially large values when multiplying all the matrices! 
	 *          The safer way is to randomize while sweeping, using Mps::setRandom(size_t loc).
	 */
	void setRandom();
	
	/**Sets all matrices at site \p loc to random using boost's uniform distribution from -1 to 1.*/
	void setRandom (size_t loc);
	
	/**Sets all matrices to zero.*/
	void setZero();
	
	/**
	 * Sweeps through the chain with DMRG::BROOM::QR, creating a canonical Mps.
	 * \param DIR : If DMRG::DIRECTION::LEFT, the result is left-canonical. If DMRG::DIRECTION::RIGHT, the result is right-canonical.
	 */
	void canonize (DMRG::DIRECTION::OPTION DIR=DMRG::DIRECTION::LEFT);
	
	#ifdef USE_HDF5_STORAGE
	///\{
	/**
	 * Save all information of the Mps to the file <FILENAME>.h5.
	 * \param filename : The format is fixed to .h5, Just enter the name without the format.
	 * \param info : Additional information about the used model. Enter the info()-method of the used Mpo here.
	 * \warning This method requires hdf5. For more information see https://www.hdfgroup.org/.
	 * \note For the filename you should use the info string of the currently used Mpo.
	 */
	void save (string filename, string info="none", double energy=std::nan("1"));
	
	/**
	 * Reads all information of the Mps from the file <FILENAME>.h5.
	 * \param filename : the format is fixed to .h5. Just enter the name without the format.
	 * \warning This method requires hdf5. For more information visit https://www.hdfgroup.org/.
	 */
	void load (string filename, double &energy=dump_Mps);
	#endif //USE_HDF5_STORAGE
	
	/**
	 * Determines all subspace quantum numbers and resizes the containers for the blocks. Memory for the matrices remains uninitialized.
	 * \param L_input : chain length
	 * \param qloc_input : local basis
	 * \param Qtot_input : target quantum number
	 * \param Nqmax_input : maximum initial number of symmetry blocks in the Mps per site
	 */
	void outerResize (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, int Nqmax_input=500);
	
	/**
	 * Resizes with all possible blocks.
	 * \param L_input : chain length
	 * \param qloc_input : local basis
	 * \param Qtot_input : target quantum number
	 */
	void outerResizeAll (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input);
	
	/**
	 * Determines all subspace quantum numbers and resizes the containers for the blocks. 
	 * Memory for the matrices remains uninitiated. Pulls info from an Mpo.
	 * \param H : chain length and local basis will be retrieved from this Mpo
	 * \param Qtot_input : target quantum number
	 * \param Nqmax_input : maximum initial number of symmetry blocks in the Mps per site
	 */
	template<typename Hamiltonian> void outerResize (const Hamiltonian &H, qarray<Nq> Qtot_input, int Nqmax_input=500);
	
	/**
	 * Determines all subspace quantum numbers and resizes the containers for the blocks. Memory for the matrices remains uninitiated. 
	 * Pulls info from another Mps.
	 * \param V : chain length, local basis and target quantum number will be equal to this Mps
	 */
	template<typename OtherMatrixType> void outerResize (const Mps<Symmetry,OtherMatrixType> &V);
	
	/**
	 * Resizes the block matrices.
	 * \param Mmax : size cutoff
	 * \note The edges will have the analytically exact size, which may be smaller than \p Mmax.
	 */
	void innerResize (size_t Mmax);
	
	/**
	 * Sets the Mps from a product state configuration.
	 * \param H : Hamiltonian, needed for Mps::outerResize
	 * \param config : classical configuration, a vector of \p qarray, e.g. (+1,-1,+1,-1,...) for the Néel state
	 */
	template<typename Hamiltonian> void setProductState (const Hamiltonian &H, const vector<qarray<Nq> > &config);
	
	/**
	 * Finds broken paths through the quantum number subspaces and mends them by resizing with appropriate zeros. 
	 * This is needed when applying an Mpo which changes quantum numbers, making some paths impossible. 
	 * For example, one can add a particle at the beginning or end of the chain with the same target particle number, 
	 * but if an annihilator is applied in the middle, only the first path survives.
	 * \warning: deprecated!
	 */
	void mend();
	
	/**
	 * Sets the A-matrix at a given site by performing SVD on the C-tensor.
	 * \warning Not implemented for non-abelian symmetries, deprecated!
	 */
	void set_A_from_C (size_t loc, const vector<Tripod<Symmetry,MatrixType> > &C, DMRG::BROOM::OPTION TOOL=DMRG::BROOM::SVD);
	
	void set_Qmultitarget (const vector<qarray<Nq> > &Qmulti_input) {Qmulti = Qmulti_input;};
	
	///@cond INTERNAL
//	/**
//	 * Takes an Mpo and flattens/purifies it into this Mps (to do time propagation in the Heisenberg picture, for example).
//	 * \param Op : the Mpo to be flattened
//	 * \param USE_SQUARE : if \p true, takes the saved square of the Mpo
//	 * \warning: Long time since this has been tested. Might be useful in the future, though.
//	 */
//	template<size_t MpoNq> void setFlattenedMpo (const Mpo<MpoNq,Scalar> &Op, bool USE_SQUARE=false);
	///\}
	///@endcond
	
	//---print infos---
	
	///\{
	/**
	 * Tests the orthogonality of the Mps.
	 * Returns a string with "A"=left-canonical (\f$\sum_s {A^s}^\dag A^s=I\f$), "B"=right-canonical (\f$\sum_s B^s {B^s}^\dag=I\f$), 
	 * "X"=both, "M"=neither; with the pivot site underlined.
	 * \param tol : The check is \f$\|\sum_s {A^s}^\dag A^s-I\|_{\infty} < tol\f$
	*/
	string test_ortho (double tol=1e-8) const;
	
	/**\describe_info*/
	string info() const;
	
	/**Prints the sizes of the matrices for testing purposes.*/
	string Asizes() const;
	
	/**
	 * Checks if the sizes of the block matrices are consistent, so that they can be multiplied.
	 * \param name : how to call the Mps in the output
	 * \returns : string with info
	 */
	string validate (string name="Mps") const;
	
	/**
	 * Writes the subspace connections as a directed graph into a file.
	 * Run \verbatim dot filename.dot -Tpdf -o filename.pdf \endverbatim to create a shiny pdf.
	 * \param filename : gets a ".dot" extension automatically
	 */
	void graph (string filename) const;
	
	/**
	 * Determines the maximal bond dimension per site (sum of \p A.rows or \p A.cols over all subspaces).
	 */
	size_t calc_Mmax() const;
	
	size_t get_Min  (size_t loc) const;
	size_t get_Mout (size_t loc) const;
	
	/**
	 * For SU(2) symmetries, determines the equivalent U(1) bond dimension.
	 */
	size_t calc_fullMmax() const;
	
	/**
	 * Determines the maximal amount of rows or columns per site and per subspace.
	 */
	size_t calc_Dmax() const;
	
	/**
	 * Determines the maximal amount of subspaces per site.
	 */
	size_t calc_Nqmax() const;
	
	size_t Nqout (size_t l) {return outbase[l].Nq();}
	
	/**
	 * Determines the average amount of subspaces per site.
	 */
	double calc_Nqavg() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	///\}
	
	//---linear algebra operations---
	
	///\{
	/**
	 * Adds another Mps to the given one and scales by \p alpha, i\.e\. performs \f$ \mathrel{+}= \alpha \cdot V_{in}\f$.
	 * \param alpha : scalar for scaling
	 * \param Vin : Mps to be added
	 * \param SVD_COMPRESS : If \p true, the resulting Mps is compressed using SVD. If \p false, the summation is exact (direct sum of the matrices).
	 * \warning Not implemented for non-abelian symmetries.
	 */
	template<typename OtherScalar> void addScale (OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin, bool SVD_COMPRESS=false);
	
	/**
	 * Performs Mps::addScale with \p alpha = +1.
	 * \warning Not implemented for non-abelian symmetries.
	 */
	Mps<Symmetry,Scalar>& operator+= (const Mps<Symmetry,Scalar> &Vin);
	
	/**
	 *Performs Mps::addScale with \p alpha = -1.
	 * \warning Not implemented for non-abelian symmetries.
	 */
	Mps<Symmetry,Scalar>& operator-= (const Mps<Symmetry,Scalar> &Vin);
	
	/**
	 * Performs \f$ \mathrel{*}= \alpha\f$. Applies it to the pivot site (if non-orthogonal, to the first site).
	 */
	template<typename OtherScalar> Mps<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	
	/**
	 * Performs \f$ \mathrel{/}= \alpha\f$. Applies it to the pivot site (if non-orthogonal, to the first site).
	 */
	template<typename OtherScalar> Mps<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);

	/**
	 * Act with a two-site gate on the MPS at sites \p l and \p l+1.
	 */
	void applyGate (const TwoSiteGate<Symmetry,Scalar> &gate, size_t l, DMRG::DIRECTION::OPTION DIR);
	
	/**
	 * Casts the matrices from \p Scalar to \p OtherScalar.
	 */
	template<typename OtherScalar> Mps<Symmetry,OtherScalar> cast() const;
	
	/**
	 * Calculates the scalar product with another Mps.
	 */
	Scalar dot (const Mps<Symmetry,Scalar> &Vket) const;
	
	/**
	 * Calculates the squared norm. Exploits the canonical form if possible, calculates the dot product with itself otherwise.
	 */
	double squaredNorm() const;
	
	/** 
	 * Calculates the expectation value with a local operator at the pivot site. 
	 * \param O : local Mpo acting on the pivot side.
	 * \param distance : distance to the end of the support of \p O.
	 */
	template<typename MpoScalar> Scalar locAvg (const Mpo<Symmetry,MpoScalar> &O, size_t distance=0) const;
	
	//**Calculates the expectation value with a local operator at pivot and pivot+1.*/
//	template<typename MpoScalar> Scalar locAvg2 (const Mpo<Symmetry,MpoScalar> &O) const;
	
	/**Swaps with another Mps.*/
	void swap (Mps<Symmetry,Scalar> &V);
	
	/**
	 * Copies the control parameters from another Mps, i.e.\ all the cutoff tolerances specified in DmrgJanitor.
	 */
	void get_controlParams (const Mps<Symmetry,Scalar> &V);
	
	/**For METTS.
	* \warning Not tested and soon abandoned.
	*/
	void collapse();
	///\}
	
	//---sweeping---
	
	///\{
	/**
	 * Performs a sweep step to the right.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc+1
	 * \param BROOM : choice of decomposition
	 * \param H : non-local information from transfer matrices is provided here when \p BROOM is DMRG::BROOM::RICH_SVD
	 * \param DISCARD_V : if \p true, don't multiply the V-matrix onto the next site
	 */
	void rightSweepStep (size_t loc, DMRG::BROOM::OPTION BROOM, PivotMatrix1<Symmetry,Scalar,Scalar> *H = NULL, bool DISCARD_V=false);
	
	/**
	 * Performs a sweep step to the left.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1
	 * \param BROOM : choice of decomposition
	 * \param H : non-local information from transfer matrices is provided here when \p BROOM is DMRG::BROOM::RICH_SVD
	 * \param DISCARD_U : if \p true, don't multiply the U-matrix onto the next site
	 */
	void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION BROOM, PivotMatrix1<Symmetry,Scalar,Scalar> *H = NULL, bool DISCARD_U=false);

	/**
	 * Calculates the nullspace of the site tensor on site \p loc when blocked into direction \p DIR.
	 * \param DIR : direction of the sweep, either LEFT or RIGHT.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1 or \p loc+1
	 * \param N : tensor to write the nullsapce to.
	 * \note The nullspace is obtained by a full QR decomposition.
	 * \note The nullspace is used for error estimation as suggested here: arXiv:1711.01104
	 */
	void calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,MatrixType> > &N);
	
	/**
	 * Performs a two-site sweep.
	 * \param DIR : direction of the sweep, either LEFT or RIGHT.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1 or \p loc+1
	 * \param Apair : pair of two Mps site tensors which are split via an SVD
	 * \param SEPARATE_SV: if \p true, the singular value matrix is discarded (iseful for IDMRG)
	 */
	void sweepStep2 (DMRG::DIRECTION::OPTION DIR, size_t loc, const vector<Biped<Symmetry,MatrixType> > &Apair, bool SEPARATE_SV=false);
	
	
	 // * Performs a two-site sweep and writes the result into \p Al, \p Ar and \p C (useful for IDMRG).
	 // * \param DIR : direction of the sweep, either LEFT or RIGHT.
	 // * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1 or \p loc+1
	 // * \param Apair : pair of two Mps site tensors which are split via an SVD
	 // * \param Al : left-orthogonal part goes here
	 // * \param Ar : right-orthogonal part goes here
	 // * \param C : singular values go here
	 // * \param SEPARATE_SV: if \p true, the singular value matrix is discarded (iseful for IDMRG)
	// void sweepStep2 (DMRG::DIRECTION::OPTION DIR, size_t loc, const vector<Biped<Symmetry,MatrixType> > &Apair, 
	//                  vector<Biped<Symmetry,MatrixType> > &Al, vector<Biped<Symmetry,MatrixType> > &Ar, Biped<Symmetry,MatrixType> &C, 
	//                  bool SEPARATE_SV);
	
	/**
	 * Performs an SVD split to the left and writes the zero-site tensor to \p C. Used in TDVPPropagator.
	 */
	void leftSplitStep  (size_t loc, Biped<Symmetry,MatrixType> &C);
	
	/**
	 * Performs an SVD split to the right and writes the zero-site tensor to \p C. Used in TDVPPropagator.
	 */
	void rightSplitStep (size_t loc, Biped<Symmetry,MatrixType> &C);
	
	/**
	 * Absorbs the zero-site tensor \p C (as obtained after an SVD split) into the Mps. Used in TDVPPropagator.
	 * \param loc : site to do the absorption
	 * \param DIR : specifies whether the absorption is on the left-sweep or right-sweep
	 * \param C : the zero-site tensor to be absorbed
	 */
	void absorb (size_t loc, DMRG::DIRECTION::OPTION DIR, const Biped<Symmetry,MatrixType> &C);
	///\}
	
	//---return stuff---
	
	///\{
	/**Returns the target quantum number.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
	
	/**Returns the multi-target quantum number for spectral functions.*/
	inline vector<qarray<Nq> > Qmultitarget() const {return Qmulti;};
	
	/**Returns the local basis.*/
	inline vector<qarray<Nq> > locBasis (size_t loc) const {return qloc[loc];}
	inline vector<vector<qarray<Nq> > > locBasis()   const {return qloc;}
	
	/**Returns the auxiliary ingoing basis.*/
	inline Qbasis<Symmetry> inBasis (size_t loc) const {return inbase[loc];}
	inline vector<Qbasis<Symmetry> > inBasis()   const {return inbase;}
	
	/**Returns the auxiliary outgoing basis.*/
	inline Qbasis<Symmetry> outBasis (size_t loc) const {return outbase[loc];}
	inline vector<Qbasis<Symmetry> > outBasis()   const {return outbase;}
	
	/**Const reference to the A-tensor at site \p loc.*/
	const vector<Biped<Symmetry,MatrixType> > &A_at (size_t loc) const {return A[loc];};
	
	/**Reference to the A-tensor at site \p loc.*/
	vector<Biped<Symmetry,MatrixType> > &A_at (size_t loc) {return A[loc];};
	
	/**Returns the pivot position.*/
	inline int get_pivot() const {return this->pivot;};
	
	/**Returns the truncated weight for all sites.*/
	inline ArrayXd get_truncWeight() const {return truncWeight;};
	
	/**Returns the entropy for all bonds.*/
	inline ArrayXd entropy() const {return S;};
	
	/**Return the full entanglement spectrum, resolved by subspace quantum number.*/
	inline vector<map<qarray<Nq>,ArrayXd> > entanglementSpectrum() const {return SVspec;};
	
	/**Return the entanglement spectrum at the site \p loc (values all subspaces merged and sorted).*/
	std::pair<vector<qarray<Symmetry::Nq> >, ArrayXd> entanglementSpectrumLoc (size_t loc) const;
	///\}
	
//	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > BoundaryL;
//	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > BoundaryR;
//	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,2> A_LR;
//	vector<vector<qarray<Symmetry::Nq> > > qlocLR;
	MpsBoundaries<Symmetry,Scalar> Boundaries;
	
	void set_open_bc()
	{
		Boundaries.set_open_bc(Qtot);
	}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > get_boundaryTensor (DMRG::DIRECTION::OPTION DIR, size_t usePower=1ul) const
	{
		if (usePower == 2ul)
		{
			if (DIR == DMRG::DIRECTION::LEFT) {return Boundaries.Lsq;}
			else                              {return Boundaries.Rsq;}
		}
		else if (usePower == 1ul)
		{
			if (DIR == DMRG::DIRECTION::LEFT) {return Boundaries.L;}
			else                              {return Boundaries.R;}
		}
		else
		{
			throw;
		}
	}
	
	void elongate_hetero (size_t Nleft=0, size_t Nright=0)
	{
		if (Nleft>0 or Nright>0)
		{
			size_t Lcell = Boundaries.length();
			size_t Lleft = Nleft * Lcell;
			size_t Lright = Nright * Lcell;
			size_t Lnew = this->N_sites + Lleft + Lright;
			
			vector<vector<Biped<Symmetry,MatrixType> > > Anew(Lnew);
			vector<vector<qarray<Nq> > > qloc_new(Lnew);
			
//			cout << "Lcell=" << Lcell << endl;
//			cout << "Nleft=" << Nleft << ", Nright=" << Nright << endl;
			for (int l=0; l<Lleft; ++l)
			{
//				cout << "adding AL at: l=" << l << " from cell index=" << l%Lcell << endl;
				Anew    [l] = Boundaries.A[0][l%Lcell];
				qloc_new[l] = Boundaries.qloc[l%Lcell];
			}
			for (int l=0; l<this->N_sites; ++l)
			{
//				cout << "using old A at: l=" << Lleft+l << " old index=" << l << endl;
				Anew    [Lleft+l] = A[l];
				qloc_new[Lleft+l] = qloc[l];
			}
			for (int l=0; l<Lright; ++l)
			{
//				cout << "adding AR at: l=" << Lleft+this->N_sites+l << " from cell index=" << l%Lcell << endl;
				Anew    [Lleft+this->N_sites+l] = Boundaries.A[1][l%Lcell];
				qloc_new[Lleft+this->N_sites+l] = Boundaries.qloc[l%Lcell];
			}
//			cout << endl;
			
			A = Anew;
			qloc = qloc_new;
			this->N_sites = Lnew;
			
			resize_arrays();
			update_inbase();
			update_outbase();
			calc_Qlimits();
		}
	}
	
	void shift_hetero (int Nshift=0)
	{
		if (Nshift!=0)
		{
			size_t Lcell = Boundaries.length();
			size_t Lleft = (Nshift<0)? 0:Nshift*Lcell;
			size_t Lright = (Nshift<0)? abs(Nshift)*Lcell:0;
			
			vector<vector<Biped<Symmetry,MatrixType>>> Anew(this->N_sites);
			vector<vector<qarray<Nq>>> qloc_new(this->N_sites);
			
//			cout << "Nshift=" << Nshift << endl;
			for (size_t l=0; l<Lleft; ++l)
			{
//				cout << "adding AL at: l=" << l << " from cell index=" << posmod(-l,Lcell) << endl;
				Anew    [l] = Boundaries.A[0][posmod(-l,Lcell)];
				qloc_new[l] = Boundaries.qloc[posmod(-l,Lcell)];
			}
			for (size_t l=0; l<this->N_sites-abs(Nshift)*Lcell; ++l)
			{
//				cout << "using old A at: l=" << Lleft+l << " old index=" << l+Lright << endl;
				Anew    [Lleft+l] = A[l+Lright];
				qloc_new[Lleft+l] = qloc[l+Lright];
			}
			for (size_t l=0; l<Lright; ++l)
			{
//				cout << "adding AR at: l=" << Lleft+this->N_sites-abs(Nshift)*Lcell+l << " from cell index=" << l%Lcell << endl;
				Anew    [Lleft+this->N_sites-abs(Nshift)*Lcell+l] = Boundaries.A[1][l%Lcell];
				qloc_new[Lleft+this->N_sites-abs(Nshift)*Lcell+l] = Boundaries.qloc[l%Lcell];
			}
//			cout << endl;
			
			A = Anew;
			qloc = qloc_new;
			
			resize_arrays();
			update_inbase();
			update_outbase();
			calc_Qlimits();
		}
	}
	
	void transform_base (qarray<Symmetry::Nq> Qtot, int L, bool PRINT = false)
	{
		if (Qtot != Symmetry::qvacuum())
		{
			for (size_t l=0; l<qloc.size(); ++l)
			for (size_t i=0; i<qloc[l].size(); ++i)
			for (size_t q=0; q<Symmetry::Nq; ++q)
			{
				if (Symmetry::kind()[q] != Sym::KIND::S and Symmetry::kind()[q] != Sym::KIND::T) //Do not transform the base for non Abelian symmetries
				{
					qloc[l][i][q] = qloc[l][i][q] * L - Qtot[q];
				}
			}
		}
	};
	
//private:
	
	/**volume of the system (normally (chain length) * (chain width))*/
	size_t N_phys;
	
	/**local basis*/
	vector<vector<qarray<Nq> > > qloc;
	
	/**total quantum number*/
	qarray<Nq> Qtot = Symmetry::qvacuum();
	
	/**multi-target quantum number for spectral functions*/
	vector<qarray<Nq> > Qmulti;
	
	/**A-tensor*/
	vector<vector<Biped<Symmetry,MatrixType> > > A; // access: A[l][s].block[q]
	
	/**truncated weight*/
	ArrayXd truncWeight;
	
	/**entropy*/
	ArrayXd S;
	
	vector<map<qarray<Nq>,ArrayXd> > SVspec;
	
	/**bases on all ingoing and outgoing legs of the Mps*/
	vector<Qbasis<Symmetry> > inbase;
	vector<Qbasis<Symmetry> > outbase;
	
	/**pre-calculated bounds for the quantum numbers that result from the finite system*/
	vector<qarray<Nq> > QinTop;
	vector<qarray<Nq> > QinBot;
	vector<qarray<Nq> > QoutTop;
	vector<qarray<Nq> > QoutBot;
	
	/**Calculate quantum number bounds.*/
	void calc_Qlimits();
	/**Set quantum number bounds to +-infinity.*/
	void set_Qlimits_to_inf();
	
	/**Update the bases in case new blocks have appeared or old ones have disappeared*/
	void update_inbase (size_t loc);
	void update_outbase (size_t loc);
	void update_inbase()  {for (size_t l=0; l<this->N_sites; l++) update_inbase(l);}
	void update_outbase() {for (size_t l=0; l<this->N_sites; l++) update_outbase(l);}
	
	/**Shorthand to resize all the relevant arrays: \p A, \p inbase, \p outbase, \p truncWeight, \p S.*/
	void resize_arrays();
	void outerResizeNoSymm();
	
	/**Adds one site at a time in addScale, conserving memory.*/
	template<typename OtherScalar> void add_site (size_t loc, OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin);
	
	/**Enriches the search space in order to dynamically find the right bond dimension. Used in sweeps with the option RICH_SVD.*/
	void enrich_left  (size_t loc, PivotMatrix1<Symmetry,Scalar,Scalar> *H);
	void enrich_right (size_t loc, PivotMatrix1<Symmetry,Scalar,Scalar> *H);
};

template<typename Symmetry, typename Scalar>
string Mps<Symmetry,Scalar>::
info() const
{
	stringstream ss;
	ss << "Mps: ";
	ss << "L=" << this->N_sites;
	if (N_phys>this->N_sites) {ss << ",V=" << N_phys;}
	ss << ", " << Symmetry::name() << ", ";
	
	if (Nq != 0)
	{
		ss << "(";
		for (size_t q=0; q<Nq; ++q)
		{
			ss << Symmetry::kind()[q];
			if (q!=Nq-1) {ss << ",";}
		}
		ss << ")=(" << Sym::format<Symmetry>(Qtot) << "), ";
	}
	
	ss << "pivot=" << this->pivot << ", ";
	
	ss << "Mmax=" << calc_Mmax() << " (";
	if (Symmetry::NON_ABELIAN)
	{
		ss << "full=" << calc_fullMmax() << ", ";
	}
	ss << "Dmax=" << calc_Dmax() << "), ";
	ss << "Nqmax=" << calc_Nqmax() << ", ";
	ss << "Nqavg=" << calc_Nqavg() << ", ";
	ss << "trunc_weight=" << truncWeight.sum();
	ss << "(";
	ss << "max=" << truncWeight.maxCoeff();
	ss << ", eps=" << this->eps_truncWeight;
	ss << ")";
	ss << ", ";
	int lSmax;
	if (this->N_sites > 1)
	{
		S.maxCoeff(&lSmax);
		if (!std::isnan(S(lSmax)) and S(lSmax) > 0)
		{
			ss << "Smax(l=" << lSmax << ")=" << S(lSmax) << ", ";
		}
	}
	ss << "mem=" << round(memory(GB),3) << "GB";
//	ss << endl << " •ortho: " << test_ortho();
//	if (truncWeight.maxCoeff() > this->eps_truncWeight)
//	{
//		lout << termcolor::yellow << "Warning: max. local truncWeight=" << truncWeight.maxCoeff() << " is larger than the tolerance " << this->eps_truncWeight << "!" << termcolor::reset << endl;
//	}
	return ss.str();
}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>::
Mps()
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar>>()
{}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>::
Mps (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input, int Qmax_input, bool TRIVIAL_BOUNDARIES)
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >(L_input), qloc(qloc_input), Qtot(Qtot_input), N_phys(N_phys_input)
{
	if (TRIVIAL_BOUNDARIES) {set_open_bc();}
	else {Boundaries.TRIVIAL_BOUNDARIES = false;}
	Qmulti = vector<qarray<Nq> >(1,Qtot);
	outerResize(L_input, qloc_input, Qtot_input, Qmax_input);
	update_inbase();
	update_outbase();
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
Mps<Symmetry,Scalar>::
Mps (const Hamiltonian &H, size_t Mmax, qarray<Nq> Qtot_input, int Nqmax_input)
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >()
{
	set_open_bc();
	N_phys = H.volume();
	Qmulti = vector<qarray<Nq> >(1,Qtot);
	outerResize(H.length(), H.locBasis(), Qtot_input, Nqmax_input);
	
	update_inbase();
	update_outbase();
	
	if (max(Mmax,calc_Nqmax()) > Mmax) lout << "DmrgSolver: Adjusting Minit to match Qinit: " << Mmax << "→" << calc_Nqmax() << endl;
	innerResize(max(Mmax,calc_Nqmax()));
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[l][s] = A[l][s].cleaned();
	}
	
	update_inbase();
	update_outbase();
}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>::
Mps (size_t L_input, const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &As,
     const vector<vector<qarray<Nq> > > &qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input)
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >(L_input), qloc(qloc_input), Qtot(Qtot_input), N_phys(N_phys_input), A(As)
{
	set_open_bc();
	Qmulti = vector<qarray<Nq> >(1,Qtot);
	assert(As.size() == L_input and qloc_input.size() == L_input);
	resize_arrays();
	update_inbase();
	update_outbase();
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
void Mps<Symmetry,Scalar>::
outerResize (const Hamiltonian &H, qarray<Nq> Qtot_input, int Nqmax_input)
{
	set_open_bc();
	N_phys = H.volume();
	outerResize(H.length(), H.locBasis(), Qtot_input, Nqmax_input);
}

template<typename Symmetry, typename Scalar>
template<typename OtherMatrixType> 
void Mps<Symmetry,Scalar>::
outerResize (const Mps<Symmetry,OtherMatrixType> &V)
{
	this->N_sites = V.N_sites;
	N_phys = V.N_phys;
	qloc = V.qloc;
	Qtot = V.Qtot;
	set_open_bc();
	Qmulti = V.Qmulti;
	
	inbase = V.inbase;
	outbase = V.outbase;
	
	QoutTop = V.QoutTop;
	QoutBot = V.QoutBot;
	QinTop  = V.QinTop;
	QinBot  = V.QinBot;
	
	A.resize(this->N_sites);
	
	truncWeight.resize(this->N_sites); truncWeight.setZero();
	S.resize(this->N_sites-1); S.setConstant(numeric_limits<double>::quiet_NaN());
	SVspec.resize(this->N_sites-1);
	
	for (size_t l=0; l<V.N_sites; ++l)
	{
		A[l].resize(qloc[l].size());
		
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			A[l][s].in = V.A[l][s].in;
			A[l][s].out = V.A[l][s].out;
			A[l][s].block.resize(V.A[l][s].dim);
			A[l][s].dict = V.A[l][s].dict;
			A[l][s].dim = V.A[l][s].dim;
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
resize_arrays()
{
	A.resize(this->N_sites);
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		A[l].resize(qloc[l].size());
	}
	
	inbase.resize(this->N_sites);
	outbase.resize(this->N_sites);
	
	truncWeight.resize(this->N_sites); truncWeight.setZero();
	S.resize(this->N_sites-1); S.setConstant(numeric_limits<double>::quiet_NaN());
	SVspec.resize(this->N_sites-1);
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
calc_Qlimits()
{
	// workaround for empty band
	bool NEED_WORKAROUND = false;
	for (int q=0; q<Symmetry::Nq; ++q)
	{
		if (Symmetry::kind()[q] == Sym::KIND::N and Qtot[q] == 0)
		{
			NEED_WORKAROUND = true;
		}
	}
	
	/*for (int q=0; q<Symmetry::Nq; ++q)
	{
		if (Symmetry::kind()[q] == Sym::KIND::K)
		{
			NEED_WORKAROUND = true;
		}
	}*/
	
//	if (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N)
//	{
//		NEED_WORKAROUND = true;
//	}
	
	if (NEED_WORKAROUND)
	{
		set_Qlimits_to_inf();
	}
	else
	{
		auto lowest_q = [] (const vector<qarray<Nq> > &qs) -> qarray<Nq>
		{
			qarray<Nq> out;
			array<vector<int>,Nq> tmp;
			for (size_t q=0; q<Nq; q++)
			{
				tmp[q].resize(qs.size());
				for (size_t i=0; i<qs.size(); i++)
				{
					tmp[q][i] = qs[i][q];
				}
			}
			for (size_t q=0; q<Nq; q++)
			{
				sort(tmp[q].begin(),tmp[q].end());
				out[q] = tmp[q][0];
			}
			return out;
		};
		
		// For spins: calculate maximal S across the chain
		size_t Smax = 1;
		if (!Symmetry::IS_TRIVIAL)
		{
			for (size_t l=0; l<this->N_sites; ++l)
			for (size_t s=0; s<qloc[l].size(); ++s)
			{
				if (ceil(0.5*(qloc[l][s][0]-1.)) > Smax) {Smax = ceil(0.5*(qloc[l][s][0]-1.));}
			}
		}
//		cout << "Smax=" << Smax << endl;
		
		auto lowest_qs = [Smax] (const vector<qarray<Nq> > &qs) -> vector<qarray<Nq> >
		{
			if (Symmetry::IS_TRIVIAL)
			{
				vector<qarray<Nq> > out(1);
				out[0] = Symmetry::qvacuum();
				return out;
			}
			
//			cout << "in:" << endl;
//			for (size_t i=0; i<qs.size(); ++i)
//			{
//				cout << qs[i] << ", ";
//			}
//			cout << endl;
			
			// sort for every q and remove duplicates
			array<vector<int>,Nq> tmp;
			for (size_t q=0; q<Nq; q++)
			{
				tmp[q].resize(qs.size());
				for (size_t i=0; i<qs.size(); i++)
				{
					tmp[q][i] = qs[i][q];
				}
				sort(tmp[q].begin(),tmp[q].end());
				tmp[q].erase(unique(tmp[q].begin(), tmp[q].end()), tmp[q].end());
			}
			
			// Can have different resulting sizes depending on q...
			Array<size_t,Dynamic,1> tmp_sizes(Nq);
			for (size_t q=0; q<Nq; q++)
			{
				tmp_sizes(q) = tmp[q].size();
			}
	//		cout << "sizes=" << tmp_sizes.transpose() << endl;
			vector<qarray<Nq> > out(min(Smax+1, tmp_sizes.minCoeff()));
			
			for (size_t q=0; q<Nq; q++)
			for (size_t i=0; i<out.size(); ++i)
			{
				out[i][q] = tmp[q][i];
			}
			
//			cout << "out:" << endl;
//			for (size_t i=0; i<out.size(); ++i)
//			{
//				cout << out[i] << ", ";
//			}
//			cout << endl;
			
//			cout << "returning lowest_qs, size=" << out.size() << endl;
			return out;
		};
		
		auto highest_q = [] (const vector<qarray<Nq> > &qs) -> qarray<Nq>
		{
			qarray<Nq> out;
			array<vector<int>,Nq> tmp;
			for (size_t q=0; q<Nq; q++)
			{
				tmp[q].resize(qs.size());
				for (size_t i=0; i<qs.size(); i++)
				{
					tmp[q][i] = qs[i][q];
				}
			}
			for (size_t q=0; q<Nq; q++)
			{
				sort(tmp[q].begin(),tmp[q].end());
				out[q] = tmp[q][qs.size()-1];
			}
			return out;
		};
		
		QinTop.resize(this->N_sites);
		QinBot.resize(this->N_sites);
		QoutTop.resize(this->N_sites);
		QoutBot.resize(this->N_sites);
		
		// If non-trivial boundaries: we have an infinite state with a heterogeneous section, no Qlimits
		if (!Boundaries.IS_TRIVIAL())
		{
	//		cout << termcolor::red << "Boundaries.IS_TRIVIAL()==false, infinite limits" << termcolor::reset << endl;
			set_Qlimits_to_inf();
		}
		else
		{
			vector<vector<qarray<Symmetry::Nq> > > QinBotRange(this->N_sites);
			vector<vector<qarray<Symmetry::Nq> > > QoutBotRange(this->N_sites);
			
			QinTop[0] = Symmetry::qvacuum();
			QinBot[0] = Symmetry::qvacuum();
			QinBotRange[0] = {Symmetry::qvacuum()};
			
			for (size_t l=1; l<this->N_sites; ++l)
			{
				auto new_tops = Symmetry::reduceSilent(qloc[l-1], QinTop[l-1]);
				auto new_bots = Symmetry::reduceSilent(qloc[l-1], QinBotRange[l-1], true);
//				cout << "l=" << l << ", new_tops.size()=" << new_tops.size() << endl;
//				cout << "l=" << l << ", new_bots.size()=" << new_bots.size() << endl;
				
				QinTop[l] = highest_q(new_tops);
//				cout << "highest done!" << endl;
				QinBot[l] = lowest_q(new_bots);
//				cout << "lowest done!" << endl;
				QinBotRange[l] = lowest_qs(new_bots);
//				cout << "a" << endl;
//				cout << "l=" << l << ", QinBotRange.size()=" << QinBotRange.size() << endl;
			}
			
//			cout << "b" << endl;
			QoutTop[this->N_sites-1] = *max_element(Qmulti.begin(), Qmulti.end());
			QoutBot[this->N_sites-1] = *min_element(Qmulti.begin(), Qmulti.end());
			QoutBotRange[this->N_sites-1] = Qmulti; //{Qtot};
			
			for (int l=this->N_sites-2; l>=0; --l)
			{
				vector<qarray<Symmetry::Nq> > qlocflip;
				for (size_t q=0; q<qloc[l+1].size(); ++q)
				{
					qlocflip.push_back(Symmetry::flip(qloc[l+1][q]));
				}
				auto new_tops = Symmetry::reduceSilent(qlocflip, QoutTop[l+1]);
				auto new_bots = Symmetry::reduceSilent(qlocflip, QoutBotRange[l+1]);
				
				QoutTop[l] = highest_q(new_tops);
				QoutBot[l] = lowest_q(new_bots);
				QoutBotRange[l] = lowest_qs(new_bots);
			}
			
			for (size_t l=0; l<this->N_sites; ++l)
			{
				if (l!=0)
				{
					for (size_t q=0; q<Nq; q++)
					{
						QinTop[l][q] = min(QinTop[l][q], QoutTop[l-1][q]);
						QinBot[l][q] = max(QinBot[l][q], QoutBot[l-1][q]);
						if (Symmetry::kind()[q] == Sym::KIND::K)
						{
							QinTop[l][q]  = Symmetry::mod()[q]-1;
							QinBot[l][q]  = 0;
						}
					}
				}
				if (l!=this->N_sites-1)
				{
					for (size_t q=0; q<Nq; q++)
					{
						QoutTop[l][q] = min(QoutTop[l][q], QinTop[l+1][q]);
						QoutBot[l][q] = max(QoutBot[l][q], QinBot[l+1][q]);
						if (Symmetry::kind()[q] == Sym::KIND::K)
						{
							QoutTop[l][q] = Symmetry::mod()[q]-1;
							QoutBot[l][q] = 0;
						}
					}
				}
				
				/*cout << "l=" << l 
					 << ", QinTop[l]=" << QinTop[l] << ", QinBot[l]=" << QinBot[l] 
					 << ", QoutTop[l]=" << QoutTop[l] << ", QoutBot[l]=" << QoutBot[l] 
					 << endl;*/
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::set_Qlimits_to_inf()
{
	lout << termcolor::blue << "Setting Qlimits to infinity!" << termcolor::reset << endl;
	QinTop.resize(this->N_sites);
	QinBot.resize(this->N_sites);
	QoutTop.resize(this->N_sites);
	QoutBot.resize(this->N_sites);
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t q=0; q<Nq; q++)
	{
		// A Z(N) quantum number can only go from 0 to N-1
		if (Symmetry::kind()[q] == Sym::KIND::K)
		{
			QinTop[l][q]  = Symmetry::mod()[q]-1;
			QinBot[l][q]  = 0;
			QoutTop[l][q] = Symmetry::mod()[q]-1;
			QoutBot[l][q] = 0;
		}
		else if (Symmetry::kind()[q] == Sym::KIND::N)
		{
			QinTop[l][q]  = Qtot[q];
			QinBot[l][q]  = 0;
			QoutTop[l][q] = Qtot[q];
			QoutBot[l][q] = 0;
		}
		else
		{
			QinTop[l][q]  = std::numeric_limits<int>::max();
			QinBot[l][q]  = std::numeric_limits<int>::min();
			QoutTop[l][q] = std::numeric_limits<int>::max();
			QoutBot[l][q] = std::numeric_limits<int>::min();
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
outerResize (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, int Nqmax_input)
{
	if (Nqmax_input == -1)
	{
		outerResizeAll(L_input,qloc_input,Qtot_input);
	}
	else
	{
		this->N_sites = L_input;
		qloc = qloc_input;
		Qtot = Qtot_input;
		Qmulti = vector<qarray<Nq> >(1,Qtot);
		this->pivot = -1;
		
		calc_Qlimits();
		
		// take the first Nqmax_input quantum numbers from qs which have the smallerst distance to mean
		auto take_first_elems = [this,Nqmax_input] (const vector<qarray<Nq> > &qs, array<double,Nq> mean, const size_t &loc) -> vector<qarray<Nq> >
		{
			vector<qarray<Nq> > out = qs;
		
			if (out.size() > Nqmax_input)
			{
				// sort the vector first according to the distance to mean
				sort(out.begin(),out.end(),[mean,loc,this] (qarray<Nq> q1, qarray<Nq> q2)
				{
					VectorXd dist_q1(Nq);
					VectorXd dist_q2(Nq);
					for (size_t q=0; q<Nq; q++)
					{
						if (Symmetry::kind()[q] == Sym::KIND::K)
						{
							double Delta = 0.5*Symmetry::mod()[q];
							dist_q1(q) = min( posmod(q1[q]-Qtot[q],Symmetry::mod()[q]), posmod(Qtot[q]-q1[q],Symmetry::mod()[q]) ) / Delta;
							dist_q2(q) = min( posmod(q2[q]-Qtot[q],Symmetry::mod()[q]), posmod(Qtot[q]-q2[q],Symmetry::mod()[q]) ) / Delta;
	//						dist_q1(q) = 0.;
	//						dist_q2(q) = 0.;
						}
						else
						{
							double Delta = QinTop[loc][q] - QinBot[loc][q];
							dist_q1(q) = (q1[q]-mean[q]) / Delta;
							dist_q2(q) = (q2[q]-mean[q]) / Delta;
						}
					}
					return (dist_q1.norm() < dist_q2.norm())? true:false;
				});
			
				out.erase(out.begin()+Nqmax_input, out.end());
			}
			return out;
		};
	
		// Qin_trunc contains the first Nqmax_input blocks (consistent with Qtot) for each site
		vector<vector<qarray<Nq> > > Qin_trunc(this->N_sites+1);
	
		// fill Qin_trunc
		Qin_trunc[0].push_back(Symmetry::qvacuum());
		for (size_t l=1; l<this->N_sites; l++)
		{
			auto new_qs = Symmetry::reduceSilent(Qin_trunc[l-1], qloc[l-1], true);
		
	//		for (int i=0; i<new_qs.size(); ++i)
	//		{
	//			lout << "generated q=" << new_qs[i] << endl;
	//		}
		
			assert(new_qs.size() > 0);
			array<double,Nq> mean;
		
			for (size_t q=0; q<Nq; q++)
			{
				mean[q] = static_cast<double>(Qtot[q])*static_cast<double>(l)/static_cast<double>(this->N_sites);
				//cout << "q=" << q << ", Qtot[q]=" << Qtot[q] << ", mean=" << mean[q] << endl;
				// Cast carefully, otherwise strange implicit cast of Qtot[q] to size_t for negative numbers that makes everything crash
			}
		
			// check if within ranges (QinBot,QinTop) for all q:
			auto candidates = take_first_elems(new_qs,mean,l);
			assert(candidates.size() > 0);
			for (const auto &candidate:candidates)
			{
				//lout << "consider candidate: " << candidate << endl;
				array<bool,Nq> WITHIN_RANGE;
				for (size_t q=0; q<Nq; ++q)
				{
					//cout << "l=" << l << ", q=" << q << ", QinTop[l][q]=" << QinTop[l][q] << ", QinBot[l][q]=" << QinBot[l][q] << ", candidate[q]=" << candidate[q] << endl;
					WITHIN_RANGE[q] = (candidate[q] <= QinTop[l][q] and candidate[q] >= QinBot[l][q]);
				}
				if (all_of(WITHIN_RANGE.begin(), WITHIN_RANGE.end(), [] (bool x) {return x;}))
				{
					Qin_trunc[l].push_back(candidate);
					//lout << "push back candidate: " << candidate << endl;
				}
			}
			assert(Qin_trunc[l].size() > 0);
		}
		Qin_trunc[this->N_sites].push_back(Qtot);
	
		if constexpr (Nq == 0)
		{
			outerResizeNoSymm();
		}
		else
		{
			resize_arrays();
		
			for (size_t l=0; l<this->N_sites; ++l)
			for (size_t s=0; s<qloc[l].size(); ++s)
			{
				for (size_t q=0; q<Qin_trunc[l].size(); ++q)
				{
					qarray<Nq> qin = Qin_trunc[l][q];
					auto qouts = Symmetry::reduceSilent(qloc[l][s],qin);
					for (const auto &qout:qouts)
					{
						auto it = find(Qin_trunc[l+1].begin(), Qin_trunc[l+1].end(), qout);
						if (it != Qin_trunc[l+1].end())
						{
							std::array<qType,2> qinout = {qin,qout};
							if (A[l][s].dict.find(qinout) == A[l][s].dict.end())
							{
								A[l][s].in.push_back(qin);
								A[l][s].out.push_back(qout);
								A[l][s].dict.insert({qinout,A[l][s].size()});
								A[l][s].plusplus();
							}
						}
					}
				}
			
				A[l][s].block.resize(A[l][s].size());
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
outerResizeAll (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input)
{
	this->N_sites = L_input;
	qloc = qloc_input;
	Qtot = Qtot_input;
	Qmulti = vector<qarray<Nq> >(1,Qtot);
	this->pivot = -1;
	
	calc_Qlimits();
	
	vector<vector<qarray<Nq> > > Qin(this->N_sites+1);
	Qin[0].push_back(Symmetry::qvacuum());
	
	for (size_t l=1; l<this->N_sites; l++)
	{
		auto candidates = Symmetry::reduceSilent(Qin[l-1], qloc[l-1], true);
		assert(candidates.size() > 0);
		
		for (const auto &candidate:candidates)
		{
			array<bool,Nq> WITHIN_RANGE;
			for (size_t q=0; q<Nq; ++q)
			{
				WITHIN_RANGE[q] = (candidate[q] <= QinTop[l][q] and candidate[q] >= QinBot[l][q]);
			}
			if (all_of(WITHIN_RANGE.begin(), WITHIN_RANGE.end(), [] (bool x) {return x;}))
			{
				Qin[l].push_back(candidate);
			}
		}
	}
	Qin[this->N_sites].push_back(Qtot);
	
	vector<vector<qarray<Nq> > > Qin_(this->N_sites+1);
	Qin_[0].push_back(Symmetry::qvacuum());
	Qin_[this->N_sites].push_back(Qtot);
	
	for (size_t l=this->N_sites-1; l>=1; l--)
	{
		set<qarray<Nq> > invalids;
		
		for (size_t q=0; q<Qin[l].size(); ++q)
		{
			// Check if Qin[l]+qloc[l] == Qin[l+1] is fulfilled, otherwise Qin[l] is invalid
			auto qouts = Symmetry::reduceSilent(qloc[l],Qin[l][q]);
			for (const auto &qout:qouts)
			{
				if (find(Qin[l+1].begin(), Qin[l+1].end(), qout) != Qin[l+1].end())
				{
					Qin_[l].push_back(Qin[l][q]);
				}
			}
		}
	}
	
	Qin = Qin_;
	
	if constexpr (Nq == 0)
	{
		outerResizeNoSymm();
	}
	else
	{
		resize_arrays();
		
		for (size_t l=0; l<this->N_sites; ++l)
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			for (size_t q=0; q<Qin[l].size(); ++q)
			{
				qarray<Nq> qin = Qin[l][q];
				auto qouts = Symmetry::reduceSilent(qloc[l][s],qin);
				for (const auto &qout:qouts)
				{
					auto it = find(Qin[l+1].begin(), Qin[l+1].end(), qout);
					if (it != Qin[l+1].end())
					{
						std::array<qType,2> qinout = {qin,qout};
						if (A[l][s].dict.find(qinout) == A[l][s].dict.end())
						{
							A[l][s].in.push_back(qin);
							A[l][s].out.push_back(qout);
							A[l][s].dict.insert({qinout,A[l][s].size()});
							A[l][s].plusplus();
						}
					}
				}
			}
			
			A[l][s].block.resize(A[l][s].size());
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
outerResizeNoSymm()
{
	assert (Nq == 0 and "Must have Nq=0 to call outerResizeNoSymm!");
	
	resize_arrays();
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			A[l][s].in.push_back(Symmetry::qvacuum());
			A[l][s].out.push_back(Symmetry::qvacuum());
			A[l][s].dict.insert({qarray2<Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, A[l][s].dim});
			A[l][s].dim = 1;
			A[l][s].block.resize(1);
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
innerResize (size_t Mmax)
{
	if constexpr (Nq == 0)
	{
		size_t Ml = qloc[0].size();
		size_t Mr = qloc[this->N_sites-1].size();
		
		for (size_t s=0; s<Ml; ++s)
		{
			A[0][s].block[0].resize(1,min(Ml,Mmax));
		}
		for (size_t s=0; s<Mr; ++s)
		{
			A[this->N_sites-1][s].block[0].resize(min(Mr,Mmax),1);
		}
		
		for (size_t l=1; l<this->N_sites/2; ++l)
		{
			size_t Ml = qloc[l].size();
			size_t Mr = qloc[this->N_sites-l-1].size();
			
			size_t Nlrows = min(Mmax, (size_t)A[l-1][0].block[0].cols());
			size_t Nlcols = min(Mmax, Nlrows*Ml);
			
			size_t Nrcols = min(Mmax, (size_t)A[this->N_sites-l][0].block[0].rows());
			size_t Nrrows = min(Mmax, Nrcols*Mr);
			
			for (size_t s=0; s<Ml; ++s)
			{
				A[l][s].block[0].resize(Nlrows,Nlcols);
			}
			for (size_t s=0; s<Mr; ++s)
			{
				A[this->N_sites-l-1][s].block[0].resize(Nrrows,Nrcols);
			}
		}
		
		// middle matrix for odd chain length:
		if (this->N_sites%2==1)
		{
			size_t centre = this->N_sites/2;
			int Nrows = A[centre-1][0].block[0].cols();
			int Ncols = A[centre+1][0].block[0].rows();
		
			for (size_t s=0; s<qloc[centre].size(); ++s)
			{
				A[centre][s].block[0].resize(Nrows,Ncols);
			}
		}
	}
	else
	{
		vector<map<qarray<Nq>,size_t> > fromL(this->N_sites+1);
		vector<map<qarray<Nq>,size_t> > fromR(this->N_sites+1);
		
		fromL[0].insert({Symmetry::qvacuum(),1});
		for (size_t l=1; l<this->N_sites+1; ++l)
		{
			assert(Mmax >= outbase[l-1].Nq() and "Choose a greater Minit to have at least one state per QN block.");
			assert(outbase[l-1].Nq() != 0 and "Probably failed to build correct quantum number graph!");
			size_t Dmax_in = Mmax / outbase[l-1].Nq();
			size_t Dmax_in_remainder = Mmax%outbase[l-1].Nq();
			for (size_t qout=0; qout<outbase[l-1].Nq(); ++qout)
			{
				fromL[l].insert({outbase[l-1][qout],0});
				for (size_t s=0; s<qloc[l-1].size(); ++s)
				for (size_t q=0; q<A[l-1][s].dim; ++q)
				{
					if (A[l-1][s].out[q] == outbase[l-1][qout])
					{
						qarray<Nq> qin = A[l-1][s].in[q];
						fromL[l][outbase[l-1][qout]] += fromL[l-1][qin];
					}
				}
				if (outbase[l-1][qout] == Symmetry::qvacuum())
				{
					fromL[l][outbase[l-1][qout]] = min(fromL[l][outbase[l-1][qout]], Dmax_in+Dmax_in_remainder);
				}
				else
				{
					fromL[l][outbase[l-1][qout]] = min(fromL[l][outbase[l-1][qout]], Dmax_in);
				}
			}
		}
//		cout << "LEFT: " << endl;
//		for (int l=0; l<this->N_sites+1; ++l)
//		{
//			cout << "l=" << l << endl;
//			for (auto it=fromL[l].begin(); it!=fromL[l].end(); ++it)
//			{
//				cout << "q=" << it->first << ": " << it->second << endl;
//			}
//		}
//		cout << endl;
		
		for (const auto &Qval:Qmulti)
		{
			fromR[this->N_sites].insert({Qval,1});
		}
		for (size_t l=this->N_sites; l-->0;)
		{
			assert(Mmax >= inbase[l].Nq() and "Choose a greater Minit to have at least one state per QN block.");
			size_t Dmax_out = Mmax / inbase[l].Nq();
			size_t Dmax_out_remainder = Mmax%inbase[l].Nq();
			
			for (size_t qin=0; qin<inbase[l].Nq(); ++qin)
			{
				fromR[l].insert({inbase[l][qin],0});
				for (size_t s=0; s<qloc[l].size(); ++s)
				for (size_t q=0; q<A[l][s].dim; ++q)
				{
					if (A[l][s].in[q] == inbase[l][qin])
					{
						qarray<Nq> qout = A[l][s].out[q];
						fromR[l][inbase[l][qin]] += fromR[l+1][qout];
					}
				}
				if (inbase[l][qin] == Symmetry::qvacuum())
				{
					fromR[l][inbase[l][qin]] = min(fromR[l][inbase[l][qin]], Dmax_out+Dmax_out_remainder);
				}
				else
				{
					fromR[l][inbase[l][qin]] = min(fromR[l][inbase[l][qin]], Dmax_out);
				}
			}
		}
	//	cout << "RIGHT: " << endl;
	//	for (size_t l=0; l<this->N_sites+1; ++l)
	//	{
	//		cout << "l=" << l << endl;
	//		for (auto it=fromR[l].begin(); it!=fromR[l].end(); ++it)
	//		{
	//			cout << "q=" << it->first << ": " << it->second << endl;
	//		}
	//	}
		
		vector<map<qarray<Nq>,size_t> > lrmin(this->N_sites+1);
		for (size_t l=0; l<this->N_sites+1; ++l)
		{
			for (auto it=fromL[l].begin(); it!=fromL[l].end(); ++it)
			{
				qarray<Nq> Qout = it->first;
				size_t Nql = it->second;
				size_t Nqr = fromR[l][Qout];
				lrmin[l].insert({Qout,min(Nql,Nqr)});
			}
		}
		
		for (size_t l=0; l<this->N_sites; ++l)
		{
			size_t Dmax_out = Mmax / outbase[l].Nq();
			size_t Dmax_out_remainder = Mmax%outbase[l].Nq();
			size_t Dmax_in = Mmax / inbase[l].Nq();
			size_t Dmax_in_remainder = Mmax%inbase[l].Nq();
			for (size_t s=0; s<qloc[l].size(); ++s)
			for (size_t q=0; q<A[l][s].dim; ++q)
			{
				qarray<Nq> Qin  = A[l][s].in[q];
				qarray<Nq> Qout = A[l][s].out[q];
				size_t Drow_lim=0; size_t Dcol_lim=0;
				if (Qin == Symmetry::qvacuum() and Qout == Symmetry::qvacuum())
				{
					Drow_lim = Dmax_in+Dmax_in_remainder;
					Dcol_lim = Dmax_out+Dmax_out_remainder;
				}
				else if (Qin == Symmetry::qvacuum() and Qout != Symmetry::qvacuum())
				{
					Drow_lim = Dmax_in+Dmax_in_remainder;
					Dcol_lim = Dmax_out;
				}
				else if (Qin != Symmetry::qvacuum() and Qout == Symmetry::qvacuum())
				{
					Drow_lim = Dmax_in;
					Dcol_lim = Dmax_out+Dmax_out_remainder;
				}
				else
				{
					Drow_lim = Dmax_in;
					Dcol_lim = Dmax_out;
				}
				size_t Nrows = min(lrmin[l][Qin],    Drow_lim);
				size_t Ncols = min(lrmin[l+1][Qout], Dcol_lim);
				A[l][s].block[q].resize(Nrows,Ncols);
			}
		}
	}
	
	update_inbase();
	update_outbase();
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
void Mps<Symmetry,Scalar>::
setProductState (const Hamiltonian &H, const vector<qarray<Nq> > &config)
{
	assert(H.length() == config.size());
	assert(!Symmetry::NON_ABELIAN);
	
	this->N_sites = config.size();
	N_phys = H.volume();
	qloc = H.locBasis();
	Qtot = accumulate(config.begin(),config.end(),Symmetry::qvacuum());
	Qmulti = vector<qarray<Nq> >(1,Qtot);
	this->pivot = -1;
	
	resize_arrays();
	
	vector<qarray<Nq> > qouts(this->N_sites+1);
	qouts[0] = Symmetry::qvacuum();
	for (size_t l=0; l<this->N_sites; ++l)
	{
		qouts[l+1] = accumulate(config.begin(), config.begin()+l+1, Symmetry::qvacuum());
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			qarray<Nq> qout = qouts[l+1];
			qarray<Nq> qin = Symmetry::reduceSilent(qout, Symmetry::flip(qloc[l][s]))[0];
			
			if (qin == qouts[l])
			{
				std::array<qType,2> qinout = {qin,qout};
				if (A[l][s].dict.find(qinout) == A[l][s].dict.end())
				{
					A[l][s].in.push_back(qin);
					A[l][s].out.push_back(qout);
					A[l][s].dict.insert({qinout,A[l][s].size()});
					A[l][s].plusplus();
				}
				
				A[l][s].block.resize(A[l][s].size());
			}
		}
	}
	
	calc_Qlimits();
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		update_inbase(l);
		update_outbase(l);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	{
		A[l][s].block[q].resize(1,1);
		A[l][s].block[q].setConstant(1.);
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
setRandom()
{
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	for (size_t a1=0; a1<A[l][s].block[q].rows(); ++a1)
	for (size_t a2=0; a2<A[l][s].block[q].cols(); ++a2)
	{
		A[l][s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.,true);
	}
	
	this->pivot = -1;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
setRandom (size_t loc)
{
	for (size_t s=0; s<qloc[loc].size(); ++s)
	for (size_t q=0; q<A[loc][s].dim; ++q)
	for (size_t a1=0; a1<A[loc][s].block[q].rows(); ++a1)
	for (size_t a2=0; a2<A[loc][s].block[q].cols(); ++a2)
	{
		A[loc][s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.,true);
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
setZero()
{
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	{
		A[l][s].block[q].setZero();
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
canonize (DMRG::DIRECTION::OPTION DIR)
{
	if (DIR == DMRG::DIRECTION::LEFT)
	{
		this->sweep(0, DMRG::BROOM::QR);
		leftSweepStep(0, DMRG::BROOM::QR);
	}
	else
	{
		this->sweep(this->N_sites-1, DMRG::BROOM::QR);
		rightSweepStep(this->N_sites-1, DMRG::BROOM::QR);
	}
}

#ifdef USE_HDF5_STORAGE
template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
save (string filename, string info, double energy)
{
	assert(Boundaries.IS_TRIVIAL());
	
	std::string append_str = ".h5";
	size_t pos = filename.rfind(append_str);
	if (pos == std::string::npos || pos != filename.size() - append_str.size())
	{
		filename += append_str;
	}
	
	HDF5Interface target(filename, WRITE);
	target.create_group("mps");
	target.create_group("qloc");
	target.create_group("Qtot");
	target.create_group("Qmulti");
	
	string DmaxLabel = "Dmax";
	string NqmaxLabel = "Nqmax";
	string eps_svdLabel = "eps_svd";
	string eps_truncWeightLabel = "eps_truncWeightLabel";
	string alpha_rsvdLabel = "alpha_rsvd";
	string add_infoLabel = "add_info";
	
	// save scalar values
	if (!isnan(energy))
	{
		target.save_scalar(energy,"energy");
	}
	target.save_scalar(this->N_sites,"L");
	target.save_scalar(this->N_phys,"Nphys");
	for (size_t q=0; q<Nq; q++)
	{
		stringstream ss; ss << "q=" << q;
		target.save_scalar(this->Qtot[q],ss.str(),"Qtot");
	}
	target.save_scalar(Qmulti.size(),"QmultiSize");
	for (size_t i=0; i<Qmulti.size(); i++)
	for (size_t q=0; q<Nq; q++)
	{
		stringstream ss; ss << "q=" << q << ",i=" << i;
		target.save_scalar(this->Qmulti[i][q],ss.str(),"Qmulti");
	}
	target.save_scalar(this->calc_Dmax(),DmaxLabel);
	target.save_scalar(this->calc_Nqmax(),NqmaxLabel);
	target.save_scalar(this->min_Nsv,"min_Nsv");
	target.save_scalar(this->max_Nsv,"max_Nsv");
	target.save_scalar(this->eps_svd,eps_svdLabel);
	target.save_scalar(this->eps_truncWeight,eps_truncWeightLabel);
	target.save_scalar(this->alpha_rsvd,alpha_rsvdLabel);
	target.save_scalar(this->get_pivot(),"pivot");
	target.save_char(info,add_infoLabel.c_str());
	
	//save qloc
	for (size_t l=0; l<this->N_sites; ++l)
	{
		stringstream ss; ss << "l=" << l;
		target.save_scalar(qloc[l].size(),ss.str(),"qloc");
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<Nq; q++)
		{
			stringstream tt; tt << "l=" << l << ",s=" << s << ",q=" << q;
			target.save_scalar((qloc[l][s])[q],tt.str(),"qloc");
		}
	}
	
	//save the A-matrices
	string label;
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		stringstream tt; tt << "l=" << l << ",s=" << s;
		target.save_scalar(A[l][s].dim,tt.str());
		for (size_t q=0; q<A[l][s].dim; ++q)
		{
			for (size_t p=0; p<Nq; p++)
			{
				stringstream in; in << "in,l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				stringstream out; out << "out,l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				target.save_scalar((A[l][s].in[q])[p],in.str(),"mps");
				target.save_scalar((A[l][s].out[q])[p],out.str(),"mps");
			}
			stringstream ss;
			ss << l << "_" << s << "_" << "(" << A[l][s].in[q] << "," << A[l][s].out[q] << ")";
			label = ss.str();
			if constexpr (std::is_same<Scalar,complex<double>>::value)
			{
				MatrixXd Re = A[l][s].block[q].real();
				MatrixXd Im = A[l][s].block[q].imag();
				target.save_matrix(Re,label+"Re","mps");
				target.save_matrix(Im,label+"Im","mps");
			}
			else
			{
				target.save_matrix(A[l][s].block[q],label,"mps");
			}
		}
	}
	target.close();
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
load (string filename, double &energy)
{
	std::string append_str = ".h5";
	size_t pos = filename.rfind(append_str);
	if (pos == std::string::npos || pos != filename.size() - append_str.size())
	{
		filename += append_str;
	}
	HDF5Interface source(filename, READ);
	
	string eps_svdLabel = "eps_svd";
	string eps_truncWeightLabel = "eps_truncWeightLabel";
	string alpha_rsvdLabel = "alpha_rsvd";
	size_t QmultiSize;
	
	//load the scalars
	if (source.CHECK("energy"))
	{
		source.load_scalar(energy,"energy");
	}
	source.load_scalar(this->N_sites,"L");
	source.load_scalar(this->N_phys,"Nphys");
	for (size_t q=0; q<Nq; q++)
	{
		stringstream ss; ss << "q=" << q;
		source.load_scalar(this->Qtot[q],ss.str(),"Qtot");
	}
	source.load_scalar(QmultiSize,"QmultiSize");
//	cout << "QmultiSize=" << QmultiSize << endl;
	this->Qmulti.resize(QmultiSize);
	for (size_t i=0; i<QmultiSize; i++)
	for (size_t q=0; q<Nq; q++)
	{
		stringstream ss; ss << "q=" << q << ",i=" << i;
//		cout << "q=" << q << ", i=" << i << endl;
		source.load_scalar(this->Qmulti[i][q],ss.str(),"Qmulti");
	}
	source.load_scalar(this->eps_svd,eps_svdLabel);
	// To ensure older files can be loaded, make check here
	// HAS_GROUP is the same for groups and single objects
	if (source.HAS_GROUP(eps_truncWeightLabel)) source.load_scalar(this->eps_truncWeight,eps_truncWeightLabel);
	source.load_scalar(this->alpha_rsvd,alpha_rsvdLabel);
	source.load_scalar(this->pivot,"pivot");
	source.load_scalar(this->min_Nsv,"min_Nsv");
	source.load_scalar(this->max_Nsv,"max_Nsv");
	
	//load qloc
	qloc.resize(this->N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		stringstream ss; ss << "l=" << l;
		size_t qloc_size;
		source.load_scalar(qloc_size,ss.str(),"qloc");
		qloc[l].resize(qloc_size);
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<Nq; q++)
		{
			stringstream tt; tt << "l=" << l << ",s=" << s << ",q=" << q;
			int Q;
			source.load_scalar(Q,tt.str(),"qloc");
			(qloc[l][s])[q] = Q;
		}
	}
	resize_arrays();
	
	//load the A-matrices
	string label;
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		size_t Asize;
		stringstream tt; tt << "l=" << l << ",s=" << s;
		source.load_scalar(Asize,tt.str());
		for (size_t q=0; q<Asize; ++q)
		{
			qarray<Nq> qin,qout;
			for (size_t p=0; p<Nq; p++)
			{
				stringstream in; in << "in,l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				stringstream out; out << "out,l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				source.load_scalar(qin[p],in.str(),"mps");
				source.load_scalar(qout[p],out.str(),"mps");
			}
			stringstream ss;
			ss << l << "_" << s << "_" << "(" << qin << "," << qout << ")";
			label = ss.str();
			MatrixType mat;
			if constexpr (std::is_same<Scalar,complex<double>>::value)
			{
				MatrixXd Re, Im;
				source.load_matrix(Re, label+"Re", "mps");
				source.load_matrix(Im, label+"Im", "mps");
				mat = Re+1.i*Im;
			}
			else
			{
				source.load_matrix(mat, label, "mps");
			}
			A[l][s].push_back(qin,qout,mat);
		}
	}
	source.close();
	update_inbase();
	update_outbase();
	calc_Qlimits();
	Boundaries.set_open_bc(Qtot);
}
#endif

template<typename Symmetry, typename Scalar>
std::size_t Mps<Symmetry,Scalar>::
calc_Mmax () const
{
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (inbase[l].M()  > res) {res = inbase[l].M();}
		if (outbase[l].M() > res) {res = outbase[l].M();}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
std::size_t Mps<Symmetry,Scalar>::
get_Min (size_t loc) const
{
	return inbase[loc].M();
}

template<typename Symmetry, typename Scalar>
std::size_t Mps<Symmetry,Scalar>::
get_Mout (size_t loc) const
{
	return outbase[loc].M();
}

template<typename Symmetry, typename Scalar>
std::size_t Mps<Symmetry,Scalar>::
calc_fullMmax () const
{
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (inbase[l].fullM()  > res) {res = inbase[l].fullM();}
		if (outbase[l].fullM() > res) {res = outbase[l].fullM();}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
size_t Mps<Symmetry,Scalar>::
calc_Dmax() const
{
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (inbase[l].Dmax()  > res) {res = inbase[l].Dmax();}
		if (outbase[l].Dmax() > res) {res = outbase[l].Dmax();}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
size_t Mps<Symmetry,Scalar>::
calc_Nqmax() const
{
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (inbase[l].Nq()  > res) {res = inbase[l].Nq();}
		if (outbase[l].Nq() > res) {res = outbase[l].Nq();}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
double Mps<Symmetry,Scalar>::
calc_Nqavg() const
{
	double res = 0.;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		res += outbase[l].Nq();
	}
	return res/this->N_sites;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
update_inbase (size_t loc)
{
	inbase[loc].clear();
	inbase[loc].pullData(A[loc],0);
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
update_outbase (size_t loc)
{
	outbase[loc].clear();
	outbase[loc].pullData(A[loc],1);
	
//	if (loc == this->N_sites-1)
//	{
//		vector<qarray<Symmetry::Nq> > Qtot_vector;
//		Qtot_vector.push_back(Symmetry::flip(Qtot));
//		Qbasis<Symmetry> Qtot_flow_out(Qtot_vector);
//		outbase[loc].add(Qtot_flow_out);
//		Qtot = Symmetry::qvacuum();
//	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
leftSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrix1<Symmetry,Scalar,Scalar> *H, bool DISCARD_U)
{
	if (TOOL == DMRG::BROOM::RICH_SVD)
	{
		enrich_left(loc,H);
	}
	ArrayXd truncWeightSub(inbase[loc].Nq()); truncWeightSub.setZero();
	ArrayXd entropySub(inbase[loc].Nq()); entropySub.setZero();
	if (loc != 0) {SVspec[loc-1].clear();}
	
	vector<Biped<Symmetry,MatrixType> > Aloc;
	Aloc.resize(qloc[loc].size());
	vector<Biped<Symmetry,MatrixType> > Aprev; 
	if (loc != 0 and DISCARD_U == false)
	{
		Aprev.resize(qloc[loc-1].size());
	}
	
	Blocker<Symmetry,Scalar> Jim(A[loc],qloc[loc],inbase[loc],outbase[loc]);
	auto Aclump = Jim.Aclump(DMRG::DIRECTION::RIGHT);
	
	bool RETURN_SPEC = false;
	if (loc != 0) RETURN_SPEC = true;
	
	double entropy;
	map<qarray<Nq>,ArrayXd> SVspec_;
	Biped<Symmetry,MatrixType> left,right;
	if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::RICH_SVD or TOOL == DMRG::BROOM::BRUTAL_SVD)
	{
		auto [U,Sigma,Vdag] = Aclump.truncateSVD(this->min_Nsv, this->max_Nsv, this->eps_truncWeight, truncWeight(loc), entropy, SVspec_, false, RETURN_SPEC); //false: DONT PRESERVE MULTIPLETS
		if (loc != 0)
		{
			S(loc-1) = entropy;
			// cout << "loc=" << loc << ", entropy in leftSweepStep: " << S(loc-1) << endl;
			SVspec[loc-1] = SVspec_;
		}
		right = Vdag;
		left = U.contract(Sigma);
	}
	else if (TOOL == DMRG::BROOM::QR)
	{
		auto [Q,R] = Aclump.adjoint().QR(true); //true: receive LQ decomposition by taking adjoint of QR
		left = R;
		right = Q;
	}
	
	Aloc = Jim.reblock(right, DMRG::DIRECTION::RIGHT);
	if (loc != 0 and DISCARD_U == false)
	{
		for (size_t s=0; s<qloc[loc-1].size(); ++s)
		for (size_t q=0; q<A[loc-1][s].dim; ++q)
		{
			MatrixType Mtmp;
			auto itleft = left.dict.find({A[loc-1][s].out[q],A[loc-1][s].out[q]});
			if (itleft != left.dict.end())
			{
				Mtmp = A[loc-1][s].block[q] * left.block[itleft->second];
				auto it = Aprev[s].dict.find(qarray2<Nq>{A[loc-1][s].in[q], A[loc-1][s].out[q]});
				if (Mtmp.size() != 0)
				{
					Aprev[s].try_push_back(A[loc-1][s].in[q], A[loc-1][s].out[q], Mtmp);
				}
			}
		}
	}
	// A[loc-1] = Jim.reblock(U.contract(S), DMRG::DIRECTION::RIGHT);
// 	#ifndef DMRG_DONT_USE_OPENMP
// 	#pragma omp parallel for
// 	#endif
// 	for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
// 	{
// 		// determine how many A's to glue together
// 		vector<size_t> svec, qvec, Ncolsvec;
// 		for (size_t s=0; s<qloc[loc].size(); ++s)
// 		for (size_t q=0; q<A[loc][s].dim; ++q)
// 		{
// 			if (A[loc][s].in[q] == inbase[loc][qin] and
// 			    A[loc][s].block[q].rows() > 0 and
// 			    A[loc][s].block[q].cols() > 0)
// 			{
// 				svec.push_back(s);
// 				qvec.push_back(q);
// 				Ncolsvec.push_back(A[loc][s].block[q].cols());
// 			}
// 		}
		
// 		if (Ncolsvec.size() > 0)
// 		{
// 			// do the glue
// 			size_t Nrows = A[loc][svec[0]].block[qvec[0]].rows();
// //			if (Qtot[0] == 5)
// //			{
// //				for (size_t i=0; i<svec.size(); ++i)
// //				{
// //					cout << "loc=" << loc << ", i=" << i << ", A[loc][svec[i]].block[qvec[i]].rows()=" << A[loc][svec[i]].block[qvec[i]].rows() 
// //					     << ", in=" << A[loc][svec[i]].in[qvec[i]] << ", out=" << A[loc][svec[i]].out[qvec[i]] << endl;
// //				}
// //				cout << endl;
// //			}
// 			for (size_t i=1; i<svec.size(); ++i) assert(A[loc][svec[i]].block[qvec[i]].rows() == Nrows);
// 			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
// 			MatrixType Aclump(Nrows,Ncols);
// 			size_t stitch = 0;
// 			for (size_t i=0; i<svec.size(); ++i)
// 			{
// 				 Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[loc][svec[i]].block[qvec[i]]*
// 				 	                                        Symmetry::coeff_leftSweep(
// 				 	                                         A[loc][svec[i]].out[qvec[i]],
// 				 	                                         A[loc][svec[i]].in[qvec[i]]);
// //				Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[loc][svec[i]].block[qvec[i]]*
// //					                                        Symmetry::coeff_leftSweep2(
// //					                                         A[loc][svec[i]].out[qvec[i]],
// //					                                         A[loc][svec[i]].in[qvec[i]],
// //															 qloc[loc][svec[i]]);
// 				stitch += Ncolsvec[i];
// 			}
			
// 			#ifdef DONT_USE_BDCSVD
// 			JacobiSVD<MatrixType> Jack; // standard SVD
// 			#else
// 			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
// 			#endif
			
// 			HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // QR
			
// 			size_t Nret = Nrows; // retained states
			
// 			if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
// 			{
// 				Jack.compute(Aclump,ComputeThinU|ComputeThinV);
// 				ArrayXd SV = Jack.singularValues();
// 				// sqrt(Symmetry::degeneracy(inbase[loc][qin]))
				
// 				if (loc != 0) {SVspec[loc-1].insert(pair<qarray<Symmetry::Nq>,ArrayXd>(outbase[loc][qin],SV));}
				
// 				if (TOOL == DMRG::BROOM::BRUTAL_SVD)
// 				{
// 					Nret = min(static_cast<size_t>(SV.rows()), this->max_Nsv);
// 				}
// 				else
// 				{
// 					Nret = (SV > this->eps_svd).count();
// 				}
// 				Nret = max(Nret, this->min_Nsv);
// 				Nret = min(Nret, this->max_Nsv);
// 				Nret = min(Nret, static_cast<size_t>(Jack.singularValues().rows()));
// 				truncWeightSub(qin) = Symmetry::degeneracy(inbase[loc][qin]) * SV.tail(SV.rows()-Nret).cwiseAbs2().sum();
				
// 				// calculate entropy
// 				size_t Nnz = (SV > 0.).count();
// 				entropySub(qin) = -Symmetry::degeneracy(inbase[loc][qin]) * 
// 				                  (SV.head(Nnz).array().square() * SV.head(Nnz).array().square().log()).sum();
// 			}
// 			else if (TOOL == DMRG::BROOM::QR)
// 			{
// 				Quirinus.compute(Aclump.adjoint());
// 				Qmatrix = (Quirinus.householderQ() * MatrixType::Identity(Aclump.cols(),Aclump.rows())).adjoint();
// 				Rmatrix = (MatrixType::Identity(Aclump.rows(),Aclump.cols()) * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
// 			}
			
// 			if (Nret > 0)
// 			{
// 				// update A[loc]
// 				stitch = 0;
// 				for (size_t i=0; i<svec.size(); ++i)
// 				{
// 					MatrixType Mtmp;
					
// 					if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
// 					{
// 						 Mtmp = Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i])*
// 						 		                         Symmetry::coeff_leftSweep(
// 						 		                          A[loc][svec[i]].in[qvec[i]],
// 						 		                          A[loc][svec[i]].out[qvec[i]]);
// //						Mtmp = Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i])*
// //								                         Symmetry::coeff_leftSweep3(
// //								                          A[loc][svec[i]].in[qvec[i]],
// //								                          A[loc][svec[i]].out[qvec[i]],
// //								                          qloc[loc][svec[i]]);
// 					}
// 					else if (TOOL == DMRG::BROOM::QR)
// 					{
// 						 Mtmp = Qmatrix.block(0,stitch, Nrows,Ncolsvec[i])*
// 						 		                         Symmetry::coeff_leftSweep(
// 						 		                          A[loc][svec[i]].in[qvec[i]],
// 						 		                          A[loc][svec[i]].out[qvec[i]]);
// //						Mtmp = Qmatrix.block(0,stitch, Nret,Ncolsvec[i])*
// //								                         Symmetry::coeff_leftSweep3(
// //								                          A[loc][svec[i]].in[qvec[i]],
// //								                          A[loc][svec[i]].out[qvec[i]],
// //								                          qloc[loc][svec[i]]);
// 					}
					
// 					if (Mtmp.size() != 0)
// 					{
// 						Aloc[svec[i]].push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
// 					}
// 					stitch += Ncolsvec[i];
// 				}
				
// 				// update A[loc-1]
// 				if (loc != 0 and DISCARD_U == false)
// 				{
// 					for (size_t s=0; s<qloc[loc-1].size(); ++s)
// 					for (size_t q=0; q<A[loc-1][s].dim; ++q)
// 					{
// 						if (A[loc-1][s].out[q] == inbase[loc][qin])
// 						{
// 							MatrixType Mtmp;
							
// 							if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
// 							{
// 								Mtmp = A[loc-1][s].block[q] * 
// 								       Jack.matrixU().leftCols(Nret) * 
// 								       Jack.singularValues().head(Nret).asDiagonal();
// 							}
// 							else if (TOOL == DMRG::BROOM::QR)
// 							{
// 								Mtmp = A[loc-1][s].block[q] * Rmatrix;
// 							}
							
// 							auto it = Aprev[s].dict.find(qarray2<Nq>{A[loc-1][s].in[q], A[loc-1][s].out[q]});
// 							if (Mtmp.size() != 0)
// 							{
// 								Aprev[s].try_push_back(A[loc-1][s].in[q], A[loc-1][s].out[q], Mtmp);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
	for (size_t s=0; s<qloc[loc].size(); ++s)
	{
		A[loc][s] = Aloc[s].cleaned();
	}
	if (loc != 0 and DISCARD_U == false)
	{
		for (size_t s=0; s<qloc[loc-1].size(); ++s)
		{
			A[loc-1][s] = Aprev[s].cleaned();
		}
	}
	
	update_inbase(loc);
	if (loc != 0 and DISCARD_U == false)
	{
		update_outbase(loc-1);
	}
	
	// if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::RICH_SVD)
	// {
	// 	truncWeight(loc) = truncWeightSub.sum();
	// }
	
	// entropy
	// if (TOOL == DMRG::BROOM::SVD or 
	//    (TOOL == DMRG::BROOM::RICH_SVD and this->alpha_rsvd == 0.))
	// {
	// 	int bond = (loc==0)? -1 : loc;
	// 	if (bond != -1)
	// 	{
	// 		S(loc-1) = entropySub.sum();
	// 	}
	// }
	this->pivot = (loc==0)? 0 : loc-1;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrix1<Symmetry,Scalar,Scalar> *H, bool DISCARD_V)
{
	if (TOOL == DMRG::BROOM::RICH_SVD)
	{
		enrich_right(loc,H);
	}
	
	ArrayXd truncWeightSub(outbase[loc].Nq()); truncWeightSub.setZero();
	ArrayXd entropySub(outbase[loc].Nq()); entropySub.setZero();
	if (loc != this->N_sites-1) {SVspec[loc].clear();}
	map<qarray<Nq>,ArrayXd> SVspec_;
	double entropy;
	
	vector<Biped<Symmetry,MatrixType> > Aloc(qloc[loc].size());
	vector<Biped<Symmetry,MatrixType> > Anext; 
	if (loc != this->N_sites-1 and DISCARD_V == false)
	{
		Anext.resize(qloc[loc+1].size());
	}
	
	Blocker<Symmetry,Scalar> Jim(A[loc],qloc[loc],inbase[loc],outbase[loc]);
	auto Aclump = Jim.Aclump(DMRG::DIRECTION::LEFT);
	Biped<Symmetry,MatrixType> left, right;
	if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::RICH_SVD or TOOL == DMRG::BROOM::BRUTAL_SVD)
	{
		auto [U,Sigma,Vdag] = Aclump.truncateSVD(this->min_Nsv, this->max_Nsv, this->eps_truncWeight, truncWeight(loc), entropy, SVspec_, false); //false: DONT PRESERVE MULTIPLETS
		if (loc != this->N_sites-1)
		{
			S(loc) = entropy;
			SVspec[loc] = SVspec_;
		}
		left = U;
		right = Sigma.contract(Vdag);
	}
	else if (TOOL == DMRG::BROOM::QR)
	{
		auto [Q,R] = Aclump.QR();
		left = Q;
		right = R;
	}
	
	// cout << "loc=" << loc << ", entropy in rightSweepStep: " << S(loc) << endl;
	Aloc = Jim.reblock(left, DMRG::DIRECTION::LEFT);
	if (loc != this->N_sites-1 and DISCARD_V == false)
	{
		for (size_t s=0; s<qloc[loc+1].size(); ++s)
		for (size_t q=0; q<A[loc+1][s].dim; ++q)
		{
			MatrixType Mtmp;
			auto itright = right.dict.find({A[loc+1][s].in[q],A[loc+1][s].in[q]});
			if (itright != right.dict.end())
			{
				Mtmp = right.block[itright->second] * A[loc+1][s].block[q];
				auto it = Anext[s].dict.find(qarray2<Nq>{A[loc+1][s].in[q], A[loc+1][s].out[q]});
				if (Mtmp.size() != 0)
				{
					Anext[s].try_push_back(A[loc+1][s].in[q], A[loc+1][s].out[q], Mtmp);
				}
			}
		}
	}
	
	// #ifndef DMRG_DONT_USE_OPENMP
	// #pragma omp parallel for
	// #endif
	// for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
	// {
	// 	// determine how many A's to glue together
	// 	vector<size_t> svec, qvec, Nrowsvec;
	// 	for (size_t s=0; s<qloc[loc].size(); ++s)
	// 	for (size_t q=0; q<A[loc][s].dim; ++q)
	// 	{
	// 		if (A[loc][s].out[q] == outbase[loc][qout] and
	// 		    A[loc][s].block[q].rows() > 0 and
	// 		    A[loc][s].block[q].cols() > 0)
	// 		{
	// 			svec.push_back(s);
	// 			qvec.push_back(q);
	// 			Nrowsvec.push_back(A[loc][s].block[q].rows());
	// 		}
	// 	}
		
	// 	if (Nrowsvec.size() > 0)
	// 	{
	// 		// do the glue
	// 		size_t Ncols = A[loc][svec[0]].block[qvec[0]].cols();
	// 		for (size_t i=1; i<svec.size(); ++i) {assert(A[loc][svec[i]].block[qvec[i]].cols() == Ncols);}
	// 		size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
			
	// 		MatrixType Aclump(Nrows,Ncols);
	// 		Aclump.setZero();
	// 		size_t stitch = 0;
	// 		for (size_t i=0; i<svec.size(); ++i)
	// 		{
	// 			Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[loc][svec[i]].block[qvec[i]];
	// 			stitch += Nrowsvec[i];
	// 		}
			
	// 		#ifdef DONT_USE_BDCSVD
	// 		JacobiSVD<MatrixType> Jack; // standard SVD
	// 		#else
	// 		BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only in Eigen available)
	// 		#endif
			
	// 		HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // Eigen QR
			
	// 		size_t Nret = Ncols; // retained states
			
	// 		if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
	// 		{
	// 			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
	// 			ArrayXd SV = Jack.singularValues();
	// 			// sqrt(Symmetry::degeneracy(outbase[loc][qout]))
				
	// 			SVspec[loc].insert(pair<qarray<Symmetry::Nq>,ArrayXd>(outbase[loc][qout],SV));
				
	// 			if (TOOL == DMRG::BROOM::BRUTAL_SVD)
	// 			{
	// 				Nret = min(static_cast<size_t>(SV.rows()), this->max_Nsv);
	// 			}
	// 			else
	// 			{
	// 				Nret = (SV > this->eps_svd).count();
	// 			}
	// 			Nret = max(Nret, this->min_Nsv);
	// 			Nret = min(Nret, this->max_Nsv);
	// 			Nret = min(Nret, static_cast<size_t>(Jack.singularValues().rows()));
	// 			truncWeightSub(qout) = Symmetry::degeneracy(outbase[loc][qout]) * SV.tail(SV.rows()-Nret).cwiseAbs2().sum();
				
	// 			// calculate entropy
	// 			size_t Nnz = (SV > 0.).count();
	// 			entropySub(qout) = -Symmetry::degeneracy(outbase[loc][qout]) * 
	// 			                   (SV.head(Nnz).array().square() * SV.head(Nnz).array().square().log()).sum();
	// 		}
	// 		else if (TOOL == DMRG::BROOM::QR)
	// 		{
	// 			Quirinus.compute(Aclump);
	// 			Qmatrix = Quirinus.householderQ() * MatrixType::Identity(Aclump.rows(),Aclump.cols());
	// 			Rmatrix = MatrixType::Identity(Aclump.cols(),Aclump.rows()) * Quirinus.matrixQR().template triangularView<Upper>();
	// 		}
			
	// 		if (Nret > 0)
	// 		{
	// 			// update A[loc]
	// 			stitch = 0;
	// 			for (size_t i=0; i<svec.size(); ++i)
	// 			{
	// 				MatrixType Mtmp;
	// 				if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
	// 				{
	// 					// A[loc][svec[i]].block[qvec[i]]
	// 					Mtmp = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret);
	// 				}
	// 				else if (TOOL == DMRG::BROOM::QR)
	// 				{
	// 					Mtmp = Qmatrix.block(stitch,0, Nrowsvec[i],Ncols);
	// 				}
					
	// 				if (Mtmp.size() != 0)
	// 				{
	// 					Aloc[svec[i]].push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
	// 				}
	// 				stitch += Nrowsvec[i];
	// 			}
				
	// 			// update A[loc+1]
	// 			if (loc != this->N_sites-1 and DISCARD_V == false)
	// 			{
	// 				for (size_t s=0; s<qloc[loc+1].size(); ++s)
	// 				for (size_t q=0; q<A[loc+1][s].dim; ++q)
	// 				{
	// 					if (A[loc+1][s].in[q] == outbase[loc][qout])
	// 					{
	// 						MatrixType Mtmp;
							
	// 						if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
	// 						{
	// 							Mtmp = Jack.singularValues().head(Nret).asDiagonal() * 
	// 								                   Jack.matrixV().adjoint().topRows(Nret) * 
	// 								                   A[loc+1][s].block[q];
	// 						}
	// 						else if (TOOL == DMRG::BROOM::QR)
	// 						{
	// 							Mtmp = Rmatrix * A[loc+1][s].block[q];
	// 						}
							
	// 						auto it = Anext[s].dict.find(qarray2<Nq>{A[loc+1][s].in[q], A[loc+1][s].out[q]});
	// 						if (Mtmp.size() != 0)
	// 						{
	// 							Anext[s].try_push_back(A[loc+1][s].in[q], A[loc+1][s].out[q], Mtmp);
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// }
	
	for (size_t s=0; s<qloc[loc].size(); ++s)
	{
		A[loc][s] = Aloc[s].cleaned();
	}
	if (loc != this->N_sites-1 and DISCARD_V == false)
	{
		for (size_t s=0; s<qloc[loc+1].size(); ++s)
		{
			A[loc+1][s] = Anext[s].cleaned();
		}
	}
	
	update_outbase(loc);
	if (loc != this->N_sites-1 and DISCARD_V == false)
	{
		update_inbase(loc+1);
	}
	
	// if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::RICH_SVD)
	// {
	// 	truncWeight(loc) = truncWeightSub.sum();
	// }
	
	// entropy
	// if (TOOL == DMRG::BROOM::SVD or 
	//    (TOOL == DMRG::BROOM::RICH_SVD and this->alpha_rsvd == 0.))
	// {
	// 	int bond = (loc==this->N_sites-1)? -1 : loc;
	// 	if (bond != -1)
	// 	{
	// 		S(loc) = entropySub.sum();
	// 	}
	// }
	this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,MatrixType> > &N)
{
	N.clear();
	N.resize(qloc[loc].size());
	
	if (DIR == DMRG::DIRECTION::LEFT)
	{
		for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
		{
			// determine how many A's to glue together
			vector<size_t> svec, qvec, Ncolsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[loc][s].dim; ++q)
			{
				if (A[loc][s].in[q] == inbase[loc][qin])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Ncolsvec.push_back(A[loc][s].block[q].cols());
				}
			}
			
			if (Ncolsvec.size() > 0)
			{
				// do the glue
				size_t Nrows = A[loc][svec[0]].block[qvec[0]].rows();
				for (size_t i=1; i<svec.size(); ++i) {assert(A[loc][svec[i]].block[qvec[i]].rows() == Nrows);}
				size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
				
				MatrixType Aclump(Nrows,Ncols);
				size_t stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[loc][svec[i]].block[qvec[i]]*
					                                            Symmetry::coeff_leftSweep(
					                                            A[loc][svec[i]].out[qvec[i]],
					                                            A[loc][svec[i]].in[qvec[i]]);
					stitch += Ncolsvec[i];
				}
				
				HouseholderQR<MatrixType> Quirinus(Aclump.adjoint());
				MatrixType Qmatrix = Quirinus.householderQ().adjoint();
				size_t Nret = Nrows; // retained states
				
				// fill N
				stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					if (Qmatrix.rows() > Nret)
					{
						size_t Nnull = Qmatrix.rows()-Nret;
						MatrixType Mtmp = Qmatrix.block(Nret,stitch, Nnull,Ncolsvec[i])*
						                  Symmetry::coeff_leftSweep(
						                  A[loc][svec[i]].in[qvec[i]],
						                  A[loc][svec[i]].out[qvec[i]]);
						N[svec[i]].try_push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Ncolsvec[i];
				}
			}
		}
	}
	else if (DIR == DMRG::DIRECTION::RIGHT)
	{
		for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
		{
			// determine how many A's to glue together
			vector<size_t> svec, qvec, Nrowsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[loc][s].dim; ++q)
			{
				if (A[loc][s].out[q] == outbase[loc][qout])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Nrowsvec.push_back(A[loc][s].block[q].rows());
				}
			}
			
			if (Nrowsvec.size() > 0)
			{
				// do the glue
				size_t Ncols = A[loc][svec[0]].block[qvec[0]].cols();
				for (size_t i=1; i<svec.size(); ++i) {assert(A[loc][svec[i]].block[qvec[i]].cols() == Ncols);}
				size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
				
				MatrixType Aclump(Nrows,Ncols);
				Aclump.setZero();
				size_t stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[loc][svec[i]].block[qvec[i]];
					stitch += Nrowsvec[i];
				}
				
				HouseholderQR<MatrixType> Quirinus(Aclump);
				MatrixType Qmatrix = Quirinus.householderQ();
				size_t Nret = Ncols; // retained states
				
				// fill N
				stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					if (Qmatrix.cols() > Nret)
					{
						size_t Nnull = Qmatrix.cols()-Nret;
//							N[loc][svec[i]].block[qvec[i]] = Qmatrix.block(stitch,Nret, Nrowsvec[i],Nnull);
						MatrixType Mtmp = Qmatrix.block(stitch,Nret, Nrowsvec[i],Nnull);
						N[svec[i]].try_push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Nrowsvec[i];
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
leftSplitStep (size_t loc, Biped<Symmetry,MatrixType> &C)
{
//	#ifndef DMRG_DONT_USE_OPENMP
//	#pragma omp parallel for
//	#endif
	for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
	{
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Ncolsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[loc][s].dim; ++q)
		{
			if (A[loc][s].in[q] == inbase[loc][qin])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Ncolsvec.push_back(A[loc][s].block[q].cols());
			}
		}
		
		if (svec.size() > 0)
		{
			// do the glue
			size_t Nrows = A[loc][svec[0]].block[qvec[0]].rows();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[loc][svec[i]].block[qvec[i]].rows() == Nrows);}
			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
			MatrixType Aclump(Nrows,Ncols);
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[loc][svec[i]].block[qvec[i]] *
					                                        Symmetry::coeff_leftSweep(A[loc][svec[i]].out[qvec[i]],
					                                                                  A[loc][svec[i]].in[qvec[i]]);
				stitch += Ncolsvec[i];
			}
			
			HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // Eigen QR
			
			Quirinus.compute(Aclump.adjoint());
			Qmatrix = (Quirinus.householderQ() * MatrixType::Identity(Aclump.cols(),Aclump.rows())).adjoint();
			Rmatrix = (MatrixType::Identity(Aclump.rows(),Aclump.cols()) * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
			
			// update A[loc]
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[loc][svec[i]].block[qvec[i]] = Qmatrix.block(0,stitch, Nrows,Ncolsvec[i])*
				                                 Symmetry::coeff_leftSweep(A[loc][svec[i]].in[qvec[i]],
				                                                           A[loc][svec[i]].out[qvec[i]]);
				stitch += Ncolsvec[i];
			}
			
			// write to C
			qarray2<Nq> quple = {inbase[loc][qin], inbase[loc][qin]};
			auto qC = C.dict.find(quple);
			
			if (qC != C.dict.end())
			{
				C.block[qC->second] += Rmatrix;
			}
			else
			{
				C.push_back(quple,Rmatrix);
			}
		}
	}
	
	this->pivot = (loc==0)? 0 : loc-1;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
rightSplitStep (size_t loc, Biped<Symmetry,MatrixType> &C)
{
//	#ifndef DMRG_DONT_USE_OPENMP
//	#pragma omp parallel for
//	#endif
	for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
	{
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[loc][s].dim; ++q)
		{
			if (A[loc][s].out[q] == outbase[loc][qout])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Nrowsvec.push_back(A[loc][s].block[q].rows());
			}
		}
		
		if (svec.size() > 0)
		{
			// do the glue
			size_t Ncols = A[loc][svec[0]].block[qvec[0]].cols();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[loc][svec[i]].block[qvec[i]].cols() == Ncols);}
			size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
			
			MatrixType Aclump(Nrows,Ncols);
			Aclump.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[loc][svec[i]].block[qvec[i]];
				stitch += Nrowsvec[i];
			}
			
			HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // Eigen QR
			
			Quirinus.compute(Aclump);
			Qmatrix = Quirinus.householderQ() * MatrixType::Identity(Aclump.rows(),Aclump.cols());
			Rmatrix = MatrixType::Identity(Aclump.cols(),Aclump.rows()) * Quirinus.matrixQR().template triangularView<Upper>();
			
			// update A[loc]
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[loc][svec[i]].block[qvec[i]] = Qmatrix.block(stitch,0, Nrowsvec[i],Ncols);
				stitch += Nrowsvec[i];
			}
			
			// write to C
			qarray2<Nq> quple = {outbase[loc][qout], outbase[loc][qout]};
			auto qC = C.dict.find(quple);
			
			if (qC != C.dict.end())
			{
				C.block[qC->second] += Rmatrix;
			}
			else
			{
				C.push_back(quple,Rmatrix);
			}
		}
	}
	
	this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
absorb (size_t loc, DMRG::DIRECTION::OPTION DIR, const Biped<Symmetry,MatrixType> &C)
{
	for (size_t qC=0; qC<C.dim; ++qC)
	{
		if (DIR == DMRG::DIRECTION::RIGHT)
		{
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[loc][s].dim; ++q)
			{
				if (A[loc][s].in[q] == C.out[qC])
				{
					A[loc][s].block[q] = C.block[qC] * A[loc][s].block[q];
				}
			}
		}
		else
		{
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[loc][s].dim; ++q)
			{
				if (A[loc][s].out[q] == C.in[qC])
				{
					A[loc][s].block[q] = A[loc][s].block[q] * C.block[qC];
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
sweepStep2 (DMRG::DIRECTION::OPTION DIR, size_t loc, const vector<Biped<Symmetry,MatrixType> > &Apair, bool SEPARATE_SV)
{
	Biped<Symmetry,MatrixType> Cdump;
	double entropy;
	map<qarray<Nq>,ArrayXd> SV;
	
	Qbasis<Symmetry> qloc_l, qloc_r;
	qloc_l.pullData(locBasis(loc)); qloc_r.pullData(locBasis((loc+1)));
	auto combined_basis = qloc_l.combine(qloc_r);
	
	//cout << "begin splitAA" << endl;
	split_AA2(DIR, combined_basis, Apair, qloc[loc], A[loc], qloc[loc+1], A[loc+1],
			  QoutTop[loc], QoutBot[loc],
			  Cdump, false, truncWeight(loc), entropy, SV,
			  this->eps_truncWeight, this->min_Nsv, this->max_Nsv);
	//cout << "end splitAA" << endl;
	
	// Warning: uses eps_svd
	// split_AA(DIR, Apair, qloc[loc], A[loc], qloc[loc+1], A[loc+1],
	//          QoutTop[loc], QoutBot[loc],
	//          Cdump, false, truncWeight(loc), entropy,
	//          this->eps_svd, this->min_Nsv, this->max_Nsv);
	update_outbase(loc);
	update_inbase(loc+1);
	
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
	}
	else
	{
		this->pivot = (loc==0)? 0 : loc;
	}
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		int bond = (loc==this->N_sites-1)? -1 : loc;
		if (bond != -1)
		{
			S(loc) = entropy;
			SVspec[loc] = SV;
		}
	}
	else
	{
		int bond = (loc==0)? -1 : loc;
		if (bond != -1)
		{
			S(loc-1) = entropy;
			SVspec[loc-1] = SV;
		}
	}
}

// template<typename Symmetry, typename Scalar>
// void Mps<Symmetry,Scalar>::
// sweepStep2 (DMRG::DIRECTION::OPTION DIR, size_t loc, const vector<Biped<Symmetry,MatrixType> > &Apair, 
//             vector<Biped<Symmetry,MatrixType> > &Al, vector<Biped<Symmetry,MatrixType> > &Ar, Biped<Symmetry,MatrixType> &C, 
//             bool SEPARATE_SV)
// {
// 	vector<qarray<Symmetry::Nq> > midset = calc_qsplit(Al, qloc[loc], Ar, qloc[loc+1], QoutTop[loc], QoutBot[loc]);
	
// 	for (size_t s=0; s<qloc[loc].size(); ++s)
// 	{
// 		Al[s].clear();
// 	}
// 	for (size_t s=0; s<qloc[loc+1].size(); ++s)
// 	{
// 		Ar[s].clear();
// 	}
	
// 	ArrayXd truncWeightSub(midset.size()); truncWeightSub.setZero();
// 	ArrayXd entropySub(midset.size()); entropySub.setZero();
	
// 	auto tensor_basis = Symmetry::tensorProd(qloc[loc], qloc[loc+1]);
	
// 	#ifndef DMRG_DONT_USE_OPENMP
// 	#pragma omp parallel for
// 	#endif
// 	for (size_t qmid=0; qmid<midset.size(); ++qmid)
// 	{
// 		map<pair<size_t,qarray<Symmetry::Nq> >,vector<pair<size_t,qarray<Symmetry::Nq> > > > s13map;
// 		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<Scalar> > cgcmap;
// 		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<size_t> > q13map;
// 		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<size_t> > s1s3map;
		
// 		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
// 		for (size_t s3=0; s3<qloc[loc+1].size(); ++s3)
// 		{
// 			auto qmerges = Symmetry::reduceSilent(qloc[loc][s1], qloc[loc+1][s3]);
			
// 			for (const auto &qmerge:qmerges)
// 			{
// 				auto qtensor = make_tuple(qloc[loc][s1], s1, qloc[loc+1][s3], s3, qmerge);
// 				auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
				
// 				for (size_t q13=0; q13<Apair[s1s3].dim; ++q13)
// 				{
// 					auto qlmids = Symmetry::reduceSilent(Apair[s1s3].in[q13], qloc[loc][s1]);
// 					auto qrmids = Symmetry::reduceSilent(Apair[s1s3].out[q13], Symmetry::flip(qloc[loc+1][s3]));
					
// 					for (const auto &qlmid:qlmids)
// 					for (const auto &qrmid:qrmids)
// 					{
// 						if (qlmid == midset[qmid] and qrmid == midset[qmid])
// 						{
// 							s13map[make_pair(s1,Apair[s1s3].in[q13])].push_back(make_pair(s3,Apair[s1s3].out[q13]));
							
// 							Scalar factor_cgc = Symmetry::coeff_Apair(Apair[s1s3].in[q13], qloc[loc][s1], midset[qmid], 
// 							                                          qloc[loc+1][s3], Apair[s1s3].out[q13], qmerge);
// 							if (DIR==DMRG::DIRECTION::LEFT)
// 							{
// 								factor_cgc *= sqrt(Symmetry::coeff_rightOrtho(Apair[s1s3].out[q13], midset[qmid]));
// 							}
							
// 							cgcmap[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(factor_cgc);
// 							q13map[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(q13);
// 							s1s3map[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(s1s3);
// 						}
// 					}
// 				}
// 			}
// 		}
		
// 		if (s13map.size() != 0)
// 		{
// 			map<pair<size_t,qarray<Symmetry::Nq> >,MatrixType> Aclumpvec;
// 			size_t istitch = 0;
// 			size_t jstitch = 0;
// 			vector<size_t> get_s3;
// 			vector<size_t> get_Ncols;
// 			vector<qarray<Symmetry::Nq> > get_qr;
// 			bool COLS_ARE_KNOWN = false;
			
// 			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
// 			{
// 				auto qls = Symmetry::reduceSilent(midset[qmid], Symmetry::flip(qloc[loc][s1]));
				
// 				for (const auto &ql:qls)
// 				{
// 					for (size_t s3=0; s3<qloc[loc+1].size(); ++s3)
// 					{
// 						auto qrs = Symmetry::reduceSilent(midset[qmid], qloc[loc+1][s3]);
						
// 						for (const auto &qr:qrs)
// 						{
// 							auto s3block = find(s13map[make_pair(s1,ql)].begin(), s13map[make_pair(s1,ql)].end(), make_pair(s3,qr));
							
// 							if (s3block != s13map[make_pair(s1,ql)].end())
// 							{
// 								MatrixType Mtmp;
// 								for (size_t i=0; i<q13map[make_tuple(s1,ql,s3,qr)].size(); ++i)
// 								{
// 									size_t q13 = q13map[make_tuple(s1,ql,s3,qr)][i];
// 									size_t s1s3 = s1s3map[make_tuple(s1,ql,s3,qr)][i];
									
// 									if (Mtmp.size() == 0)
// 									{
// 										Mtmp = cgcmap[make_tuple(s1,ql,s3,qr)][i] * Apair[s1s3].block[q13];
// 									}
// 									else if (Mtmp.size() > 0 and Apair[s1s3].block[q13].size() > 0)
// 									{
// 										Mtmp += cgcmap[make_tuple(s1,ql,s3,qr)][i] * Apair[s1s3].block[q13];
// 									}
// 								}
// 								if (Mtmp.size() == 0) {continue;}
								
// 								addRight(Mtmp, Aclumpvec[make_pair(s1,ql)]);
								
// 								if (COLS_ARE_KNOWN == false)
// 								{
// 									get_s3.push_back(s3);
// 									get_Ncols.push_back(Mtmp.cols());
// 									get_qr.push_back(qr);
// 								}
// 							}
// 						}
// 					}
// 					if (get_s3.size() != 0) {COLS_ARE_KNOWN = true;}
// 				}
// 			}
			
// 			vector<size_t> get_s1;
// 			vector<size_t> get_Nrows;
// 			vector<qarray<Symmetry::Nq> > get_ql;
// 			MatrixType Aclump;
// 			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
// 			{
// 				auto qls = Symmetry::reduceSilent(midset[qmid], Symmetry::flip(qloc[loc][s1]));
				
// 				for (const auto &ql:qls)
// 				{
// 					size_t Aclump_rows_old = Aclump.rows();
					
// 					// If cols don't match, it means that zeros were cut, restore them 
// 					// (happens in MpsCompressor::polyCompress):
// 					if (Aclumpvec[make_pair(s1,ql)].cols() < Aclump.cols())
// 					{
// 						size_t dcols = Aclump.cols() - Aclumpvec[make_pair(s1,ql)].cols();
// 						Aclumpvec[make_pair(s1,ql)].conservativeResize(Aclumpvec[make_pair(s1,ql)].rows(), Aclump.cols());
// 						Aclumpvec[make_pair(s1,ql)].rightCols(dcols).setZero();
// 					}
// 					else if (Aclumpvec[make_pair(s1,ql)].cols() > Aclump.cols())
// 					{
// 						size_t dcols = Aclumpvec[make_pair(s1,ql)].cols() - Aclump.cols();
// 						Aclump.conservativeResize(Aclump.rows(), Aclump.cols()+dcols);
// 						Aclump.rightCols(dcols).setZero();
// 					}
					
// 					addBottom(Aclumpvec[make_pair(s1,ql)], Aclump);
					
// 					if (Aclump.rows() > Aclump_rows_old)
// 					{
// 						get_s1.push_back(s1);
// 						get_Nrows.push_back(Aclump.rows()-Aclump_rows_old);
// 						get_ql.push_back(ql);
// 					}
// 				}
// 			}
// 			if (Aclump.size() == 0)
// 			{
// //				if (DIR == DMRG::DIRECTION::RIGHT)
// //				{
// //					this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
// //				}
// //				else
// //				{
// //					this->pivot = (loc==0)? 0 : loc;
// //				}
// 				continue;
// 			}
			
// 			#ifdef DONT_USE_BDCSVD
// 			JacobiSVD<MatrixType> Jack; // standard SVD
// 			#else
// 			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
// 			#endif
// 			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
// 			VectorXd SV = Jack.singularValues();
			
// 			// retained states:
// 			size_t Nret = (SV.array().abs() > this->eps_svd).count();
// 			Nret = max(Nret, this->min_Nsv);
// 			Nret = min(Nret, this->max_Nsv);
// 			truncWeightSub(qmid) = Symmetry::degeneracy(midset[qmid]) * SV.tail(SV.rows()-Nret).cwiseAbs2().sum();
// 			size_t Nnz = (Jack.singularValues().array() > 0.).count();
// 			entropySub(qmid) = -Symmetry::degeneracy(midset[qmid]) *
//                  			   (SV.head(Nnz).array().square() * SV.head(Nnz).array().square().log()).sum();
			
// 			MatrixType Aleft, Aright, Cmatrix;
// 			if (DIR == DMRG::DIRECTION::RIGHT)
// 			{
// 				Aleft = Jack.matrixU().leftCols(Nret);
// 				if (SEPARATE_SV)
// 				{
// 					Aright = Jack.matrixV().adjoint().topRows(Nret);
// 					Cmatrix = Jack.singularValues().head(Nret).asDiagonal();
// 				}
// 				else
// 				{
// 					Aright = Jack.singularValues().head(Nret).asDiagonal() * Jack.matrixV().adjoint().topRows(Nret);
// 				}
// //				this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
// 			}
// 			else
// 			{
// 				Aright = Jack.matrixV().adjoint().topRows(Nret);
// 				if (SEPARATE_SV)
// 				{
// 					Aleft = Jack.matrixU().leftCols(Nret);
// 					Cmatrix = Jack.singularValues().head(Nret).asDiagonal();
// 				}
// 				else
// 				{
// 					Aleft = Jack.matrixU().leftCols(Nret) * Jack.singularValues().head(Nret).asDiagonal();
// 				}
// //				this->pivot = (loc==0)? 0 : loc;
// 			}
			
// 			// update Al
// 			istitch = 0;
// 			for (size_t i=0; i<get_s1.size(); ++i)
// 			{
// 				size_t s1 = get_s1[i];
// 				size_t Nrows = get_Nrows[i];
				
// 				qarray2<Nq> quple = {get_ql[i], midset[qmid]};
// 				auto q = Al[s1].dict.find(quple);
// 				if (q != Al[s1].dict.end())
// 				{
// 					Al[s1].block[q->second] += Aleft.block(istitch,0, Nrows,Nret);
// 				}
// 				else
// 				{
// 					Al[s1].push_back(get_ql[i], midset[qmid], Aleft.block(istitch,0, Nrows,Nret));
// 				}
// 				istitch += Nrows;
// 			}
			
// 			// update Ar
// 			jstitch = 0;
// 			for (size_t i=0; i<get_s3.size(); ++i)
// 			{
// 				size_t s3 = get_s3[i];
// 				size_t Ncols = get_Ncols[i];
				
// 				qarray2<Nq> quple = {midset[qmid], get_qr[i]};
// 				auto q = Ar[s3].dict.find(quple);
// 				Scalar factor_cgc3 = (DIR==DMRG::DIRECTION::LEFT)? sqrt(Symmetry::coeff_rightOrtho(midset[qmid], get_qr[i])):1.;
// 				if (q != Ar[s3].dict.end())
// 				{
// 					Ar[s3].block[q->second] += factor_cgc3 * Aright.block(0,jstitch, Nret,Ncols);
// 				}
// 				else
// 				{
// 					Ar[s3].push_back(midset[qmid], get_qr[i], factor_cgc3 * Aright.block(0,jstitch, Nret,Ncols));
// 				}
// 				jstitch += Ncols;
// 			}
			
// 			if (SEPARATE_SV)
// 			{
// 				qarray2<Nq> quple = {midset[qmid], midset[qmid]};
// 				auto q = C.dict.find(quple);
// 				if (q != C.dict.end())
// 				{
// 					C.block[q->second] += Cmatrix;
// 				}
// 				else
// 				{
// 					C.push_back(midset[qmid], midset[qmid], Cmatrix);
// 				}
// 			}
// 		}
// 	}
	
// 	// remove unwanted zero-sized blocks
// 	for (size_t s=0; s<qloc[loc].size(); ++s)
// 	{
// 		Al[s] = Al[s].cleaned();
// 	}
// 	for (size_t s=0; s<qloc[loc+1].size(); ++s)
// 	{
// 		Ar[s] = Ar[s].cleaned();
// 	}
	
// 	truncWeight(loc) = truncWeightSub.sum();
	
// 	if (DIR == DMRG::DIRECTION::RIGHT)
// 	{
// 		int bond = (loc==this->N_sites-1)? -1 : loc;
// 		if (bond != -1)
// 		{
// 			S(loc) = entropySub.sum();
// 		}
// 	}
// 	else
// 	{
// 		int bond = (loc==0)? -1 : loc;
// 		if (bond != -1)
// 		{
// 			S(loc-1) = entropySub.sum();
// 		}
// 	}
// }

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
enrich_left (size_t loc, PivotMatrix1<Symmetry,Scalar,Scalar> *H)
{
	if (this->alpha_rsvd > mynumeric_limits<Scalar>::epsilon())
	{
		std::vector<Biped<Symmetry,MatrixType> > P(qloc[loc].size());
		
//		Qbasis<Symmetry> QbasisW;
//		QbasisW.pullData(H->W,0);
//		auto QbasisP = inbase[loc].combine(QbasisW);
//		
//		// create tensor P
//		#ifndef DMRG_DONT_USE_OPENMP
//		#pragma omp parallel for
//		#endif
//		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
//		for (size_t s2=0; s2<qloc[loc].size(); ++s2)
//		for (size_t k=0; k<H->qOp.size(); ++k)
//		{
//			if (H->W[s1][s2][k].size() == 0) {continue;}
//			for (size_t qR=0; qR<H->R.size(); ++qR)
//			{
//				auto qAs = Symmetry::reduceSilent(H->R.in(qR),Symmetry::flip(qloc[loc][s2]));
//				for (const auto& qA : qAs)
//				{
//					qarray2<Symmetry::Nq> quple1 = {qA, H->R.in(qR)};
//					auto itA = A[loc][s2].dict.find(quple1);
//					
//					if (itA != A[loc][s2].dict.end())
//					{
//						auto qWs = Symmetry::reduceSilent(H->R.mid(qR), Symmetry::flip(H->qOp[k]));
//						
//						for (const auto& qW : qWs)
//						{
//							auto qPs = Symmetry::reduceSilent(qA,qW);
//							
//							for (const auto& qP : qPs)
//							{
//								if (qP > QinTop[loc] or qP < QinBot[loc]) {continue;}
//								
//								Scalar factor_cgc = Symmetry::coeff_HPsi(A[loc][s2].in[itA->second], qloc[loc][s2], A[loc][s2].out[itA->second],
//								                                         qW, H->qOp[k], H->R.mid(qR),
//								                                         qP, qloc[loc][s1], H->R.out(qR));
//								
//								if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
//								
//								auto dict_entry = H->W[s1][s2][k].dict.find({qW,H->R.mid(qR)});
//								if(dict_entry == H->W[s1][s2][k].dict.end()) continue;
//								for (int spInd=0; spInd<H->W[s1][s2][k].block[dict_entry->second].outerSize(); ++spInd)
//								for (typename SparseMatrix<Scalar>::InnerIterator iW(H->W[s1][s2][k].block[dict_entry->second],spInd); iW; ++iW)
//								{
//									size_t a = iW.row();
//									size_t b = iW.col();
//									size_t Prows = QbasisP.inner_dim(qP);
//									if(Prows==0) { continue;}
//									size_t Pcols = H->R.block[qR][b][0].cols();
//									if(Pcols==0) { continue;}
//									size_t Arows = A[loc][s2].block[itA->second].rows();
//									size_t stitch = QbasisP.leftAmount(qP,{qA,qW});
//									
//									MatrixType Mtmp(Prows,Pcols);
//									Mtmp.setZero();
//									
//									if (stitch >= Prows) {continue;}
//									if (H->R.block[qR][b][0].size() != 0)
//									{
//										Mtmp.block(stitch + a*Arows,0, Arows,Pcols) += (this->alpha_rsvd * 
//										                                                factor_cgc * 
//										                                                iW.value()) * 
//										                                                A[loc][s2].block[itA->second] * 
//										                                                H->R.block[qR][b][0];
//									}
//									
//									
//									// VectorXd norms = Mtmp.rowwise().norm();
////									vector<int> indices(Mtmp.rows());
////									iota(indices.begin(), indices.end(), 0);
//									// sort(indices.begin(), indices.end(), [norms](int i, int j){return norms(i) > norms(j);});
//									
//									// int Nret = min(static_cast<int>(0.1*Prows),20);
//									// Nret = max(Nret,1);
//									// Nret = min(Mtmp.rows(), Nret);
//									int Nret = (this->max_Nrich<0)? Mtmp.rows():
//									                                min(static_cast<int>(Mtmp.rows()), this->max_Nrich);
//									
////									MatrixType Mret(Nret,Mtmp.cols());
////									for (int i=0; i<Nret; ++i)
////									{
////										Mret.row(i) = Mtmp.row(indices[i]);
////									}
//									if( Nret < Mtmp.rows() ) { Mtmp = Mtmp.topRows(Nret).eval(); }
//									if (Mtmp.size() != 0)
//									{
//										qarray2<Symmetry::Nq> qupleP = {qP, H->R.out(qR)};
//										auto it = P[s1].dict.find(qupleP);
//										if (it != P[s1].dict.end())
//										{
//											if (P[s1].block[it->second].rows() == 0)
//											{
//												P[s1].block[it->second] = Mtmp;
//											}
//											else
//											{
//												P[s1].block[it->second] += Mtmp;
//											}
//										}
//										else
//										{
//											P[s1].push_back(qupleP, Mtmp);
//										}
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
		
		vector<vector<Biped<Symmetry,MatrixType> > > Pt(H->Terms.size());
		for (size_t t=0; t<H->Terms.size(); ++t)
		{
			Pt[t].resize(qloc[loc].size());
		}
		
		for (size_t t=0; t<H->Terms.size(); ++t)
		{
			Qbasis<Symmetry> QbasisW;
			QbasisW.pullData(H->Terms[t].W,0);
			auto QbasisP = inbase[loc].combine(QbasisW);
			
			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
			for (size_t s2=0; s2<qloc[loc].size(); ++s2)
			for (size_t k=0; k<H->Terms[t].qOp.size(); ++k)
			{
				if (H->Terms[t].W[s1][s2][k].size() == 0) {continue;}
				for (size_t qR=0; qR<H->Terms[t].R.size(); ++qR)
				{
					auto qAs = Symmetry::reduceSilent(H->Terms[t].R.in(qR),Symmetry::flip(qloc[loc][s2]));
					for (const auto& qA : qAs)
					{
						qarray2<Symmetry::Nq> quple1 = {qA, H->Terms[t].R.in(qR)};
						auto itA = A[loc][s2].dict.find(quple1);
						
						if (itA != A[loc][s2].dict.end())
						{
							auto qWs = Symmetry::reduceSilent(H->Terms[t].R.mid(qR), Symmetry::flip(H->Terms[t].qOp[k]));
							
							for (const auto& qW : qWs)
							{
								auto qPs = Symmetry::reduceSilent(qA,qW);
								
								for (const auto& qP : qPs)
								{
									if (qP > QinTop[loc] or qP < QinBot[loc]) {continue;}
									
									Scalar factor_cgc = Symmetry::coeff_HPsi(A[loc][s2].in[itA->second], qloc[loc][s2], A[loc][s2].out[itA->second],
										                                     qW, H->Terms[t].qOp[k], H->Terms[t].R.mid(qR),
										                                     qP, qloc[loc][s1], H->Terms[t].R.out(qR));
									
									if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
									
									auto dict_entry = H->Terms[t].W[s1][s2][k].dict.find({qW,H->Terms[t].R.mid(qR)});
									if(dict_entry == H->Terms[t].W[s1][s2][k].dict.end()) continue;
									for (int spInd=0; spInd<H->Terms[t].W[s1][s2][k].block[dict_entry->second].outerSize(); ++spInd)
									for (typename SparseMatrix<Scalar>::InnerIterator iW(H->Terms[t].W[s1][s2][k].block[dict_entry->second],spInd); iW; ++iW)
									{
										size_t a = iW.row();
										size_t b = iW.col();
										size_t Prows = QbasisP.inner_dim(qP);
										if(Prows==0) { continue;}
										size_t Pcols = H->Terms[t].R.block[qR][b][0].cols();
										if(Pcols==0) { continue;}
										size_t Arows = A[loc][s2].block[itA->second].rows();
										size_t stitch = QbasisP.leftAmount(qP,{qA,qW});
										
										MatrixType Mtmp(Prows,Pcols);
										Mtmp.setZero();
										
										if (stitch >= Prows) {continue;}
										if (H->Terms[t].R.block[qR][b][0].size() != 0)
										{
											Mtmp.block(stitch + a*Arows,0, Arows,Pcols) += (this->alpha_rsvd * 
												                                            factor_cgc * 
												                                            iW.value()) * 
												                                            A[loc][s2].block[itA->second] * 
												                                            H->Terms[t].R.block[qR][b][0];
										}
										
										int Nret = (this->max_Nrich<0)? Mtmp.rows():
											                            min(static_cast<int>(Mtmp.rows()), this->max_Nrich);
										
										if( Nret < Mtmp.rows() ) { Mtmp = Mtmp.topRows(Nret).eval(); }
										if (Mtmp.size() != 0)
										{
											qarray2<Symmetry::Nq> qupleP = {qP, H->Terms[t].R.out(qR)};
											auto it = Pt[t][s1].dict.find(qupleP);
											if (it != Pt[t][s1].dict.end())
											{
												if (Pt[t][s1].block[it->second].rows() == 0)
												{
													Pt[t][s1].block[it->second] = Mtmp;
												}
												else
												{
													Pt[t][s1].block[it->second] += Mtmp;
												}
											}
											else
											{
												Pt[t][s1].push_back(qupleP, Mtmp);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			P[s] = Pt[0][s];
		}
		
		for (size_t t=1; t<H->Terms.size(); ++t)
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			P[s].addScale_extend(1.,Pt[t][s]);
		}
		
		if (H->Terms.size() > 0) for (size_t s=0; s<qloc[loc].size(); ++s) P[s] = P[s].cleaned();
		
		// extend the A matrices
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t qP=0; qP<P[s].size(); ++qP)
		{
			qarray2<Symmetry::Nq> quple = {P[s].in[qP], P[s].out[qP]};
			auto qA = A[loc][s].dict.find(quple);
			
			if (qA != A[loc][s].dict.end())
			{
				addBottom(P[s].block[qP], A[loc][s].block[qA->second]);
			}
			else
			{
				if (inbase[loc].find(P[s].in[qP]))
				{
					MatrixType Mtmp(inbase[loc].inner_dim(P[s].in[qP]), P[s].block[qP].cols());
					Mtmp.setZero();
					addBottom(P[s].block[qP], Mtmp);
					A[loc][s].push_back(quple, Mtmp);
				}
				else
				{
					if (loc != 0)
					{
						bool BLOCK_INSERTED_AT_LOC = false;
						
						for (size_t qin=0; qin<inbase[loc-1].Nq(); ++qin)
						for (size_t sprev=0; sprev<qloc[loc-1].size(); ++sprev)
						{
							auto qCandidates = Symmetry::reduceSilent(inbase[loc-1][qin], qloc[loc-1][sprev]);
							auto it = find(qCandidates.begin(), qCandidates.end(), P[s].in[qP]);
							
							if (it != qCandidates.end())
							{
								if (!BLOCK_INSERTED_AT_LOC)
								{
									A[loc][s].push_back(quple, P[s].block[qP]);
									BLOCK_INSERTED_AT_LOC = true;
								}
								MatrixType Mtmp(inbase[loc-1].inner_dim(inbase[loc-1][qin]), P[s].block[qP].rows());
								Mtmp.setZero();
								A[loc-1][sprev].try_push_back(inbase[loc-1][qin], P[s].in[qP], Mtmp);
							}
						}
					}
					else
					{
						if (P[s].in[qP] == Symmetry::qvacuum())
						{
							A[loc][s].push_back(quple, P[s].block[qP]);
						}
					}
				}
			}
		}
		
		if (loc != 0)
		{
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t qA=0; qA<A[loc][s].dim; ++qA)
			for (size_t sprev=0; sprev<qloc[loc-1].size(); ++sprev)
			for (size_t qAprev=0; qAprev<A[loc-1][sprev].size(); ++qAprev)
			{
				if (A[loc-1][sprev].out[qAprev]          == A[loc][s].in[qA] and
				    A[loc-1][sprev].block[qAprev].cols() != A[loc][s].block[qA].rows())
				{
					size_t rows = A[loc-1][sprev].block[qAprev].rows();
					size_t cols = A[loc-1][sprev].block[qAprev].cols();
					int dcols = A[loc][s].block[qA].rows()-cols;
					
					A[loc-1][sprev].block[qAprev].conservativeResize(rows, cols+dcols);
					
					if (dcols > 0)
					{
						A[loc-1][sprev].block[qAprev].rightCols(dcols).setZero();
					}
				}
			}
		}
		
		update_inbase(loc);
		if (loc != 0)
		{
			update_outbase(loc-1);
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
enrich_right (size_t loc, PivotMatrix1<Symmetry,Scalar,Scalar> *H)
{
	if (this->alpha_rsvd > mynumeric_limits<Scalar>::epsilon())
	{
		std::vector<Biped<Symmetry,MatrixType> > P(qloc[loc].size());
		
//		Qbasis<Symmetry> QbasisW;
//		QbasisW.pullData(H->W, 1);
//		auto QbasisP = outbase[loc].combine(QbasisW);
//		
//		// create tensor P
//		#ifndef DMRG_DONT_USE_OPENMP
//		#pragma omp parallel for
//		#endif
//		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
//		for (size_t s2=0; s2<qloc[loc].size(); ++s2)
//		for (size_t k=0; k<H->qOp.size(); ++k)
//		{
//			if (H->W[s1][s2][k].size() == 0) {continue;}
//			for (size_t qL=0; qL<H->L.size(); ++qL)
//			{
//				auto qAs = Symmetry::reduceSilent(H->L.out(qL),qloc[loc][s2]);
//				for (const auto& qA : qAs)
//				{
//					qarray2<Symmetry::Nq> quple1 = {H->L.out(qL), qA};
//					auto itA = A[loc][s2].dict.find(quple1);
//					
//					if (itA != A[loc][s2].dict.end())
//					{
//						auto qWs = Symmetry::reduceSilent(H->L.mid(qL), H->qOp[k]);
//						
//						for (const auto& qW : qWs)
//						{
//							auto qPs = Symmetry::reduceSilent(qA,qW);
//							
//							for (const auto& qP : qPs)
//							{
//								if (qP > QoutTop[loc] or qP < QoutBot[loc]) {continue;}
//								
//								Scalar factor_cgc = Symmetry::coeff_HPsi(A[loc][s2].in[itA->second], qloc[loc][s2], A[loc][s2].out[itA->second],
//								                                         H->L.mid(qL), H->qOp[k], qW,
//								                                         H->L.in(qL), qloc[loc][s1], qP);
//								
//								if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
//								
//								auto dict_entry = H->W[s1][s2][k].dict.find({H->L.mid(qL),qW});
//								if(dict_entry == H->W[s1][s2][k].dict.end()) continue;
//								for (int spInd=0; spInd<H->W[s1][s2][k].block[dict_entry->second].outerSize(); ++spInd)
//								for (typename SparseMatrix<Scalar>::InnerIterator iW(H->W[s1][s2][k].block[dict_entry->second],spInd); iW; ++iW)
//								{
//									size_t a = iW.row();
//									size_t b = iW.col();
//									
//									size_t Prows = H->L.block[qL][a][0].rows();
//									if(Prows==0) { continue; }
//									size_t Pcols = QbasisP.inner_dim(qP);
//									if(Pcols==0) { continue; }
//									size_t Acols = A[loc][s2].block[itA->second].cols();
//									size_t stitch = QbasisP.leftAmount(qP,{qA,qW});
//									
//									MatrixType Mtmp(Prows,Pcols);
//									Mtmp.setZero();
//									
//									if (stitch >= Pcols) {continue;}
//									if (H->L.block[qL][a][0].rows() != 0 and
//									    H->L.block[qL][a][0].cols() != 0)
//									{
//										Mtmp.block(0,stitch+b*Acols, Prows,Acols) += (this->alpha_rsvd * 
//										                                             factor_cgc * 
//										                                             iW.value()) * 
//										                                             H->L.block[qL][a][0] * 
//										                                             A[loc][s2].block[itA->second];
//									}
//									
//									// VectorXd norms = Mtmp.colwise().norm();
////									vector<int> indices(Mtmp.cols());
////									iota(indices.begin(), indices.end(), 0);
//									// sort(indices.begin(), indices.end(), [norms](int i, int j){return norms(i) > norms(j);});
////									
//////									int Nret = min(static_cast<int>(0.1*Pcols),20);
//////									Nret = max(Nret,1);
//////									Nret = min(Mtmp.cols(), Nret);
//									int Nret = (this->max_Nrich<0)? Mtmp.cols():
//									                                min(static_cast<int>(Mtmp.cols()), this->max_Nrich);
//									
////									MatrixType Mret(Mtmp.rows(),Nret);
////									for (int i=0; i<Nret; ++i)
////									{
////										Mret.col(i) = Mtmp.col(indices[i]);
////									}
////									Mtmp = Mret;
//									if( Nret < Mtmp.cols() ) { Mtmp = Mtmp.leftCols(Nret).eval(); }
//									
//									if (Mtmp.size() != 0)
//									{
//										qarray2<Symmetry::Nq> qupleP = {H->L.in(qL), qP};
//										auto it = P[s1].dict.find(qupleP);
//										if (it != P[s1].dict.end())
//										{
//											if (P[s1].block[it->second].rows() == 0)
//											{
//												P[s1].block[it->second] = Mtmp;
//											}
//											else
//											{
//												P[s1].block[it->second] += Mtmp;
//											}
//										}
//										else
//										{
//											P[s1].push_back(qupleP, Mtmp);
//										}
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
		
		vector<vector<Biped<Symmetry,MatrixType> > > Pt(H->Terms.size());
		for (size_t t=0; t<H->Terms.size(); ++t) Pt[t].resize(qloc[loc].size());
		
		#ifndef DMRG_DONT_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (size_t t=0; t<H->Terms.size(); ++t)
		{
			Qbasis<Symmetry> QbasisW;
			QbasisW.pullData(H->Terms[t].W, 1);
			auto QbasisP = outbase[loc].combine(QbasisW);
			
			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
			for (size_t s2=0; s2<qloc[loc].size(); ++s2)
			for (size_t k=0; k<H->Terms[t].qOp.size(); ++k)
			{
				if (H->Terms[t].W[s1][s2][k].size() == 0) {continue;}
				for (size_t qL=0; qL<H->Terms[t].L.size(); ++qL)
				{
					auto qAs = Symmetry::reduceSilent(H->Terms[t].L.out(qL),qloc[loc][s2]);
					for (const auto& qA : qAs)
					{
						qarray2<Symmetry::Nq> quple1 = {H->Terms[t].L.out(qL), qA};
						auto itA = A[loc][s2].dict.find(quple1);
						
						if (itA != A[loc][s2].dict.end())
						{
							auto qWs = Symmetry::reduceSilent(H->Terms[t].L.mid(qL), H->Terms[t].qOp[k]);
							
							for (const auto& qW : qWs)
							{
								auto qPs = Symmetry::reduceSilent(qA,qW);
								
								for (const auto& qP : qPs)
								{
									if (qP > QoutTop[loc] or qP < QoutBot[loc]) {continue;}
									
									Scalar factor_cgc = Symmetry::coeff_HPsi(A[loc][s2].in[itA->second], qloc[loc][s2], A[loc][s2].out[itA->second],
										                                     H->Terms[t].L.mid(qL), H->Terms[t].qOp[k], qW,
										                                     H->Terms[t].L.in(qL), qloc[loc][s1], qP);
									
									if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
									
									auto dict_entry = H->Terms[t].W[s1][s2][k].dict.find({H->Terms[t].L.mid(qL),qW});
									if(dict_entry == H->Terms[t].W[s1][s2][k].dict.end()) continue;
									for (int spInd=0; spInd<H->Terms[t].W[s1][s2][k].block[dict_entry->second].outerSize(); ++spInd)
									for (typename SparseMatrix<Scalar>::InnerIterator iW(H->Terms[t].W[s1][s2][k].block[dict_entry->second],spInd); iW; ++iW)
									{
										size_t a = iW.row();
										size_t b = iW.col();
										
										size_t Prows = H->Terms[t].L.block[qL][a][0].rows();
										if(Prows==0) { continue; }
										size_t Pcols = QbasisP.inner_dim(qP);
										if(Pcols==0) { continue; }
										size_t Acols = A[loc][s2].block[itA->second].cols();
										size_t stitch = QbasisP.leftAmount(qP,{qA,qW});
										
										MatrixType Mtmp(Prows,Pcols);
										Mtmp.setZero();
										
										if (stitch >= Pcols) {continue;}
										if (H->Terms[t].L.block[qL][a][0].rows() != 0 and
											H->Terms[t].L.block[qL][a][0].cols() != 0)
										{
											Mtmp.block(0,stitch+b*Acols, Prows,Acols) += (this->alpha_rsvd * 
												                                         factor_cgc * 
												                                         iW.value()) * 
												                                         H->Terms[t].L.block[qL][a][0] * 
												                                         A[loc][s2].block[itA->second];
										}
										
										int Nret = (this->max_Nrich<0)? Mtmp.cols():
											                            min(static_cast<int>(Mtmp.cols()), this->max_Nrich);
										
										if( Nret < Mtmp.cols() ) { Mtmp = Mtmp.leftCols(Nret).eval(); }
										
										if (Mtmp.size() != 0)
										{
											qarray2<Symmetry::Nq> qupleP = {H->Terms[t].L.in(qL), qP};
											auto it = Pt[t][s1].dict.find(qupleP);
											if (it != Pt[t][s1].dict.end())
											{
												if (Pt[t][s1].block[it->second].rows() == 0)
												{
													Pt[t][s1].block[it->second] = Mtmp;
												}
												else
												{
													Pt[t][s1].block[it->second] += Mtmp;
												}
											}
											else
											{
												Pt[t][s1].push_back(qupleP, Mtmp);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			P[s] = Pt[0][s];
		}
		
		for (size_t t=1; t<H->Terms.size(); ++t)
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			P[s].addScale_extend(1.,Pt[t][s]);
		}
		
		if (H->Terms.size() > 0) for (size_t s=0; s<qloc[loc].size(); ++s) P[s] = P[s].cleaned();
		
		// extend the A matrices
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t qP=0; qP<P[s].size(); ++qP)
		{
			qarray2<Symmetry::Nq> quple = {P[s].in[qP], P[s].out[qP]};
			auto qA = A[loc][s].dict.find(quple);
			
			if (qA != A[loc][s].dict.end())
			{
				addRight(P[s].block[qP], A[loc][s].block[qA->second]);
			}
			else
			{
				if (outbase[loc].find(P[s].out[qP]))
				{
					MatrixType Mtmp(P[s].block[qP].rows(), outbase[loc].inner_dim(P[s].out[qP]));
					Mtmp.setZero();
					addRight(P[s].block[qP], Mtmp);
					A[loc][s].push_back(quple, Mtmp);
				}
				else
				{
					if (loc != this->N_sites-1)
					{
						bool BLOCK_INSERTED_AT_LOC = false;
						
						for (size_t qout=0; qout<outbase[loc+1].Nq(); ++qout)
						for (size_t snext=0; snext<qloc[loc+1].size(); ++snext)
						{
							auto qCandidates = Symmetry::reduceSilent(outbase[loc+1][qout], Symmetry::flip(qloc[loc+1][snext]));
							auto it = find(qCandidates.begin(), qCandidates.end(), P[s].out[qP]);
							
							if (it != qCandidates.end())
							{
								if (!BLOCK_INSERTED_AT_LOC)
								{
									A[loc][s].push_back(quple, P[s].block[qP]);
									BLOCK_INSERTED_AT_LOC = true;
								}
								MatrixType Mtmp(P[s].block[qP].cols(), outbase[loc+1].inner_dim(outbase[loc+1][qout]));
								Mtmp.setZero();
								A[loc+1][snext].try_push_back(P[s].out[qP], outbase[loc+1][qout], Mtmp);
							}
						}
					}
					else
					{
						if (P[s].out[qP] == Qtarget())
						{
							A[loc][s].push_back(quple, P[s].block[qP]);
						}
					}
				}
			}
		}
		
		if (loc != this->N_sites-1)
		{
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t qA=0; qA<A[loc][s].size(); ++qA)
			for (size_t snext=0; snext<qloc[loc+1].size(); ++snext)
			for (size_t qAnext=0; qAnext<A[loc+1][snext].size(); ++qAnext)
			{
				if (A[loc+1][snext].in[qAnext] == A[loc][s].out[qA] and 
				    A[loc+1][snext].block[qAnext].rows() != A[loc][s].block[qA].cols())
				{
					size_t rows = A[loc+1][snext].block[qAnext].rows();
					size_t cols = A[loc+1][snext].block[qAnext].cols();
					int drows = A[loc][s].block[qA].cols()-rows;
					
					A[loc+1][snext].block[qAnext].conservativeResize(rows+drows, cols);
					if (drows > 0)
					{
						A[loc+1][snext].block[qAnext].bottomRows(drows).setZero();
					}
				}
			}
		}
		
		update_outbase(loc);
		if (loc != this->N_sites-1)
		{
			update_inbase(loc+1);
		}
	}
}

template<typename Symmetry, typename Scalar>
Scalar Mps<Symmetry,Scalar>::
dot (const Mps<Symmetry,Scalar> &Vket) const
{
	if (Qtot != Vket.Qtarget())
	{
		lout << "calculating <φ|ψ> with different quantum numbers, " << "bra: " << Qtot << ", ket:" << Vket.Qtarget() << endl;
		return 0.;
	}
	
	Biped<Symmetry,Eigen::Matrix<Scalar,Dynamic,Dynamic> > L; 
	L.setIdentity(inBasis(0), inBasis(0));
	Biped<Symmetry,Eigen::Matrix<Scalar,Dynamic,Dynamic> > Lnext;
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		contract_L(L, A[l], Vket.A_at(l), qloc[l], Lnext);
		L.clear();
		L = Lnext;
		Lnext.clear();
	}
	
	Lnext.setIdentity(outBasis(this->N_sites-1), outBasis(this->N_sites-1));
	
	return L.contract(Lnext).trace();
}

template<typename Symmetry, typename Scalar>
template<typename MpoScalar>
Scalar Mps<Symmetry,Scalar>::
locAvg (const Mpo<Symmetry,MpoScalar> &O, size_t distance) const
{
//	cout << O.info() << endl;
	assert(this->pivot != -1 and "This function can only compute averages for Mps in mixed canonical form. Use avg() instead.");
	//assert(O.Qtarget() == Symmetry::qvacuum() and "This function can only calculate averages with local singlet operators. Use avg() instead.");
	
	size_t loc1 = this->pivot;
	size_t loc2 = this->pivot+distance;
	
	//assert(O.locality() >= loc1 and O.locality() <= loc2);
/*	lout << "loc1=" << loc1 << ", loc2=" << loc2 << endl;*/
/*	lout << O.Qtarget() << endl;*/
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Lnext;
	L.setIdentity(1,1,inBasis(loc1));
	
	for (size_t l=loc1; l<loc1+distance+1; ++l)
	{
		contract_L(L, A[l], O.W_at(l), A[l], O.locBasis(l), O.opBasis(l), Lnext);
		L = Lnext;
		Lnext.clear();
	}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	R.setIdentity(1,1,outBasis(loc2));
	
	return contract_LR(L,R);
}

template<typename Symmetry, typename Scalar>
double Mps<Symmetry,Scalar>::
squaredNorm() const
{
	double res = 0.;
	// exploit canonical form:
	if (this->pivot != -1)
	{
		/* Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > out = A[this->pivot][0].adjoint().contract(A[this->pivot][0]); */
		for (size_t s=0; s<qloc[this->pivot].size(); s++)
		for (size_t q=0; q<A[this->pivot][s].dim; ++q)
		{
			res += isReal((A[this->pivot][s].block[q].adjoint() * A[this->pivot][s].block[q]).trace()) * Symmetry::coeff_dot(A[this->pivot][s].out[q]);
			/* out += A[this->pivot][s].adjoint().contract(A[this->pivot][s]); */
		}
		/* res = out.trace(); */
	}
	// use dot product otherwise:
	else
	{
		res = isReal(dot(*this));
	}
	return res;
}

template<typename Symmetry, typename Scalar> 
void Mps<Symmetry,Scalar>::
swap (Mps<Symmetry,Scalar> &V)
{
	assert(Qmulti == V.Qmultitarget() and this->N_sites == V.length());
	
	inbase.swap(V.inbase);
	outbase.swap(V.outbase);
	
	QinTop.swap(V.QinTop);
	QinBot.swap(V.QinTop);
	QoutTop.swap(V.QinTop);
	QoutBot.swap(V.QinTop);
	
	truncWeight.swap(V.truncWeight);
	std::swap(this->pivot, V.pivot);
	std::swap(this->N_sites, V.N_sites);
	std::swap(N_phys, V.N_phys);
	
	std::swap(this->eps_svd, V.eps_svd);
	std::swap(this->eps_truncWeight, V.eps_truncWeight);
	std::swap(this->max_Nsv, V.max_Nsv);
	std::swap(this->min_Nsv, V.min_Nsv);
	std::swap(this->S, V.S);
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[l][s].in.swap(V.A[l][s].in);
		A[l][s].out.swap(V.A[l][s].out);
		A[l][s].dict.swap(V.A[l][s].dict);
		std::swap(A[l][s].dim, V.A[l][s].dim);
		
		for (size_t q=0; q<A[l][s].dim; ++q)
		{
			A[l][s].block[q].swap(V.A[l][s].block[q]);
		}
	}
}

template<typename Symmetry, typename Scalar> 
void Mps<Symmetry,Scalar>::
get_controlParams (const Mps<Symmetry,Scalar> &V)
{
	this->eps_svd = V.eps_svd;
	this->eps_truncWeight = V.eps_truncWeight;
	this->max_Nsv = V.max_Nsv;
	this->min_Nsv = V.min_Nsv;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
Mps<Symmetry,Scalar>& Mps<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	// scale the pivot site, if available; otherwise the first site
	int loc = (this->pivot == -1)? 0 : this->pivot;
	for (size_t s=0; s<qloc[loc].size(); ++s)
	for (size_t q=0; q<A[loc][s].dim; ++q)
	{
		A[loc][s].block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
Mps<Symmetry,Scalar>& Mps<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	// scale the pivot site, if available; otherwise the first site
	int loc = (this->pivot == -1)? 0 : this->pivot;
	for (size_t s=0; s<qloc[loc].size(); ++s)
	for (size_t q=0; q<A[loc][s].dim; ++q)
	{
		A[loc][s].block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
Mps<Symmetry,OtherScalar> operator* (const OtherScalar &alpha, const Mps<Symmetry,Scalar> &Vin)
{
	Mps<Symmetry,OtherScalar> Vout = Vin.template cast<OtherScalar>();
	Vout *= alpha;
	return Vout;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
Mps<Symmetry,OtherScalar> operator/ (const Mps<Symmetry,Scalar> &Vin, const OtherScalar &alpha)
{
	Mps<Symmetry,OtherScalar> Vout = Vin.template cast<OtherScalar>();
	Vout /= alpha;
	return Vout;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
applyGate(const TwoSiteGate<Symmetry,Scalar> &gate, size_t l, DMRG::DIRECTION::OPTION DIR)
{
	assert(l < this->N_sites-1 and "Can not apply a gate because l is too large.");
	assert(qloc[l] == gate.leftBasis().qloc() and "Mismatching basis at left site from gate.");
	assert(qloc[l+1] == gate.rightBasis().qloc() and "Mismatching basis at right site from gate.");

	// cout << termcolor::red << "Interchanging sites " << l << " <==> " << l+1 << termcolor::reset << endl;
	
	auto locBasis_l = gate.leftBasis();
	auto locBasis_r = gate.rightBasis();
	auto locBasis_m = gate.midBasis();
	auto qmid = locBasis_m.qs();

	//Apply the gate and get the two-site Atensor Apair.
	vector<Biped<Symmetry,MatrixType> > Apair(locBasis_m.size());
	for (size_t s1=0;  s1<qloc[l].size();  s1++)
	for (size_t s2=0;  s2<qloc[l+1].size();  s2++)
	for (size_t k=0;    k<qmid.size();   k++)
	for (size_t s1p=0; s1p<qloc[l].size(); s1p++)
	for (size_t s2p=0; s2p<qloc[l+1].size(); s2p++)
	{
		if (!Symmetry::triangle(qarray3<Symmetry::Nq>{qloc[l][s1],qloc[l+1][s2],qmid[k]})) {continue;}
		if (!Symmetry::triangle(qarray3<Symmetry::Nq>{qloc[l][s1p],qloc[l+1][s2p],qmid[k]})) {continue;}
		if (gate.data[s1][s2][s1p][s2p][k] == 0.) {continue;}
		for (size_t ql=0; ql<A[l][s1p].size(); ql++)
		{
			typename Symmetry::qType qm = A[l][s1p].out[ql];
			auto qrs = Symmetry::reduceSilent(qm,qloc[l+1][s2p]);
			for (const auto &qr : qrs)
			{
				auto it_qr = A[l+1][s2p].dict.find({qm,qr});
				if ( it_qr == A[l+1][s2p].dict.end()) {continue;}
				MatrixType Mtmp(A[l][s1p].block[ql].rows(),A[l+1][s2p].block[it_qr->second].cols());
				Scalar factor_cgc = Symmetry::coeff_twoSiteGate(A[l][s1p].in[ql], qloc[l][s1p], qm,
																qloc[l+1][s2p]  , qr          , qmid[k]);
				if (abs(factor_cgc) < ::mynumeric_limits<double>::epsilon()) {continue;}
				// cout << "l=" << l << ", s1p=" << s1p << ", ql=" << ql << ", s2p=" << s2p << "qr=" << it_qr->second << endl;
				// print_size(A[l][s1p].block[ql], "A[l][s1p].block[ql]");
				// print_size(A[l+1][s2p].block[it_qr->second], "A[l+1][s2p].block[it_qr->second]");
					
				Mtmp = factor_cgc * gate.data[s1][s2][s1p][s2p][k] * A[l][s1p].block[ql] * A[l+1][s2p].block[it_qr->second];
				size_t s1s2 = locBasis_m.outer_num(qmid[k]) + locBasis_m.leftAmount(qmid[k],{qloc[l][s1],qloc[l+1][s2]}) + locBasis_l.inner_num(s1) + locBasis_r.inner_num(s2)*locBasis_l.inner_dim(qloc[l][s1]);
				auto it_pair = Apair[s1s2].dict.find({A[l][s1p].in[ql],qr});
				if (it_pair == Apair[s1s2].dict.end())
				{
					Apair[s1s2].push_back(A[l][s1p].in[ql],qr,Mtmp);
				}
				else
				{
					Apair[s1s2].block[it_pair->second] += Mtmp;
				}
			}
		}
	}

	//Decompose the two-site Atensor Apair
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Cdumb;
	double trunc, Sdumb;
	map<qarray<Nq>,ArrayXd> SV_dumb;
	split_AA2(DIR, locBasis_m, Apair, qloc[l], A[l], qloc[l+1], A[l+1], QoutTop[l], QoutBot[l], Cdumb, false, trunc, Sdumb, SV_dumb, this->eps_truncWeight, this->min_Nsv, this->max_Nsv);
	truncWeight(l) = trunc;
	update_outbase(l);
	update_inbase(l+1);
//	split_AA(DIR, Apair, qloc[l], A[l], qloc[l+1], A[l+1], QoutTop[l], QoutBot[l], this->eps_truncWeight, this->min_Nsv, this->max_Nsv);
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
Mps<Symmetry,OtherScalar> Mps<Symmetry,Scalar>::
cast() const
{
	Mps<Symmetry,OtherScalar> Vout;
	Vout.outerResize(*this);
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	{
		Vout.A[l][s].block[q] = A[l][s].block[q].template cast<OtherScalar>();
	}
	
	Vout.eps_svd = this->eps_svd;
	Vout.eps_truncWeight = this->eps_truncWeight;
	Vout.alpha_rsvd = this->alpha_rsvd;
	Vout.max_Nsv = this->max_Nsv;
	Vout.pivot = this->pivot;
	Vout.truncWeight = truncWeight;
	
	Vout.QinTop = QinTop;
	Vout.QinBot = QinBot;
	Vout.QoutTop = QoutTop;
	Vout.QoutBot = QoutBot;
	
	Vout.Boundaries = Boundaries.template cast<OtherScalar>();
	
	Vout.update_inbase();
	Vout.update_outbase();
	
	return Vout;
}

//template<size_t D, size_t Nq>
//Mps<D,Nq,MatrixXd> operator* (const double &alpha, const Mps<D,Nq,MatrixXd> &Vin)
//{
//	Mps<D,Nq,MatrixXd> Vout = Vin;
//	Vout *= alpha;
//	return Vout;
//}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>& Mps<Symmetry,Scalar>::
operator+= (const Mps<Symmetry,Scalar> &Vin)
{
	addScale(+1.,Vin);
	return *this;
}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>& Mps<Symmetry,Scalar>::
operator-= (const Mps<Symmetry,Scalar> &Vin)
{
	addScale(-1.,Vin);
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
void Mps<Symmetry,Scalar>::
add_site (size_t loc, OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin)
{
//	if (loc == 0)
//	{
//		for (size_t s=0; s<qloc[0].size(); ++s)
//		for (size_t q=0; q<A[0][s].dim; ++q)
//		{
//			qarray2<Nq> quple = {A[0][s].in[q], A[0][s].out[q]};
//			auto it = Vin.A[0][s].dict.find(quple);
//			addRight(alpha*Vin.A[0][s].block[it->second], A[0][s].block[q]);
//		}
//	}
//	else if (loc == this->N_sites-1)
//	{
//		for (size_t s=0; s<qloc[this->N_sites-1].size(); ++s)
//		for (size_t q=0; q<A[this->N_sites-1][s].dim; ++q)
//		{
//			qarray2<Nq> quple = {A[this->N_sites-1][s].in[q], A[this->N_sites-1][s].out[q]};
//			auto it = Vin.A[this->N_sites-1][s].dict.find(quple);
//			addBottom(Vin.A[this->N_sites-1][s].block[it->second], A[this->N_sites-1][s].block[q]);
//		}
//	}
//	else
//	{
//		for (size_t s=0; s<qloc[loc].size(); ++s)
//		for (size_t q=0; q<A[loc][s].dim; ++q)
//		{
//			qarray2<Nq> quple = {A[loc][s].in[q], A[loc][s].out[q]};
//			auto it = Vin.A[loc][s].dict.find(quple);
//			addBottomRight(Vin.A[loc][s].block[it->second], A[loc][s].block[q]);
//		}
//	}
	
	// NOTE: Does not work if blocks don't match!
	if (loc == 0)
	{
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			A[loc][s].addScale(alpha, Vin.A[loc][s], RIGHT);
		}
	}
	else if (loc == this->N_sites-1)
	{
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			A[loc][s].addScale(static_cast<OtherScalar>(1.), Vin.A[loc][s], BOTTOM);
		}
	}
	else
	{
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			A[loc][s].addScale(static_cast<OtherScalar>(1.), Vin.A[loc][s], BOTTOM_RIGHT);
		}
	}
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
void Mps<Symmetry,Scalar>::
addScale (OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin, bool SVD_COMPRESS)
{
	assert(Qmulti == Vin.Qmultitarget() and 
	       "Mismatched quantum numbers in addition of Mps!");
	this->pivot = -1;
	
	if (&Vin.A == &A) // v+=α·v; results in v*=2·α;
	{
		operator*=(2.*alpha);
	}
	else
	{
		add_site(0,alpha,Vin);
		add_site(1,alpha,Vin);
//		if (SVD_COMPRESS == true)
//		{
//			rightSweepStep(0,DMRG::BROOM::SVD);
//		}
		for (size_t l=2; l<this->N_sites; ++l)
		{
			add_site(l,alpha,Vin);
//			if (SVD_COMPRESS == true)
//			{
//				rightSweepStep(l-1,DMRG::BROOM::SVD);
//			}
		}
		
		// mend the blocks without match
/*		for (size_t l=1; l<this->N_sites-1; ++l)*/
/*		for (size_t s=0; s<qloc[l].size(); ++s)*/
/*		for (size_t q=0; q<A[l][s].dim; ++q)*/
/*		{*/
/*			size_t rows = A[l][s].block[q].rows();*/
/*			size_t cols = A[l][s].block[q].cols();*/
/*			size_t rows_old = rows;*/
/*			size_t cols_old = cols;*/
/*			*/
/*			for (size_t snext=0; snext<qloc[l+1].size(); ++snext)*/
/*			for (size_t qnext=0; qnext<A[l+1][snext].dim; ++qnext)*/
/*			{*/
/*				if (A[l+1][snext].in[qnext] == A[l][s].out[q] and*/
/*				    A[l+1][snext].block[qnext].rows() > A[l][s].block[q].cols())*/
/*				{*/
/*					cols = A[l+1][snext].block[qnext].rows();*/
/*					break;*/
/*				}*/
/*			}*/
/*			*/
/*			for (size_t sprev=0; sprev<qloc[l-1].size(); ++sprev)*/
/*			for (size_t qprev=0; qprev<A[l-1][sprev].dim; ++qprev)*/
/*			{*/
/*				if (A[l-1][sprev].out[qprev] == A[l][s].in[q] and*/
/*				    A[l-1][sprev].block[qprev].cols() > A[l][s].block[q].rows())*/
/*				{*/
/*					rows = A[l-1][sprev].block[qprev].cols();*/
/*					break;*/
/*				}*/
/*			}*/
/*			*/
/*			A[l][s].block[q].conservativeResize(rows,cols);*/
/*			A[l][s].block[q].bottomRows(rows-rows_old).setZero();*/
/*			A[l][s].block[q].rightCols(cols-cols_old).setZero();*/
/*		}*/
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
set_A_from_C (size_t loc, const vector<Tripod<Symmetry,MatrixType> > &C, DMRG::BROOM::OPTION TOOL)
{
	lout << termcolor::red << "set_A_from_C is highly deprecated!" << termcolor::reset << endl;
	if (loc == this->N_sites-1)
	{
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<C[s].dim; ++q)
		{
			qarray2<Nq> cmpA = {C[s].out(q)+C[s].mid(q)-qloc[loc][s], 
			                    C[s].out(q)+C[s].mid(q)};
			auto qA = A[this->N_sites-1][s].dict.find(cmpA);
			
			if (qA != A[this->N_sites-1][s].dict.end())
			{
				// find the non-zero matrix in C[s].block[q]
				size_t w=0; while (C[s].block[q][w][0].rows() == 0) {++w;}
				A[this->N_sites-1][s].block[qA->second] = C[s].block[q][w][0];
			}
		}
	}
	else
	{
		vector<vector<MatrixType> > Omega(qloc[loc].size());
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			Omega[s].resize(C[s].dim);
			for (size_t q=0; q<C[s].dim; ++q)
			{
				size_t r=0; while (C[s].block[q][r][0].rows()==0 or C[s].block[q][r][0].cols()==0) {++r;}
				typename MatrixType::Index Crows = C[s].block[q][r][0].rows();
				typename MatrixType::Index Ccols = C[s].block[q][r][0].cols();
				
				for (size_t w=0; w<C[s].block[q].size(); ++w)
				{
					if (C[s].block[q][w][0].rows() != 0)
					{
						addRight(C[s].block[q][w][0], Omega[s][q]);
					}
					else
					{
						MatrixType Mtmp(Crows,Ccols);
						Mtmp.setZero();
						addRight(Mtmp, Omega[s][q]);
					}
				}
			}
		}
		
		ArrayXd truncWeightSub(outbase[loc].Nq());
		truncWeightSub.setZero();
		
//		#ifndef DMRG_DONT_USE_OPENMP
//		#pragma omp parallel for
//		#endif
		for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
		{
			map<tuple<size_t,size_t,size_t>,vector<size_t> > sqmap; // map s,qA,rows -> q{mid}
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<C[s].dim; ++q)
			{
				qarray2<Nq> cmpA = {outbase[loc][qout]-qloc[loc][s], outbase[loc][qout]};
				auto qA = A[loc][s].dict.find(cmpA);
				
				if (C[s].mid(q)+C[s].out(q) == outbase[loc][qout])
//				if (C[s].mid(q)+C[s].out(q) == outbase[loc][qout] and qA != A[loc][s].dict.end())
				{
					tuple<size_t,size_t,size_t> key = make_tuple(s, qA->second, Omega[s][q].rows());
					sqmap[key].push_back(q);
				}
			}
			
			vector<size_t> svec;
			vector<size_t> qAvec;
			vector<size_t> Nrowsvec;
			vector<vector<size_t> > qmidvec;
			for (auto it=sqmap.begin(); it!=sqmap.end(); ++it)
			{
				svec.push_back(get<0>(it->first));
				qAvec.push_back(get<1>(it->first));
				Nrowsvec.push_back(get<2>(it->first));
				qmidvec.push_back(it->second);
			}
			
			if (Nrowsvec.size() != 0)
			{
				size_t Nrows = accumulate(Nrowsvec.begin(), Nrowsvec.end(), 0);
				
				vector<vector<size_t> > Ncolsvec(qmidvec.size());
				for (size_t i=0; i<qmidvec.size(); ++i)
				{
					size_t s = svec[i];
					for (size_t j=0; j<qmidvec[i].size(); ++j)
					{
						size_t q = qmidvec[i][j];
						Ncolsvec[i].push_back(Omega[s][q].cols());
					}
				}
				
				size_t Ncols = accumulate(Ncolsvec[0].begin(), Ncolsvec[0].end(), 0);
				for (size_t i=0; i<Ncolsvec.size(); ++i)
				{
					size_t Ncols_new = accumulate(Ncolsvec[i].begin(), Ncolsvec[i].end(), 0);
					if (Ncols_new > Ncols) {Ncols = Ncols_new;}
				}
				
				MatrixType Cclump(Nrows,Ncols);
				Cclump.setZero();
				size_t istitch = 0;
				size_t jstitch = 0;
				for (size_t i=0; i<Nrowsvec.size(); ++i)
				{
					for (size_t j=0; j<Ncolsvec[i].size(); ++j)
					{
						Cclump.block(istitch,jstitch, Nrowsvec[i],Ncolsvec[i][j]) = Omega[svec[i]][qmidvec[i][j]];
						jstitch += Ncolsvec[i][j];
					}
					istitch += Nrowsvec[i];
					jstitch = 0;
				}
				
//				cout << Cclump.rows() << ", " << Cclump.cols() << endl;
//				size_t Nzerocols = 0;
				for (size_t i=0; i<Cclump.cols(); ++i)
				{
					if (Cclump.col(i).norm() == 0 and Cclump.cols() > 1)
					{
						remove_col(i,Cclump);
//						++Nzerocols;
					}
				}
//				cout << "Nzerocols=" << Nzerocols << endl;
//				cout << Cclump.rows() << ", " << Cclump.cols() << endl;
				
//				for (size_t i=0; i<Nrowsvec.size(); ++i)
//				{
//					for (size_t j=0; j<Ncolsvec[i].size(); ++j)
//					{
//						cout << svec[i] << "," << qmidvec[i][j] << " ";
//					}
//					cout << endl;
//				}
//				cout << endl;
				
				#ifdef DONT_USE_BDCSVD
				JacobiSVD<MatrixType>  Jack(Cclump,ComputeThinU); // standard SVD
				#else
				BDCSVD<MatrixType> Jack(Cclump,ComputeThinU); // "Divide and conquer" SVD (only available in Eigen)
				#endif
				
				size_t Nret;
				if (TOOL == DMRG::BROOM::BRUTAL_SVD)
				{
					Nret = (Jack.singularValues().array() > 0.).count();
					Nret = min(Nret, this->max_Nsv);
				}
				else // SVD
				{
					Nret = (Jack.singularValues().array() > this->eps_svd).count();
				}
				Nret = max(Nret,1ul);
				truncWeightSub(qout) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
				
				size_t stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					if (TOOL == DMRG::BROOM::QR)
					{
						Nret = min(A[loc][svec[i]].block[qAvec[i]].cols(), Jack.matrixU().cols());
					}
					A[loc][svec[i]].block[qAvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret);
					stitch += Nrowsvec[i];
				}
			}
		}
		
		truncWeight(loc) = truncWeightSub.sum();
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
mend()
{
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	{
		if (A[l][s].block[q].rows()==0 and A[l][s].block[q].cols()==0)
		{
			size_t rows = 1;
			size_t cols = 1;
			
			if (l != 0)
			{
				size_t sm = 0;
				bool GOT_A_MATCH = false;
				while (GOT_A_MATCH == false and sm<qloc[l-1].size())
				{
					qarray2<Nq> cmpm = {A[l][s].in[q]-qloc[l-1][sm], A[l][s].in[q]};
					auto qm = A[l-1][sm].dict.find(cmpm);
					if (qm != A[l-1][sm].dict.end())
					{
						rows = max(static_cast<size_t>(A[l-1][sm].block[qm->second].cols()),1ul);
						GOT_A_MATCH = true;
					}
					else {++sm;}
				}
			}
			
			if (l != this->N_sites-1)
			{
				size_t sp = 0;
				bool GOT_A_MATCH = false;
				while (GOT_A_MATCH == false and sp<qloc[l+1].size())
				{
					qarray2<Nq> cmpp = {A[l][s].out[q], A[l][s].out[q]+qloc[l+1][sp]};
					auto qp = A[l+1][sp].dict.find(cmpp);
					if (qp != A[l+1][sp].dict.end())
					{
						cols = max(static_cast<size_t>(A[l+1][sp].block[qp->second].rows()),1ul);
						GOT_A_MATCH = true;
					}
					else {++sp;}
				}
			}
			
			A[l][s].block[q].resize(rows,cols);
			A[l][s].block[q].setZero();
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
collapse()
{
//	vector<qarray<Nq> > conf(this->N_sites);
//	vector<std::array<double,D> > newAvals(this->N_sites);
//	
//	for (size_t l=0; l<this->N_sites; ++l)
//	{
//		MatrixXd BasisTrafo = randOrtho(qloc[l].size());
//		vector<double> prob(qloc[l].size());
//		vector<double> ranges(D+1);
//		ranges[0] = 0.;
//		
//		for (size_t i=0; i<qloc[i].size(); ++i)
//		{
//			prob[i] = 0.;
//			
//			Biped<Symmetry,MatrixType> Arow = BasisTrafo(0,i) * A[l][0].adjoint();
//			for (size_t s=1; s<qloc[i].size(); ++s)
//			{
//				Arow += BasisTrafo(s,i) * A[l][s].adjoint();
//			}
//			
//			Biped<Symmetry,MatrixType> Acol = BasisTrafo(0,i) * A[l][0];
//			for (size_t s=1; s<qloc[i].size(); ++s)
//			{
//				Acol += BasisTrafo(s,i) * A[l][s];
//			}
//			
//			for (size_t q=0; q<Arow.dim; ++q)
//			{
//				qarray2<Nq> quple = {Arow.out[q], Arow.in[q]};
//				auto it = Acol.dict.find(quple);
//				if (it != Acol.dict.end())
//				{
//					prob[i] += (Acol.block[it->second] * Arow.block[q])(0,0);
//				}
//			}
//			
//			ranges[i+1] = ranges[i] + prob[i];
//		}
//		assert(fabs(ranges[D]-1.) < 1e-14 and 
//		       "Probabilities in collapse don't add up to 1!");
//		
//		double die = UniformDist(MtEngine);
//		size_t select;
//		for (size_t i=1; i<D+1; ++i)
//		{
//			if (die>=ranges[i-1] and die<ranges[i])
//			{
//				select = i-1;
//			}
//		}
//		conf[l] = qloc[select];
//		
//		if (l<this->N_sites-1)
//		{
//			for (size_t s2=0; s2<qloc[l].size(); ++s2)
//			{
//				Biped<Symmetry,MatrixType> Mtmp = BasisTrafo(0,select) * A[l][0] * A[l+1][s2];
//				for (size_t s1=1; s1<qloc[l].size(); ++s1)
//				{
//					Mtmp += BasisTrafo(s1,select) * A[l][s1] * A[l+1][s2];
//				}
//				A[l+1][s2] = 1./sqrt(prob[select]) * Mtmp;
//			}
//		}
//		
//		for (size_t s1=0; s1<qloc[l].size(); ++s1)
//		{
//			newAvals[l][s1] = BasisTrafo(s1,select);
//		}
//	}
//	
//	setProductState(conf);
//	
////	outerResize(this->N_sites, qloc, accumulate(conf.begin(),conf.end(),Symmetry::qvacuum()));
////	
////	for (size_t l=0; l<this->N_sites; ++l)
////	for (size_t s=0; s<qloc[l].size(); ++s)
////	for (size_t q=0; q<A[l][s].dim; ++q)
////	{
////		A[l][s].block[q].resize(1,1);
////		A[l][s].block[q](0,0) = newAvals[l][s];
////	}
}

// template<typename Symmetry, typename Scalar>
// template<size_t MpoNq>
// void Mps<Symmetry,Scalar>::
// setFlattenedMpo (const Mpo<MpoNq,Scalar> &Op, bool USE_SQUARE)
// {
// 	static_assert (Nq == 0, "A flattened Mpo must have Nq=0!");
	
// 	this->N_sites = Op.length();
// 	Qtot = qvacuum<0>();
// 	this->set_defaultCutoffs();
	
// 	// set local basis
// 	qloc.resize(this->N_sites);
// 	for (size_t l=0; l<this->N_sites; ++l)
// 	{
// 		size_t D = Op.locBasis(l).size();
// 		qloc[l].resize(D*D);
		
// 		for (size_t s1=0; s1<D; ++s1)
// 		for (size_t s2=0; s2<D; ++s2)
// 		{
// 			qloc[l][s2+D*s1] = qvacuum<0>();
// 		}
// 	}
	
// 	resize_arrays();
// 	outerResizeNoSymm();
// 	innerResize(1);
	
// 	for (size_t l=0; l<this->N_sites; ++l)
// 	{
// 		size_t D = Op.locBasis(l).size();
// 		for (size_t s1=0; s1<D; ++s1)
// 		for (size_t s2=0; s2<D; ++s2)
// 		{
// 			size_t s = s2 + D*s1;
// 			if (USE_SQUARE)
// 			{
// 				A[l][s].block[0] = MatrixType(Op.Wsq_at(l)[s1][s2]);
// 			}
// 			else
// 			{
// 				A[l][s].block[0] = MatrixType(Op.W_at(l)[s1][s2]);
// 			}
// 		}
// 	}
// }

template<typename Symmetry, typename Scalar>
string Mps<Symmetry,Scalar>::
validate (string name) const
{
	stringstream ss;
	
//	for (size_t s=0; s<qloc[0].size(); ++s)
//	for (size_t q=0; q<A[0][s].dim; ++q)
//	{
//		if (A[0][s].block[q].rows() != 1)
//		{
//			ss << name << " has wrong dimensions at: l=0: rows=" << A[0][s].block[q].rows() << " != 1" << endl;
//		}
//	}
	
	for (size_t l=0; l<this->N_sites-1; ++l)
	for (size_t s1=0; s1<qloc[l].size(); ++s1)
	for (size_t q1=0; q1<A[l][s1].dim; ++q1)
	for (size_t s2=0; s2<qloc[l+1].size(); ++s2)
	for (size_t q2=0; q2<A[l+1][s2].dim; ++q2)
	{
		if (A[l][s1].out[q1] == A[l+1][s2].in[q2])
		{
			if (A[l][s1].block[q1].cols()-A[l+1][s2].block[q2].rows() != 0)
			{
				ss << name << " has wrong dimensions at: l=" << l << "→" << l+1
				<< ", qnum=" << A[l][s1].out[q1] 
			    << ", s1=" << Sym::format<Symmetry>(qloc[l][s1]) << ", s2=" << Sym::format<Symmetry>(qloc[l+1][s1])
				<< ", cols=" << A[l][s1].block[q1].cols() << " → rows=" << A[l+1][s2].block[q2].rows() << endl;
			}
			if (A[l][s1].block[q1].cols() == 0 or A[l+1][s2].block[q2].rows() == 0)
			{
				ss << name << " has zero dimensions at: l=" << l << "→" << l+1
				<< ", qnum=" << A[l][s1].out[q1] 
				<< ", s1=" << Sym::format<Symmetry>(qloc[l][s1]) << ", s2=" << Sym::format<Symmetry>(qloc[l+1][s1])
				<< ", cols=" << A[l][s1].block[q1].cols() << " → rows=" << A[l+1][s2].block[q2].rows() << endl;
			}
		}
	}
	
//	for (size_t s=0; s<qloc[this->N_sites-1].size(); ++s)
//	for (size_t q=0; q<A[this->N_sites-1][s].dim; ++q)
//	{
//		if (A[this->N_sites-1][s].block[q].cols() != 1)
//		{
//			ss << name << " has wrong dimensions at: l=" << this->N_sites-1 << ": cols=" << A[this->N_sites-1][s].block[q].cols() << " != 1" << endl;
//		}
//	}
	
	if (ss.str().size() == 0)
	{
		ss << name << " looks okay!";
	}
	return ss.str();
}

template<typename Symmetry, typename Scalar>
string Mps<Symmetry,Scalar>::
test_ortho (double tol) const
{
	stringstream sout;
	std::array<string,4> normal_token  = {"A","B","M","X"};
	std::array<string,4> special_token = {"\e[4mA\e[0m","\e[4mB\e[0m","\e[4mM\e[0m","\e[4mX\e[0m"};
	
	std::array<string,4> normal_token_for_nullspace  = {"F","G","M","X"};
	std::array<string,4> special_token_for_nullspace = {"\e[4mF\e[0m","\e[4mG\e[0m","\e[4mM\e[0m","\e[4mX\e[0m"};
	
	for (int l=0; l<this->N_sites; ++l)
	{
		// check for A
		Biped<Symmetry,MatrixType> Test = A[l][0].adjoint().contract(A[l][0]);
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[l][s].adjoint().contract(A[l][s]);
		}
		
		vector<bool> A_CHECK(Test.dim);
		vector<double> A_infnorm(Test.dim);
		for (size_t q=0; q<Test.dim; ++q)
		{
//			if (l == 0) {cout << "A l=" << l << ", Test.block[q]=" << endl << Test.block[q] << endl << endl;}
//			MatrixType Id = MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
//			if (l == 0) {Id.bottomRows(Id.rows()-1).setZero();}
//			Test.block[q] -= Id;
			Test.block[q] -= MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
			A_CHECK[q] = Test.block[q].template lpNorm<Infinity>()<tol ? true : false;
			A_infnorm[q] = Test.block[q].template lpNorm<Infinity>();
		}
		
		// check for B
		Test.clear();
		Test = A[l][0].contract(A[l][0].adjoint(),contract::MODE::OORR);
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test = Test + A[l][s].contract(A[l][s].adjoint(),contract::MODE::OORR);
		}
		
		vector<bool> B_CHECK(Test.dim);
		vector<double> B_infnorm(Test.dim);
		for (size_t q=0; q<Test.dim; ++q)
		{
//			MatrixType Id = MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
//			if (l == this->N_sites-1) {Id.bottomRows(Id.rows()-1).setZero();}
//			Test.block[q] -= Id;
//			cout << "B l=" << l << ", Test.block[q]=" << Test.block[q] << endl << endl;
			Test.block[q] -=  MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
			B_CHECK[q] = Test.block[q].template lpNorm<Infinity>()<tol ? true : false;
			B_infnorm[q] = Test.block[q].template lpNorm<Infinity>();
		}
		
		// interpret result
		if (all_of(A_CHECK.begin(),A_CHECK.end(),[](bool x){return x;}) and 
		    all_of(B_CHECK.begin(),B_CHECK.end(),[](bool x){return x;}))
		{
			sout << termcolor::magenta;
			sout << ((l==this->pivot) ? special_token[3] : normal_token[3]); // X
		}
		else if (all_of(A_CHECK.begin(),A_CHECK.end(),[](bool x){return x;}))
		{
			sout << termcolor::red;
			sout << ((l==this->pivot) ? special_token[0] : normal_token[0]); // A
		}
		else if (all_of(B_CHECK.begin(),B_CHECK.end(),[](bool x){return x;}))
		{
			sout << termcolor::blue;
			sout << ((l==this->pivot) ? special_token[1] : normal_token[1]); // B
		}
		else
		{
			sout << termcolor::green;
			sout << ((l==this->pivot) ? special_token[2] : normal_token[2]); // M
		}
	}
	sout << termcolor::reset;
	return sout.str();
}

template<typename Symmetry, typename Scalar>
std::pair<vector<qarray<Symmetry::Nq> >, ArrayXd> Mps<Symmetry,Scalar>::
entanglementSpectrumLoc (size_t loc) const
{
	vector<pair<qarray<Nq>, double> > Svals;
	for (const auto &x : SVspec[loc])
	for (int i=0; i<x.second.size(); ++i)
	{
		Svals.push_back(std::make_pair(x.first,x.second(i)));
	}
	sort(Svals.begin(), Svals.end(), [] (const pair<qarray<Nq>, double> &p1, const pair<qarray<Nq>, double> &p2) { return p2.second < p1.second;});
	// reverse(Svals.begin(), Svals.end());
	
	ArrayXd Sout(Svals.size());
	vector<qarray<Nq> > Qout(Svals.size());
	for (int i=0; i<Svals.size(); ++i)
	{
		Sout(i) = Svals[i].second;
		Qout[i] = Svals[i].first;
	}
	return std::make_pair(Qout,Sout);
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
graph (string filename) const
{
	stringstream ss;
	
	ss << "#!/usr/bin/dot dot -Tpdf -o " << filename << ".pdf\n\n";
	ss << "digraph G\n{\n";
	ss << "rankdir = LR;\n";
	ss << "labelloc=\"t\";\n";
	ss << "label=\"MPS: L=" << this->N_sites << ", (";
	for (size_t q=0; q<Nq; ++q)
	{
		ss << Symmetry::kind()[q];
		if (q!=Nq-1) {ss << ",";}
	}
	ss << ")=" << Sym::format<Symmetry>(Qtot) << "\";\n";
	
	// vacuum node
	ss << "\"l=" << 0 << ", " << Sym::format<Symmetry>(Symmetry::qvacuum()) << "\"";
	ss << "[label=" << "\"" << Sym::format<Symmetry>(Symmetry::qvacuum()) << "\"" << "];\n";
	
	// site nodes
	for (size_t l=0; l<this->N_sites; ++l)
	{
		ss << "subgraph" << " cluster_" << l << "\n{\n";
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<A[l][s].dim; ++q)
		{
			string qin  = Sym::format<Symmetry>(A[l][s].in[q]);
			ss << "\"l=" << l << ", " << qin << "\"";
			ss << "[label=" << "\"" << qin << "\"" << "];\n";
		}
		if (l>0) {ss << "label=\"l=" << l << "\"\n";}
		else     {ss << "label=\"vacuum\"\n";}
		ss << "}\n";
	}
	
	// last node
	ss << "subgraph" << " cluster_" << this->N_sites << "\n{\n";
	ss << "\"l=" << this->N_sites << ", " << Sym::format<Symmetry>(Qtot) << "\"";
	ss << "[label=" << "\"" << Sym::format<Symmetry>(Qtot) << "\"" << "];\n";
	ss << "label=\"l=" << this->N_sites << "\"\n";
	ss << "}\n";
	
	// edges
	for (size_t l=0; l<this->N_sites; ++l)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<A[l][s].dim; ++q)
		{
			string qin  = Sym::format<Symmetry>(A[l][s].in[q]);
			string qout = Sym::format<Symmetry>(A[l][s].out[q]);
			ss << "\"l=" << l << ", " << qin << "\"";
			ss << "->";
			ss << "\"l=" << l+1 << ", " << qout << "\"";
			ss << " [label=\"" << A[l][s].block[q].rows() << "x" << A[l][s].block[q].cols() << "\"";
			ss << "];\n";
		}
	}
	
	ss << "\n}";
	
	ofstream f(filename+".dot");
	f << ss.str();
	f.close();
}

template<typename Symmetry, typename Scalar>
string Mps<Symmetry,Scalar>::
Asizes() const
{
	stringstream ss;
	ss << endl << "Asizes:" << endl;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		ss << "\tl=" << l << ": ";
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			ss << "s=" << s << ": ";
			for (size_t q=0; q<A[l][s].dim; ++q)
			{
				ss << "(" << A[l][s].block[q].rows() << "," << A[l][s].block[q].cols() << ") ";
			}
		}
		if (l!=this->N_sites-1) {ss << endl;}
	}
	return ss.str();
}

template<typename Symmetry, typename Scalar>
double Mps<Symmetry,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		res += A[l][s].memory(memunit);
	}
	return res;
}

template<typename Symmetry, typename Scalar>
ostream &operator<< (ostream& os, const Mps<Symmetry,Scalar> &V)
{	
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "Mps: L=" << V.length();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<V.length(); ++l)
	{
		for (size_t s=0; s<V.locBasis(l).size(); ++s)
		{
			os << "l=" << l << "\ts=" << Sym::format<Symmetry>(V.locBasis(l)[s]) << endl;
			os << V.A_at(l)[s].print(true); //V.A_at(l)[s].formatted();
			os << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != V.length()-1) {os << endl;}
	}
	return os;
}

#endif
