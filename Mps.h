#ifndef STRAWBERRY_MPS_WITH_Q
#define STRAWBERRY_MPS_WITH_Q

#include <set>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <type_traits>
#include <iostream>
#include <fstream>

#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "Mpo.h"
#include "pivot/DmrgPivotMatrix1.h"
#include "DmrgJanitor.h"
#include "tensors/DmrgConglutinations.h"
#ifdef USE_HDF5_STORAGE
	#include <HDF5Interface.h>
#endif
#include "PolychromaticConsole.h" // from TOOLS
#include "RandomVector.h" // from ALGS
#include "numeric_limits.h" // from TOOLS

/**
 * Matrix Product State with conserved quantum numbers (Abelian and non abelian symmetries).
 * \describe_Symmetry
 * \describe_Scalar
 */
template<typename Symmetry, typename Scalar=double>
class Mps : public DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	static constexpr size_t Nq = Symmetry::Nq;
	typedef typename Symmetry::qType qType;
	
	// Note: Cannot partially specialize template friends (or anything else, really).
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class DmrgSolver;
	template<typename Symmetry_, typename S1, typename S2> friend class MpsCompressor;
	template<typename H, typename Symmetry_, typename S1, typename S2, typename V> friend class TDVPPropagator;
	template<typename Symmetry_, typename S1, typename S2> friend
	void HxV (const Mpo<Symmetry_,S1> &H, const Mps<Symmetry_,S2> &Vin, Mps<Symmetry_,S2> &Vout, DMRG::VERBOSITY::OPTION VERBOSITY);
	template<typename Symmetry_, typename S1, typename S2> friend 
	void OxV (const Mpo<Symmetry_,S1> &H, const Mps<Symmetry_,S2> &Vin, Mps<Symmetry_,S2> &Vout, DMRG::BROOM::OPTION TOOL);
	template<typename Symmetry_, typename S_> friend class Mps; // in order to exchange data between real & complex Mps
	
public:
	
	/**Does nothing.*/
	Mps();
	
	/**
	 * Construct by setting all the relevant parameters.
	 * \param L_input : chain length
	 * \param qloc_input : local basis
	 * \param Qtot_input : target quantum number
	 * \param N_phys_input : the volume of the system (normally (chain length) * (chain width))
	 * \param Qmax_input : Maximal number of symmetry blocks per site in the Mps
	 */
	Mps (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input, int Qmax_input);
	
	/** 
	 * Construct by pulling info from an MPO.
	 * \param H : chain length and local basis will be retrieved from this Mpo
	 * \param Dmax : size cutoff (per subspace)
	 * \param Qtot_input : target quantum number
	 * \param Qmax_input : Maximal number of symmetry blocks per site in the Mps
	 */
	template<typename Hamiltonian> Mps (const Hamiltonian &H, size_t Dmax, qarray<Nq> Qtot_input, int Qmax_input);
	
	/** 
	 * Construct by explicitly provide the A-matrices. Basically only for testing purposes.
	 * \param L_input : chain length
	 * \param As : vector of vector of A matrices. (For all sites, and for all local quantumnumbers)
	 * \param qloc_input : vector of local basis for all sites.
	 * \param Qtot_input : target quantum number
	 * \param N_phys_input : the volume of the system (normally (chain length) * (chain width))
	 */
	Mps (size_t L_input, const vector<vector<Biped<Symmetry,MatrixXd> > > &As,
	     const vector<vector<qarray<Nq> > > &qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input);
	
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
	 * Save all matrices of the MPS to the file <FILENAME>.h5.
	 * \param filename : the format is fixed to .h5. Just enter the name without the format.
	 * \param info : Additional information about the used model. Enter the info()-method of the used MPO here.
	 * \warning This method requires hdf5. For more information visit https://www.hdfgroup.org/.
	 * \note For the filename you should use the info string of the current used Mpo.
	 */
	void save (string filename,string info="none");
	
	/**
	 * Reads all matrices of the MPS from the file <FILENAME>.h5.
	 * \param filename : the format is fixed to .h5. Just enter the name without the format.
	 * \warning This method requires hdf5. For more information visit https://www.hdfgroup.org/.
	 */
	void load (string filename);
	
	/**
	 * Returns the maximal bond-dimension of an MPS stored in a file <FILENAME>.h5.
	 * \param filename : the format is fixed to .h5. Just enter the name without the format.
	 * \warning This method requires hdf5. For more information visit https://www.hdfgroup.org/.
	 * \note Use case : First call loadDmax to construct the Mps with Mps::Mps(const Hamiltonian &H, size_t Dmax, qarray<Nq> Qtot_input).
	 *                  Then call Mps::load() to get the Mps matrices.
	 */
	size_t loadDmax (string filename);
	///\}
	#endif //USE_HDF5_STORAGE
	
	/**
	 * Determines all subspace quantum numbers and resizes the containers for the blocks. Memory for the matrices remains uninitialized.
	 * \param L_input : chain length
	 * \param qloc_input : local basis
	 * \param Qtot_input : target quantum number
	 * \param Nqmax_input : Maximum number of symmetry blocks in the Mps per site.
	 */
	void outerResize (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, int Nqmax_input=500);
	
	/**
	 * Determines all subspace quantum numbers and resizes the containers for the blocks. Memory for the matrices remains uninitiated. Pulls info from an Mpo.
	 * \param H : chain length and local basis will be retrieved from this Mpo
	 * \param Qtot_input : target quantum number
	 * \param Nqmax_input : Maximum number of symmetry blocks in the Mps per site.
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
	 * \param Dmax : size cutoff (per subspace)
	 */
	void innerResize (size_t Dmax);
	
	/**
	 * Sets the Mps from a product state configuration.
	 * \param H : Hamiltonian, needed for Mps::outerResize
	 * \param config : classical configuration, a vector of \p qarray
	 */
	template<typename Hamiltonian> void setProductState (const Hamiltonian &H, const vector<qarray<Nq> > &config);
	
	/**
	 * Finds broken paths through the quantum number subspaces and mends them by resizing with appropriate zeros. 
	 * The chain length and total quantum number are determined from \p config.
	 * This is needed when applying an Mpo which changes quantum numbers, making some paths impossible. 
	 * For example, one can add a particle at the beginning or end of the chain with the same target particle number, 
	 * but if an annihilator is applied in the middle, only the first path survives.
	 */
	void mend();
	
	/**
	 * Sets the A-matrix at a given site by performing SVD on the C-tensor.
	 * \warning Not implemented for non abelian symmetries.
	 */
	void set_A_from_C (size_t loc, const vector<Tripod<Symmetry,MatrixType> > &C, DMRG::BROOM::OPTION TOOL=DMRG::BROOM::SVD);
	
	
	// \param Op : 
	// \param USE_SQUARE :
	// template<size_t MpoNq> void setFlattenedMpo (const Mpo<MpoNq,Scalar> &Op, bool USE_SQUARE=false);
	
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
	
	size_t calc_fullMmax() const;
	
	/**
	 * Determines the maximal amount of rows or columns per site and subspace.
	 */
	size_t calc_Dmax() const;
	
	/**
	 * Determines the maximal amount of subspaces per site.
	 */
	size_t calc_Nqmax() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	///\}
	
	///\{
	/**
	 * Adds another Mps to the given one and scales by \p alpha, i\.e\. performs \f$ \mathrel{+}= \alpha \cdot V_{in}\f$.
	 * \param alpha : scalar for scaling
	 * \param Vin : Mps to be added
	 * \param SVD_COMPRESS : If \p true, the resulting Mps is compressed using SVD. If \p false, the summation is exact (direct sum of the matrices).
	 * \warning Not implemented for non abelian symmetries.
	 */
	template<typename OtherScalar> void addScale (OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin, bool SVD_COMPRESS=false);
	
	/**
	 *Performs Mps::addScale with \p alpha = 1.
	 * \warning Not implemented for non abelian symmetries.
	 */
	Mps<Symmetry,Scalar>& operator+= (const Mps<Symmetry,Scalar> &Vin);
	
	/**
	 *Performs Mps::addScale with \p alpha = -1.
	 * \warning Not implemented for non abelian symmetries.
	 */
	Mps<Symmetry,Scalar>& operator-= (const Mps<Symmetry,Scalar> &Vin);
	
	/**
	 * Performs \f$ \mathrel{*}= \alpha\f$. Applies it to the first site.
	 */
	template<typename OtherScalar> Mps<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	
	/**
	 * Performs \f$ \mathrel{/}= \alpha\f$. Applies it to the first site.
	 */
	template<typename OtherScalar> Mps<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	
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
	 * \param O : Local Mpo acting on the pivot side.
	 * \warning Not implemented for non abelian symmetries.
	 */
	template<typename MpoScalar> Scalar locAvg (const Mpo<Symmetry,MpoScalar> &O) const;
	
	/**Calculates the expectation value with a local operator at pivot and pivot+1.*/
	// template<typename MpoScalar> Scalar locAvg2 (const Mpo<Nq,MpoScalar> &O) const;
	
	/**Swaps with another Mps.*/
	void swap (Mps<Symmetry,Scalar> &V);
	
	/**
	 * Copies the control parameters from another Mps, i.e.\ all the cutoff tolerances specified in DmrgJanitor.
	 */
	void get_controlParams (const Mps<Symmetry,Scalar> &V);
	
	/**For METTS.*/
	void collapse();
	///\}
	
	///\{
	/**
	 * Performs a sweep step to the right.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc+1
	 * \param BROOM : choice of decomposition
	 * \param H : non-local information from transfer matrices is provided here when \p BROOM is DMRG::BROOM::RDM or DMRG::BROOM::RICH_SVD
	 * \param DISCARD_V : If \p true, don't multiply the V-matrix onto the next site
	 */
	void rightSweepStep (size_t loc, DMRG::BROOM::OPTION BROOM, PivotMatrix1<Symmetry,Scalar,Scalar> *H = NULL, bool DISCARD_V=false);
	
	/**
	 * Performs a sweep step to the left.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1
	 * \param BROOM : choice of decomposition
	 * \param H : non-local information from transfer matrices is provided here when \p BROOM is DMRG::BROOM::RDM or DMRG::BROOM::RICH_SVD
	 * \param DISCARD_U : If \p true, don't multiply the U-matrix onto the next site
	 */
	void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION BROOM, PivotMatrix1<Symmetry,Scalar,Scalar> *H = NULL, bool DISCARD_U=false);

	/**
	 * Calculates the nullspace of the site tensor on site \p loc when blocked into direction \p DIR.
	 * \param DIR : Direction of the weep. Either LEFT or RIGHT.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1
	 * \param N : Tensor to write the Nullsapce to.
	 * \note Nullspace is obtained by a full QR decomposition.
	 * \note The nullspace is used for error estimation as suggested here: arXiv:1711.01104
	 */
	void calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,MatrixType> > &N);
	
	/**
	 * Performs a two-site sweep.
	 * \param DIR : Direction of the weep. Either LEFT or RIGHT.
	 * \param loc : site to perform the sweep on; afterwards the pivot is shifted to \p loc-1
	 * \param Apair : Pair of two Mps site tensors which are splitted via a singular value decomposition.
	 * \param DISCARD_SV: If \p true, the singular value matrix is discarded. Useful for iDMRG.
	 * \warning Not implemented for non abelian symmetries.
	 * \todo Implemented this function for SU(2) symmetry.
	 */
	void sweepStep2 (DMRG::DIRECTION::OPTION DIR, size_t loc, const vector<Biped<Symmetry,MatrixType> > &Apair, bool DISCARD_SV=false);
	
	/**
	 * Performs an SVD split to the left and writes the zero-site tensor to \p C.
	 */
	void leftSplitStep  (size_t loc, Biped<Symmetry,MatrixType> &C);
	
	/**
	 * Performs an SVD split to the right and writes the zero-site tensor to \p C.
	 */
	void rightSplitStep (size_t loc, Biped<Symmetry,MatrixType> &C);
	
	/**
	 * Absorbs the zero-site tensor \p C (as obtained after an SVD split) into the Mps.
	 * \param loc : site to do the absorption
	 * \param DIR : specifies whether the absorption is on the left-sweep or right-sweep
	 * \param C : the zero-site tensor to be absorbed
	 * \warning Not implemented for non abelian symmetries.
	 */
	void absorb (size_t loc, DMRG::DIRECTION::OPTION DIR, const Biped<Symmetry,MatrixType> &C);
	///\}
	
	///\{
	/**Returns the target quantum number.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
	
	/**Returns the local basis.*/
	inline vector<qarray<Nq> > locBasis (size_t loc) const {return qloc[loc];}
	inline vector<vector<qarray<Nq> > > locBasis()   const {return qloc;}
	
	/**Const reference to the A-tensor at site \p loc.*/
	const vector<Biped<Symmetry,MatrixType> > &A_at (size_t loc) const {return A[loc];};
	
	/**Returns the pivot position.*/
	inline int get_pivot() const {return this->pivot;};
	
	/**Returns the truncated weight per site (Eigen array).*/
	inline ArrayXd get_truncWeight() const {return truncWeight;};
	
	/**Returns the entropy when cut at site (Eigen array).*/
	ArrayXd get_entropy() const {return entropy;};
	///\}
	
private:
	
	size_t N_phys;
	
	/**local basis.*/
	vector<vector<qarray<Nq> > > qloc;
	
	qarray<Nq> Qtot = Symmetry::qvacuum();
	
	//*The Mps site-tensor.*/
	vector<vector<Biped<Symmetry,MatrixType> > > A; // access: A[l][s].block[q]
	ArrayXd truncWeight;
	ArrayXd entropy;
	
	// Bases on all ingoing and outgoing legs of the MPS
	vector<Qbasis<Symmetry> > inbase;
	vector<Qbasis<Symmetry> > outbase;
	
	vector<qarray<Nq> > QinTop;
	vector<qarray<Nq> > QinBot;
	vector<qarray<Nq> > QoutTop;
	vector<qarray<Nq> > QoutBot;
	
	void calc_Qlimits();
	
	void update_inbase (size_t loc);
	void update_outbase (size_t loc);
	void update_inbase () { for(size_t l=0; l<this->N_sites; l++) {update_inbase(l); } }
	void update_outbase () { for(size_t l=0; l<this->N_sites; l++) {update_outbase(l); } }

	void resize_arrays();
	void outerResizeNoSymm();
	
	// adds one site at a time in addScale, conserving memory
	template<typename OtherScalar> void add_site (size_t loc, OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin);
	
	// sweep stuff RICH_SVD
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
	ss << "trunc_weight=" << truncWeight.sum() << ", ";
	int lSmax;
	if (this->N_sites > 1)
	{
		entropy.maxCoeff(&lSmax);
		if (!std::isnan(entropy(lSmax)))
		{
			ss << "entropy(" << lSmax << ")=" << entropy(lSmax) << ", ";
		}
	}
	ss << "mem=" << round(memory(GB),3) << "GB";
	
//	ss << endl << " •ortho: " << test_ortho();
	return ss.str();
}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>::
Mps()
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar>>()
{}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>::
Mps (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input, int Qmax_input)
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >(L_input), qloc(qloc_input), Qtot(Qtot_input), N_phys(N_phys_input)
{
	outerResize(L_input, qloc_input, Qtot_input, Qmax_input);
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
Mps<Symmetry,Scalar>::
Mps (const Hamiltonian &H, size_t Dmax, qarray<Nq> Qtot_input, int Qmax_input)
:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >()
{
	N_phys = H.volume();
	outerResize(H.length(), H.locBasis(), Qtot_input, Qmax_input);
	update_inbase();
	update_outbase();
	innerResize(Dmax);
	update_inbase();
	update_outbase();
}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>::
Mps (size_t L_input, const vector<vector<Biped<Symmetry,MatrixXd> > > &As,
	 const vector<vector<qarray<Nq> > > &qloc_input, qarray<Nq> Qtot_input, size_t N_phys_input)
	:DmrgJanitor<PivotMatrix1<Symmetry,Scalar,Scalar> >(L_input), qloc(qloc_input), Qtot(Qtot_input), N_phys(N_phys_input), A(As)
{
	assert(As.size() == L_input and qloc_input.size() == L_input);
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
void Mps<Symmetry,Scalar>::
outerResize (const Hamiltonian &H, qarray<Nq> Qtot_input, int Qmax_input)
{
	N_phys = H.volume();
	outerResize(H.length(), H.locBasis(), Qtot_input, Qmax_input);
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
	
	inbase = V.inbase;
	outbase = V.outbase;
	
	QoutTop = V.QoutTop;
	QoutBot = V.QoutBot;
	QinTop  = V.QinTop;
	QinBot  = V.QinBot;
	
	A.resize(this->N_sites);
	
	truncWeight.resize(this->N_sites); truncWeight.setZero();
	entropy.resize(this->N_sites-1); entropy.setConstant(numeric_limits<double>::quiet_NaN());
	
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
	entropy.resize(this->N_sites-1); entropy.setConstant(numeric_limits<double>::quiet_NaN());
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
calc_Qlimits()
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
	
	QinTop[0] = Symmetry::qvacuum();
	QinBot[0] = Symmetry::qvacuum();
	for (size_t l=1; l<this->N_sites; ++l)
	{
		auto new_tops = Symmetry::reduceSilent(qloc[l], QinTop[l-1]);
		auto new_bots = Symmetry::reduceSilent(qloc[l], QinBot[l-1]);
		
		QinTop[l] = highest_q(new_tops);
		QinBot[l] = lowest_q(new_bots);
	}
	
	QoutTop[this->N_sites-1] = Qtot;
	QoutBot[this->N_sites-1] = Qtot;
	for (int l=this->N_sites-2; l>=0; --l)
	{
		vector<qarray<Symmetry::Nq> > qlocflip;
		for (size_t q=0; q<qloc[l].size(); ++q)
		{
			qlocflip.push_back(Symmetry::flip(qloc[l][q]));
		}
		auto new_tops = Symmetry::reduceSilent(qlocflip, QoutTop[l+1]);
		auto new_bots = Symmetry::reduceSilent(qlocflip, QoutBot[l+1]);
		
		QoutTop[l] = highest_q(new_tops);
		QoutBot[l] = lowest_q(new_bots);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (l!=0)
		{
			for (size_t q=0; q<Nq; q++)
			{
				QinTop[l][q] = min(QinTop[l][q], QoutTop[l-1][q]);
				QinBot[l][q]  = max(QinBot[l][q], QoutBot[l-1][q]);
				
			}
		}
		if (l!=this->N_sites-1)
		{
			for (size_t q=0; q<Nq; q++)
			{
				QoutTop[l][q] = min(QoutTop[l][q], QinTop[l+1][q]);
				QoutBot[l][q]  = max(QoutBot[l][q], QinBot[l+1][q]);
			}
		}
		
//		cout << "QinTop[l]=" << QinTop[l] << ", QinBot[l]=" << QinBot[l] << ", QoutTop[l]=" << QoutTop[l] << ", QoutBot[l]=" << QoutBot[l] << endl;
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
outerResize (size_t L_input, vector<vector<qarray<Nq> > > qloc_input, qarray<Nq> Qtot_input, int Nqmax_input)
{
//	cout << "Nqmax_input=" << Nqmax_input << endl;
	this->N_sites = L_input;
	qloc = qloc_input;
	Qtot = Qtot_input;
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
//				for (size_t q=0; q<Nq; q++)
//				{
//					if (abs(q1[q]-mean[q]) < abs(q2[q]-mean[q]))
//					{
//						return true;
//					}
//				}
//				return false;
				
//				cout << "loc=" << loc << endl;
				VectorXd dist_q1(Nq);
				VectorXd dist_q2(Nq);
				for (size_t q=0; q<Nq; q++)
				{
					double Delta = QinTop[loc][q] - QinBot[loc][q];
					dist_q1(q) = (q1[q]-mean[q]) / Delta;
					dist_q2(q) = (q2[q]-mean[q]) / Delta;
//					cout << "q1=" << q1 << ", dist=" << dist_q1(q) << ", mean[q]=" << mean[q] << ", Delta=" << Delta << endl;
//					cout << "q2=" << q2 << ", dist=" << dist_q2(q) << ", mean[q]=" << mean[q] << ", Delta=" << Delta << endl;
				}
//				cout << "dist_q1.norm()=" << dist_q1.norm() << ", dist_q2.norm()=" << dist_q2.norm() << endl;
//				cout << endl;
				
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
		array<double,Nq> mean;
		for (size_t q=0; q<Nq; q++)
		{
			mean[q] = Qtot[q]*l*1./this->N_sites;
		}
//		Qin_trunc[l] = take_first_elems(new_qs,mean,l);
		
		// check if within ranges (QinBot,QinTop) for all q:
		auto candidates = take_first_elems(new_qs,mean,l);
		for (const auto &candidate:candidates)
		{
			array<bool,Nq> WITHIN_RANGE;
			for (size_t q=0; q<Nq; ++q)
			{
				WITHIN_RANGE[q] = (candidate[q] <= QinTop[l][q] and candidate[q] >= QinBot[l][q]);
			}
			if (all_of(WITHIN_RANGE.begin(), WITHIN_RANGE.end(), [] (bool x) {return x;}))
			{
				Qin_trunc[l].push_back(candidate);
			}
		}
	}
	Qin_trunc[this->N_sites].push_back(Qtot);
	
	calc_Qlimits();
	
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
			A[l][s].in.push_back(qvacuum<Nq>());
			A[l][s].out.push_back(qvacuum<Nq>());
			A[l][s].dict.insert({qarray2<Nq>{qvacuum<Nq>(),qvacuum<Nq>()}, A[l][s].dim});
			A[l][s].dim = 1;
			A[l][s].block.resize(1);
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
innerResize (size_t Dmax)
{
	if constexpr (Nq == 0)
	{
		size_t Dl = qloc[0].size();
		size_t Dr = qloc[this->N_sites-1].size();
		
		for (size_t s=0; s<Dl; ++s)
		{
			A[0][s].block[0].resize(1,min(Dl,Dmax));
		}
		for (size_t s=0; s<Dr; ++s)
		{
			A[this->N_sites-1][s].block[0].resize(min(Dr,Dmax),1);
		}
		
		for (size_t l=1; l<this->N_sites/2; ++l)
		{
			size_t Dl = qloc[l].size();
			size_t Dr = qloc[this->N_sites-l-1].size();
			
			size_t Nlrows = min(Dmax, (size_t)A[l-1][0].block[0].cols());
			size_t Nlcols = min(Dmax, Nlrows*Dl);
			
			size_t Nrcols = min(Dmax, (size_t)A[this->N_sites-l][0].block[0].rows());
			size_t Nrrows = min(Dmax, Nrcols*Dr);
			
			for (size_t s=0; s<Dl; ++s)
			{
				A[l][s].block[0].resize(Nlrows,Nlcols);
			}
			for (size_t s=0; s<Dr; ++s)
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
			fromL[l][outbase[l-1][qout]] = min(fromL[l][outbase[l-1][qout]], Dmax);
		}
	
	//	cout << "LEFT: " << endl;
	//	for (int l=0; l<this->N_sites+1; ++l)
	//	{
	//		cout << "l=" << l << endl;
	//		for (auto it=fromL[l].begin(); it!=fromL[l].end(); ++it)
	//		{
	//			cout << "q=" << it->first << ": " << it->second << endl;
	//		}
	//	}
	//	cout << endl;
	
		fromR[this->N_sites].insert({Qtot,1});
		for (size_t l=this->N_sites; l-->0;)
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
			fromR[l][inbase[l][qin]] = min(fromR[l][inbase[l][qin]], Dmax);
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
			for (size_t s=0; s<qloc[l].size(); ++s)
			for (size_t q=0; q<A[l][s].dim; ++q)
			{
				qarray<Nq> Qin  = A[l][s].in[q];
				qarray<Nq> Qout = A[l][s].out[q];
				
				size_t Nrows = min(lrmin[l][Qin],    Dmax);
				size_t Ncols = min(lrmin[l+1][Qout], Dmax);
				A[l][s].block[q].resize(Nrows,Ncols);
			}
		}
	}
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
	Qtot = accumulate(config.begin(),config.end(),qvacuum<Nq>());
	this->pivot = -1;
	
	resize_arrays();
	
	vector<qarray<Nq> > qouts(this->N_sites+1);
	qouts[0] = Symmetry::qvacuum();
	for (size_t l=0; l<this->N_sites; ++l)
	{
		qouts[l+1] = accumulate(config.begin(), config.begin()+l+1, qvacuum<Nq>());
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
		A[l][s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
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
		A[loc][s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
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
save (string filename, string info)
{
	filename+=".h5";
	HDF5Interface target(filename, WRITE);
	
	string DmaxLabel = "Dmax";
	string eps_svdLabel = "eps_svd";
	string alpha_rsvdLabel = "alpha_rsvd";
	string add_infoLabel = "add_info";
	target.save_scalar(this->calc_Dmax(),DmaxLabel);
	target.save_scalar(this->min_Nsv,"min_Nsv");
	target.save_scalar(this->max_Nsv,"max_Nsv");
	target.save_scalar(this->eps_svd,eps_svdLabel);
	target.save_scalar(this->alpha_rsvd,alpha_rsvdLabel);
	target.save_char(info,add_infoLabel.c_str());
	
	std::string label;
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	{
		std::stringstream ss;
		ss << l << "_" << s << "_" << "(" << A[l][s].in[q] << "," << A[l][s].out[q] << ")";
		label = ss.str();
		target.save_matrix(A[l][s].block[q],label);
	}
}

template<typename Symmetry, typename Scalar>
size_t Mps<Symmetry,Scalar>::
loadDmax (string filename)
{
	filename+=".h5";
	HDF5Interface source(filename, READ);

	string DmaxLabel = "Dmax";
	size_t Dmax;
	source.load_scalar(Dmax,DmaxLabel);
	return Dmax;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
load (string filename)
{
	filename+=".h5";
	HDF5Interface source(filename, READ);
	
	string eps_svdLabel = "eps_svd";
	string alpha_rsvdLabel = "alpha_rsvd";
	source.load_scalar(this->eps_svd,eps_svdLabel);
	source.load_scalar(this->alpha_rsvd,alpha_rsvdLabel);
	source.load_scalar(this->min_Nsv,"min_Nsv");
	source.load_scalar(this->max_Nsv,"max_Nsv");
	
	std::string label;
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[l][s].dim; ++q)
	{
		std::stringstream ss;
		ss << l << "_" << s << "_" << "(" << A[l][s].in[q] << "," << A[l][s].out[q] << ")";
		label = ss.str();
		source.load_matrix(A[l][s].block[q], label);
	}
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
	
	vector<Biped<Symmetry,MatrixType> > Aloc;
	Aloc.resize(qloc[loc].size());
	vector<Biped<Symmetry,MatrixType> > Aprev; 
	if (loc != 0 and DISCARD_U == false)
	{
		Aprev.resize(qloc[loc-1].size());
	}
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
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
			for (size_t i=1; i<svec.size(); ++i)
			{
				assert(A[loc][svec[i]].block[qvec[i]].rows() == Nrows);
			}
			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
			MatrixType Aclump(Nrows,Ncols);
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[loc][svec[i]].block[qvec[i]]*
					                                        Symmetry::coeff_leftSweep(
					                                         A[loc][svec[i]].out[qvec[i]],
					                                         A[loc][svec[i]].in[qvec[i]],
					                                         qloc[loc][svec[i]]);
				stitch += Ncolsvec[i];
			}
			
			#ifdef DONT_USE_BDCSVD
			JacobiSVD<MatrixType> Jack; // standard SVD
			#else
			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
			#endif
			
			HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // QR
			
			size_t Nret = Nrows; // retained states
			
			if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
			{
				Jack.compute(Aclump,ComputeThinU|ComputeThinV);
				if (TOOL == DMRG::BROOM::BRUTAL_SVD)
				{
					Nret = min(static_cast<size_t>(Jack.singularValues().rows()), this->max_Nsv);
				}
				else
				{
					Nret = (Jack.singularValues().array() > this->eps_svd).count();
				}
//				Nret = min(max(Nret,this->min_Nsv),static_cast<size_t>(Jack.singularValues().rows()));
				Nret = max(Nret, this->min_Nsv);
				Nret = min(Nret, this->max_Nsv);
				truncWeightSub(qin) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
				
				// calculate entropy
				size_t Nnz = (Jack.singularValues().array() > 0.).count();
				entropySub(qin) = -(Jack.singularValues().head(Nnz).array().square() * Jack.singularValues().head(Nnz).array().square().log()).sum();
			}
			else if (TOOL == DMRG::BROOM::QR)
			{
				Quirinus.compute(Aclump.adjoint());
				Qmatrix = (Quirinus.householderQ() * MatrixType::Identity(Aclump.cols(),Aclump.rows())).adjoint();
				Rmatrix = (MatrixType::Identity(Aclump.rows(),Aclump.cols()) * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
			}
			
			if (Nret > 0)
			{
				// update A[loc]
				stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					MatrixType Mtmp;
					
					if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
					{
						Mtmp = Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i])*
								                         Symmetry::coeff_sign(
								                          A[loc][svec[i]].out[qvec[i]],
								                          A[loc][svec[i]].in[qvec[i]],
								                          qloc[loc][svec[i]]);
					}
					else if (TOOL == DMRG::BROOM::QR)
					{
						Mtmp = Qmatrix.block(0,stitch, Nrows,Ncolsvec[i])*
								                         Symmetry::coeff_sign(
								                          A[loc][svec[i]].out[qvec[i]],
								                          A[loc][svec[i]].in[qvec[i]],
								                          qloc[loc][svec[i]]);
					}
					
					if (Mtmp.size() != 0)
					{
						Aloc[svec[i]].push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Ncolsvec[i];
				}
				
				// update A[loc-1]
				if (loc != 0 and DISCARD_U == false)
				{
					for (size_t s=0; s<qloc[loc-1].size(); ++s)
					for (size_t q=0; q<A[loc-1][s].dim; ++q)
					{
						if (A[loc-1][s].out[q] == inbase[loc][qin])
						{
							MatrixType Mtmp;
							
							if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
							{
								Mtmp = A[loc-1][s].block[q] * 
								       Jack.matrixU().leftCols(Nret) * 
								       Jack.singularValues().head(Nret).asDiagonal();
							}
							else if (TOOL == DMRG::BROOM::QR)
							{
								Mtmp = A[loc-1][s].block[q] * Rmatrix;
							}
							
							auto it = Aprev[s].dict.find(qarray2<Nq>{A[loc-1][s].in[q], A[loc-1][s].out[q]});
							if (Mtmp.size() != 0)
							{
								Aprev[s].try_push_back(A[loc-1][s].in[q], A[loc-1][s].out[q], Mtmp);
							}
						}
					}
				}
			}
		}
	}
	
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
	
	if (TOOL != DMRG::BROOM::QR)
	{
		truncWeight(loc) = truncWeightSub.sum();
		int bond = (loc==0)? -1 : loc;
		if (bond != -1)
		{
			entropy(loc-1) = entropySub.sum();
		}
	}
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
	
	ArrayXd truncWeightSub(outbase[loc].size()); truncWeightSub.setZero();
	ArrayXd entropySub(outbase[loc].size()); entropySub.setZero();
	
	vector<Biped<Symmetry,MatrixType> > Aloc(qloc[loc].size());
	vector<Biped<Symmetry,MatrixType> > Anext; 
	if (loc != this->N_sites-1 and DISCARD_V == false)
	{
		Anext.resize(qloc[loc+1].size());
	}
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
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
			
			#ifdef DONT_USE_BDCSVD
			JacobiSVD<MatrixType> Jack; // standard SVD
			#else
			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only in Eigen available)
			#endif
			
			HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix; // Eigen QR
			
			size_t Nret = Ncols; // retained states
			
			if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
			{
				Jack.compute(Aclump,ComputeThinU|ComputeThinV);
				if (TOOL == DMRG::BROOM::BRUTAL_SVD)
				{
					Nret = min(static_cast<size_t>(Jack.singularValues().rows()), this->max_Nsv);
				}
				else
				{
					Nret = (Jack.singularValues().array() > this->eps_svd).count();
				}
//				Nret = min(max(Nret,this->min_Nsv),static_cast<size_t>(Jack.singularValues().rows()));
				Nret = max(Nret, this->min_Nsv);
				Nret = min(Nret, this->max_Nsv);
				truncWeightSub(qout) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
				
				// calculate entropy
				size_t Nnz = (Jack.singularValues().array() > 0.).count();
				entropySub(qout) = -(Jack.singularValues().head(Nnz).array().square() * Jack.singularValues().head(Nnz).array().square().log()).sum();
			}
			else if (TOOL == DMRG::BROOM::QR)
			{
				Quirinus.compute(Aclump);
				Qmatrix = Quirinus.householderQ() * MatrixType::Identity(Aclump.rows(),Aclump.cols());
				Rmatrix = MatrixType::Identity(Aclump.cols(),Aclump.rows()) * Quirinus.matrixQR().template triangularView<Upper>();
			}
			
			if (Nret > 0)
			{
				// update A[loc]
				stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					MatrixType Mtmp;
					if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
					{
						// A[loc][svec[i]].block[qvec[i]]
						Mtmp = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret);
					}
					else if (TOOL == DMRG::BROOM::QR)
					{
						Mtmp = Qmatrix.block(stitch,0, Nrowsvec[i],Ncols);
					}
					
					if (Mtmp.size() != 0)
					{
						Aloc[svec[i]].push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Nrowsvec[i];
				}
			
				// update A[loc+1]
				if (loc != this->N_sites-1 and DISCARD_V == false)
				{
					for (size_t s=0; s<qloc[loc+1].size(); ++s)
					for (size_t q=0; q<A[loc+1][s].dim; ++q)
					{
						if (A[loc+1][s].in[q] == outbase[loc][qout])
						{
							MatrixType Mtmp;
							
							if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
							{
								Mtmp = Jack.singularValues().head(Nret).asDiagonal() * 
									                   Jack.matrixV().adjoint().topRows(Nret) * 
									                   A[loc+1][s].block[q];
							}
							else if (TOOL == DMRG::BROOM::QR)
							{
								Mtmp = Rmatrix * A[loc+1][s].block[q];
							}
							
							auto it = Anext[s].dict.find(qarray2<Nq>{A[loc+1][s].in[q], A[loc+1][s].out[q]});
							if (Mtmp.size() != 0)
							{
								Anext[s].try_push_back(A[loc+1][s].in[q], A[loc+1][s].out[q], Mtmp);
							}
						}
					}
				}
			}
		}
	}
	
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
	
	if (TOOL != DMRG::BROOM::QR)
	{
		truncWeight(loc) = truncWeightSub.sum();
		int bond = (loc==this->N_sites-1)? -1 : loc;
		if (bond != -1)
		{
			entropy(loc) = entropySub.sum();
		}
	}
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
					                                            A[loc][svec[i]].in[qvec[i]],
					                                            qloc[loc][svec[i]]);
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
						                  Symmetry::coeff_sign(
						                  A[loc][svec[i]].out[qvec[i]],
						                  A[loc][svec[i]].in[qvec[i]],
						                  qloc[loc][svec[i]]);
						N[svec[i]].try_push_back(A[loc][svec[i]].in[qvec[i]], A[loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Ncolsvec[i];
				}
			}
		}
	}
	else if (DIR == DMRG::DIRECTION::RIGHT)
	{
		for (size_t qout=0; qout<outbase[loc].size(); ++qout)
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
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
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
			                                                                      A[loc][svec[i]].in[qvec[i]],
			                                                                      qloc[loc][svec[i]]);
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
			                                 Symmetry::coeff_sign(A[loc][svec[i]].out[qvec[i]],
			                                                      A[loc][svec[i]].in[qvec[i]],
			                                                      qloc[loc][svec[i]]);
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
	
	this->pivot = (loc==0)? 0 : loc-1;
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
rightSplitStep (size_t loc, Biped<Symmetry,MatrixType> &C)
{
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
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
sweepStep2 (DMRG::DIRECTION::OPTION DIR, size_t loc, const vector<Biped<Symmetry,MatrixType> > &Apair, bool DISCARD_SV)
{
	vector<qarray<Symmetry::Nq> > midset = calc_qsplit(A[loc], qloc[loc], A[loc+1], qloc[loc+1], QoutTop[loc], QoutBot[loc]);
	
	for (size_t s=0; s<qloc[loc].size(); ++s)
	{
		A[loc][s].clear();
	}
	for (size_t s=0; s<qloc[loc+1].size(); ++s)
	{
		A[loc+1][s].clear();
	}
	
	ArrayXd truncWeightSub(midset.size()); truncWeightSub.setZero();
	ArrayXd entropySub(midset.size()); entropySub.setZero();
	
	auto tensor_basis = Symmetry::tensorProd(qloc[loc], qloc[loc+1]);
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
	for (size_t qmid=0; qmid<midset.size(); ++qmid)
	{
		map<pair<size_t,qarray<Symmetry::Nq> >,vector<pair<size_t,qarray<Symmetry::Nq> > > > s13map;
		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<Scalar> > cgcmap;
		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<size_t> > q13map;
		map<tuple<size_t,qarray<Symmetry::Nq>,size_t,qarray<Symmetry::Nq> >,vector<size_t> > s1s3map;
		
		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
		for (size_t s3=0; s3<qloc[loc+1].size(); ++s3)
		{
			auto qmerges = Symmetry::reduceSilent(qloc[loc][s1], qloc[loc+1][s3]);
			
			for (const auto &qmerge:qmerges)
			{
				auto qtensor = make_tuple(qloc[loc][s1], s1, qloc[loc+1][s3], s3, qmerge);
				auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
				
				for (size_t q13=0; q13<Apair[s1s3].dim; ++q13)
				{
					auto qlmids = Symmetry::reduceSilent(Apair[s1s3].in[q13], qloc[loc][s1]);
					auto qrmids = Symmetry::reduceSilent(Apair[s1s3].out[q13], Symmetry::flip(qloc[loc+1][s3]));
					
					for (const auto &qlmid:qlmids)
					for (const auto &qrmid:qrmids)
					{
						if (qlmid == midset[qmid] and qrmid == midset[qmid])
						{
							s13map[make_pair(s1,Apair[s1s3].in[q13])].push_back(make_pair(s3,Apair[s1s3].out[q13]));
							
							Scalar factor_cgc = Symmetry::coeff_Apair(Apair[s1s3].in[q13], qloc[loc][s1], midset[qmid], 
							                                          qloc[loc+1][s3], Apair[s1s3].out[q13], qmerge);
							if (DIR==DMRG::DIRECTION::LEFT)
							{
								factor_cgc *= sqrt(Symmetry::coeff_rightOrtho(Apair[s1s3].out[q13], midset[qmid]));
							}
							
							cgcmap[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(factor_cgc);
							q13map[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(q13);
							s1s3map[make_tuple(s1,Apair[s1s3].in[q13],s3,Apair[s1s3].out[q13])].push_back(s1s3);
						}
					}
				}
			}
		}
		
		if (s13map.size() != 0)
		{
			map<pair<size_t,qarray<Symmetry::Nq> >,MatrixType> Aclumpvec;
			size_t istitch = 0;
			size_t jstitch = 0;
			vector<size_t> get_s3;
			vector<size_t> get_Ncols;
			vector<qarray<Symmetry::Nq> > get_qr;
			bool COLS_ARE_KNOWN = false;
			
			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
			{
				auto qls = Symmetry::reduceSilent(midset[qmid], Symmetry::flip(qloc[loc][s1]));
				
				for (const auto &ql:qls)
				{
					for (size_t s3=0; s3<qloc[loc+1].size(); ++s3)
					{
						auto qrs = Symmetry::reduceSilent(midset[qmid], qloc[loc+1][s3]);
						
						for (const auto &qr:qrs)
						{
							auto s3block = find(s13map[make_pair(s1,ql)].begin(), s13map[make_pair(s1,ql)].end(), make_pair(s3,qr));
							
							if (s3block != s13map[make_pair(s1,ql)].end())
							{
								MatrixType Mtmp;
								for (size_t i=0; i<q13map[make_tuple(s1,ql,s3,qr)].size(); ++i)
								{
									size_t q13 = q13map[make_tuple(s1,ql,s3,qr)][i];
									size_t s1s3 = s1s3map[make_tuple(s1,ql,s3,qr)][i];
									
									if (Mtmp.size() == 0)
									{
										Mtmp = cgcmap[make_tuple(s1,ql,s3,qr)][i] * Apair[s1s3].block[q13];
									}
									else if (Mtmp.size() > 0 and Apair[s1s3].block[q13].size() > 0)
									{
										Mtmp += cgcmap[make_tuple(s1,ql,s3,qr)][i] * Apair[s1s3].block[q13];
									}
								}
								if (Mtmp.size() == 0) {continue;}
								
								addRight(Mtmp, Aclumpvec[make_pair(s1,ql)]);
								
								if (COLS_ARE_KNOWN == false)
								{
									get_s3.push_back(s3);
									get_Ncols.push_back(Mtmp.cols());
									get_qr.push_back(qr);
								}
							}
						}
					}
					if (get_s3.size() != 0) {COLS_ARE_KNOWN = true;}
				}
			}
			
			vector<size_t> get_s1;
			vector<size_t> get_Nrows;
			vector<qarray<Symmetry::Nq> > get_ql;
			MatrixType Aclump;
			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
			{
				auto qls = Symmetry::reduceSilent(midset[qmid], Symmetry::flip(qloc[loc][s1]));
				
				for (const auto &ql:qls)
				{
					size_t Aclump_rows_old = Aclump.rows();
					
//					cout << "ql=" << ql << ", s1=" << qloc[loc][s1] << endl;
//					cout << Aclumpvec[make_pair(s1,ql)].rows() << "x" << Aclumpvec[make_pair(s1,ql)].cols() << endl;
//					cout << Aclump.rows() << "x" << Aclump.cols() << endl;
					
					// If cols don't match, it means that zeros were cut, restore them:
					if (Aclumpvec[make_pair(s1,ql)].cols() < Aclump.cols())
					{
						size_t dcols = Aclump.cols() - Aclumpvec[make_pair(s1,ql)].cols();
						Aclumpvec[make_pair(s1,ql)].conservativeResize(Aclumpvec[make_pair(s1,ql)].rows(), Aclump.cols());
						Aclumpvec[make_pair(s1,ql)].rightCols(dcols).setZero();
					}
					else if (Aclumpvec[make_pair(s1,ql)].cols() > Aclump.cols())
					{
						size_t dcols = Aclumpvec[make_pair(s1,ql)].cols() - Aclump.cols();
						Aclump.conservativeResize(Aclump.rows(), Aclump.cols()+dcols);
						Aclump.rightCols(dcols).setZero();
					}
					
					addBottom(Aclumpvec[make_pair(s1,ql)], Aclump);
					
					if (Aclump.rows() > Aclump_rows_old)
					{
						get_s1.push_back(s1);
						get_Nrows.push_back(Aclump.rows()-Aclump_rows_old);
						get_ql.push_back(ql);
					}
				}
			}
			if (Aclump.size() == 0)
			{
				if (DIR == DMRG::DIRECTION::RIGHT)
				{
					this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
				}
				else
				{
					this->pivot = (loc==0)? 0 : loc;
				}
				continue;
			}
			
			#ifdef DONT_USE_BDCSVD
			JacobiSVD<MatrixType> Jack; // standard SVD
			#else
			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
			#endif
			
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			
			// retained states:
			size_t Nret = Aclump.cols();
			Nret = (Jack.singularValues().array().abs() > this->eps_svd).count();
			Nret = max(Nret,this->min_Nsv);
			Nret = min(Nret,this->max_Nsv);
			
			truncWeightSub(qmid) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
			size_t Nnz = (Jack.singularValues().array() > 0.).count();
			entropySub(qmid) = -(Jack.singularValues().head(Nnz).array().square() * Jack.singularValues().head(Nnz).array().square().log()).sum();
			
			MatrixType Aleft, Aright;
			if (DIR == DMRG::DIRECTION::RIGHT)
			{
				Aleft = Jack.matrixU().leftCols(Nret);
				if (DISCARD_SV)
				{
					Aright = Jack.matrixV().adjoint().topRows(Nret);
				}
				else
				{
					Aright = Jack.singularValues().head(Nret).asDiagonal() * Jack.matrixV().adjoint().topRows(Nret);
				}
				this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
			}
			else
			{
				Aleft = Jack.matrixU().leftCols(Nret) * Jack.singularValues().head(Nret).asDiagonal();
				Aright = Jack.matrixV().adjoint().topRows(Nret);				
				this->pivot = (loc==0)? 0 : loc;
			}
			
			// update A[loc]
			istitch = 0;
			for (size_t i=0; i<get_s1.size(); ++i)
			{
				size_t s1 = get_s1[i];
				size_t Nrows = get_Nrows[i];
				
				qarray2<Nq> quple = {get_ql[i], midset[qmid]};
				auto q = A[loc][s1].dict.find(quple);
				if (q != A[loc][s1].dict.end())
				{
					A[loc][s1].block[q->second] += Aleft.block(istitch,0, Nrows,Nret);
				}
				else
				{
					A[loc][s1].push_back(get_ql[i], midset[qmid], Aleft.block(istitch,0, Nrows,Nret));
				}
				istitch += Nrows;
			}
			
			// update A[loc+1]
			jstitch = 0;
			for (size_t i=0; i<get_s3.size(); ++i)
			{
				size_t s3 = get_s3[i];
				size_t Ncols = get_Ncols[i];
				
				qarray2<Nq> quple = {midset[qmid], get_qr[i]};
				auto q = A[loc+1][s3].dict.find(quple);
				Scalar factor_cgc3 = (DIR==DMRG::DIRECTION::LEFT)? sqrt(Symmetry::coeff_rightOrtho(midset[qmid], get_qr[i])):1.;
				if (q != A[loc+1][s3].dict.end())
				{
					A[loc+1][s3].block[q->second] += factor_cgc3 * Aright.block(0,jstitch, Nret,Ncols);
				}
				else
				{
					A[loc+1][s3].push_back(midset[qmid], get_qr[i], factor_cgc3 * Aright.block(0,jstitch, Nret,Ncols));
				}
				jstitch += Ncols;
			}
		}
	}
	
	// remove unwanted zero-sized blocks
	for (size_t s=0; s<qloc[loc].size(); ++s)
	{
		A[loc][s]   = A[loc][s].cleaned();
		A[loc+1][s] = A[loc+1][s].cleaned();
	}
	
	update_outbase(loc);
	update_inbase(loc+1);
	
	truncWeight(loc) = truncWeightSub.sum();
	
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		int bond = (loc==this->N_sites-1)? -1 : loc;
		if (bond != -1)
		{
			entropy(loc) = entropySub.sum();
		}
	}
	else
	{
		int bond = (loc==0)? -1 : loc;
		if (bond != -1)
		{
			entropy(loc-1) = entropySub.sum();
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
enrich_left (size_t loc, PivotMatrix1<Symmetry,Scalar,Scalar> *H)
{
	if (this->alpha_rsvd > 0.)
	{
		std::vector<Biped<Symmetry,MatrixType> > P(qloc[loc].size());
		
		set<qarray<Nq> > Rmid_set;
		for (size_t qR=0; qR<H->R.size(); ++qR)
		{
			Rmid_set.insert(H->R.mid(qR));
		}
		
		Qbasis<Symmetry> QbasisR(Rmid_set, H->W[0][0][0].rows());
		Qbasis<Symmetry> QbasisOp;
		for (size_t k=0; k<H->qOp.size(); ++k)
		{
			QbasisOp.push_back(H->qOp[k],1);
		}
		auto QbasisW = QbasisR.add(QbasisOp);
		auto QbasisP = inbase[loc].combine(QbasisW);
		
		// create tensor P
		#ifndef DMRG_DONT_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
		for (size_t s2=0; s2<qloc[loc].size(); ++s2)
		for (size_t k=0; k<H->qOp.size(); ++k)
		{
			if (H->W[s1][s2][k].size() == 0) {continue;}
			for (size_t qR=0; qR<H->R.size(); ++qR)
			{
				auto qAs = Symmetry::reduceSilent(H->R.in(qR),Symmetry::flip(qloc[loc][s2]));
				for (const auto& qA : qAs)
				{
					qarray2<Symmetry::Nq> quple1 = {qA, H->R.in(qR)};
					auto itA = A[loc][s2].dict.find(quple1);
					
					if (itA != A[loc][s2].dict.end())
					{
						auto qWs = Symmetry::reduceSilent(H->R.mid(qR), Symmetry::flip(H->qOp[k]));
						
						for (const auto& qW : qWs)
						{
							auto qPs = Symmetry::reduceSilent(qA,qW);
							
							for (const auto& qP : qPs)
							{
								if (qP > QinTop[loc] or qP < QinBot[loc]) {continue;}
								
								Scalar factor_cgc = Symmetry::coeff_HPsi(A[loc][s2].out[itA->second], qloc[loc][s2], A[loc][s2].in[itA->second],
								                                         H->R.mid(qR), H->qOp[k], qW,
								                                         H->R.out(qR), qloc[loc][s1], qP);
								if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
								
								for (int spInd=0; spInd<H->W[s1][s2][k].outerSize(); ++spInd)
								for (typename SparseMatrix<Scalar>::InnerIterator iW(H->W[s1][s2][k],spInd); iW; ++iW)
								{
									size_t a = iW.row();
									size_t b = iW.col();
									size_t Prows = QbasisP.inner_dim(qP);
									size_t Pcols = H->R.block[qR][b][0].cols();
									size_t Arows = A[loc][s2].block[itA->second].rows();
									size_t stitch = QbasisP.leftAmount(qP,{qA,qW});
									
									MatrixType Mtmp(Prows,Pcols);
									Mtmp.setZero();
									
									if (stitch >= Prows) {continue;}
									
									if (H->R.block[qR][b][0].size() != 0)
									{
										Mtmp.block(stitch + a*Arows,0, Arows,Pcols) += (this->alpha_rsvd * 
										                                                factor_cgc * 
										                                                iW.value()) * 
										                                                A[loc][s2].block[itA->second] * 
										                                                H->R.block[qR][b][0];
									}
									
									// VectorXd norms = Mtmp.rowwise().norm();
//									vector<int> indices(Mtmp.rows());
//									iota(indices.begin(), indices.end(), 0);
									// sort(indices.begin(), indices.end(), [norms](int i, int j){return norms(i) > norms(j);});
									
									// int Nret = min(static_cast<int>(0.1*Prows),20);
									// Nret = max(Nret,1);
									// Nret = min(Mtmp.rows(), Nret);
									int Nret = (this->max_Nrich<0)? Mtmp.rows():
									                                min(static_cast<int>(Mtmp.rows()), this->max_Nrich);
									
//									cout << "Nret=" << Nret << ", Mtmp.rows()=" << Mtmp.rows() << endl;
//									MatrixType Mret(Nret,Mtmp.cols());
//									for (int i=0; i<Nret; ++i)
//									{
//										Mret.row(i) = Mtmp.row(indices[i]);
//									}
									Mtmp = Mtmp.topRows(Nret);
									
									if (Mtmp.size() != 0)
									{
										qarray2<Symmetry::Nq> qupleP = {qP, H->R.out(qR)};
										auto it = P[s1].dict.find(qupleP);
										if (it != P[s1].dict.end())
										{
											if (P[s1].block[it->second].rows() == 0)
											{
												P[s1].block[it->second] = Mtmp;
											}
											else
											{
												P[s1].block[it->second] += Mtmp;
											}
										}
										else
										{
											P[s1].push_back(qupleP, Mtmp);
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
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
		update_outbase(loc-1);
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
enrich_right (size_t loc, PivotMatrix1<Symmetry,Scalar,Scalar> *H)
{
	if (this->alpha_rsvd > 0.)
	{
		std::vector<Biped<Symmetry,MatrixType> > P(qloc[loc].size());
		
		set<qarray<Nq> > Lmid_set;
		for (size_t qL=0; qL<H->L.size(); ++qL)
		{
			Lmid_set.insert(H->L.mid(qL));
		}
		
		Qbasis<Symmetry> QbasisL(Lmid_set, H->W[0][0][0].cols());
		Qbasis<Symmetry> QbasisOp;
		for (size_t k=0; k<H->qOp.size(); ++k)
		{
			QbasisOp.push_back(H->qOp[k],1);
		}
		auto QbasisW = QbasisL.add(QbasisOp);
		auto QbasisP = outbase[loc].combine(QbasisW);
		
		// create tensor P
		#ifndef DMRG_DONT_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
		for (size_t s2=0; s2<qloc[loc].size(); ++s2)
		for (size_t k=0; k<H->qOp.size(); ++k)
		{
			if (H->W[s1][s2][k].size() == 0) {continue;}
			for (size_t qL=0; qL<H->L.size(); ++qL)
			{
				auto qAs = Symmetry::reduceSilent(H->L.out(qL),qloc[loc][s2]);
				for (const auto& qA : qAs)
				{
					qarray2<Symmetry::Nq> quple1 = {H->L.out(qL), qA};
					auto itA = A[loc][s2].dict.find(quple1);
					
					if (itA != A[loc][s2].dict.end())
					{
						auto qWs = Symmetry::reduceSilent(H->L.mid(qL), H->qOp[k]);
						
						for (const auto& qW : qWs)
						{
							auto qPs = Symmetry::reduceSilent(qA,qW);
							
							for (const auto& qP : qPs)
							{
								if (qP > QoutTop[loc] or qP < QoutBot[loc]) {continue;}
								
								Scalar factor_cgc = Symmetry::coeff_HPsi(A[loc][s2].out[itA->second], qloc[loc][s2], A[loc][s2].in[itA->second],
								                                         qW, H->qOp[k], H->L.mid(qL),
								                                         qP, qloc[loc][s1], H->L.in(qL));
								if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
								
								for (int spInd=0; spInd<H->W[s1][s2][k].outerSize(); ++spInd)
								for (typename SparseMatrix<Scalar>::InnerIterator iW(H->W[s1][s2][k],spInd); iW; ++iW)
								{
									size_t a = iW.row();
									size_t b = iW.col();
									
									size_t Prows = H->L.block[qL][a][0].rows();
									size_t Pcols = QbasisP.inner_dim(qP);
									size_t Acols = A[loc][s2].block[itA->second].cols();
									size_t stitch = QbasisP.leftAmount(qP,{qA,qW});
									
									MatrixType Mtmp(Prows,Pcols);
									Mtmp.setZero();
									
									if (stitch >= Pcols) {continue;}
									
									if (H->L.block[qL][a][0].rows() != 0 and
									    H->L.block[qL][a][0].cols() != 0)
									{
										Mtmp.block(0,stitch+b*Acols, Prows,Acols) += (this->alpha_rsvd * 
										                                             factor_cgc * 
										                                             iW.value()) * 
										                                             H->L.block[qL][a][0] * 
										                                             A[loc][s2].block[itA->second];
									}
									
									// VectorXd norms = Mtmp.colwise().norm();
//									vector<int> indices(Mtmp.cols());
//									iota(indices.begin(), indices.end(), 0);
									// sort(indices.begin(), indices.end(), [norms](int i, int j){return norms(i) > norms(j);});
//									
////									int Nret = min(static_cast<int>(0.1*Pcols),20);
////									Nret = max(Nret,1);
////									Nret = min(Mtmp.cols(), Nret);
									int Nret = (this->max_Nrich<0)? Mtmp.cols():
									                                min(static_cast<int>(Mtmp.cols()), this->max_Nrich);
									
//									MatrixType Mret(Mtmp.rows(),Nret);
//									for (int i=0; i<Nret; ++i)
//									{
//										Mret.col(i) = Mtmp.col(indices[i]);
//									}
//									Mtmp = Mret;
									Mtmp = Mtmp.leftCols(Nret);
									
									if (Mtmp.size() != 0)
									{
										qarray2<Symmetry::Nq> qupleP = {H->L.in(qL), qP};
										auto it = P[s1].dict.find(qupleP);
										if (it != P[s1].dict.end())
										{
											if (P[s1].block[it->second].rows() == 0)
											{
												P[s1].block[it->second] = Mtmp;
											}
											else
											{
												P[s1].block[it->second] += Mtmp;
											}
										}
										else
										{
											P[s1].push_back(qupleP, Mtmp);
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
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
		update_inbase(loc+1);
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
	
//	if (this->pivot != -1 and this->pivot == Vket.pivot and false)
//	{
//		Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > out = A[this->pivot][0].adjoint().contract(Vket.A[Vket.pivot][0]);
//		for (size_t s=1; s<qloc[this->pivot].size(); s++)
//		{
//			out += A[this->pivot][s].adjoint().contract(Vket.A[Vket.pivot][s]);
//		}
//		Scalar res = out.trace();
//		return res;
//	}
//	else
//	{
		Biped<Symmetry,Eigen::Matrix<Scalar,Dynamic,Dynamic> > Mtmp = A[0][0].adjoint().contract(Vket.A[0][0]);
		for (size_t s=1; s<qloc[0].size(); ++s)
		{
			Mtmp += A[0][s].adjoint().contract(Vket.A[0][s]);
		}
		Biped<Symmetry,Eigen::Matrix<Scalar,Dynamic,Dynamic> > Mout = Mtmp;
		
		for (size_t l=1; l<this->N_sites; ++l)
		{
			Mtmp = (A[l][0].adjoint() * Mout ).contract(Vket.A[l][0]);
			for (size_t s=1; s<qloc[l].size(); ++s)
			{
				Mtmp += (A[l][s].adjoint() * Mout).contract(Vket.A[l][s]);
			}
			Mout = Mtmp;
		}
		
		assert(Mout.dim == 1 and 
			   Mout.block[0].rows() == 1 and 
			   Mout.block[0].cols() == 1 and 
			   "Result of contraction in <φ|ψ> is not a scalar!");
		Scalar out = Mtmp.block[0](0,0);
		out *= Symmetry::coeff_dot(Qtot);
		return out;
//	}
	
//	assert(Mout.dim == 1 and 
//	       Mout.block[0].rows() == 1 and 
//	       Mout.block[0].cols() == 1 and 
//	       "Result of contraction in <φ|ψ> is not a scalar!");
	
//	return Mout.block[0](0,0);
	// return Mout.block[0].trace();
}

template<typename Symmetry, typename Scalar>
template<typename MpoScalar>
Scalar Mps<Symmetry,Scalar>::
locAvg (const Mpo<Symmetry,MpoScalar> &O) const
{
	assert(this->pivot != -1);
	Scalar res = 0.;
	
	for (size_t s1=0; s1<qloc[this->pivot].size(); ++s1)
	for (size_t s2=0; s2<qloc[this->pivot].size(); ++s2)
	for (size_t k=0; k<O.opBasis(this->pivot).size(); ++k)
	{
		Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Aprod = A[this->pivot][s1].adjoint() * A[this->pivot][s2];
		Scalar trace = 0.;
		for (size_t q=0; q<Aprod.dim; ++q)
		{
			trace += Aprod.block[q].trace();
		}
		
		for (int r=0; r<O.W_at(this->pivot)[s1][s2][k].outerSize(); ++r)
		for (typename SparseMatrix<MpoScalar>::InnerIterator iW(O.W_at(this->pivot)[s1][s2][k],r); iW; ++iW)
		{
			res += iW.value() * trace;
		}
	}
	
	return res;
}

// template<typename Symmetry, typename Scalar>
// template<typename MpoScalar>
// Scalar Mps<Symmetry,Scalar>::
// locAvg2 (const Mpo<Nq,MpoScalar> &O) const
// {
// 	assert(this->pivot != -1);
// 	Scalar res = 0;
	
// 	for (size_t s1=0; s1<qloc[this->pivot].size(); ++s1)
// 	for (size_t s2=0; s2<qloc[this->pivot].size(); ++s2)
// 	for (size_t s3=0; s3<qloc[this->pivot+1].size(); ++s3)
// 	for (size_t s4=0; s4<qloc[this->pivot+1].size(); ++s4)
// 	{
// 		for (int k12=0; k12<O.W_at(this->pivot)[s1][s2].outerSize(); ++k12)
// 		for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(O.W_at(this->pivot)[s1][s2],k12); iW12; ++iW12)
// 		for (int k34=0; k34<O.W_at(this->pivot+1)[s3][s4].outerSize(); ++k34)
// 		for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(O.W_at(this->pivot+1)[s3][s4],k34); iW34; ++iW34)
// 		{
// 			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Aprod12 = A[this->pivot][s1].adjoint() * A[this->pivot][s2];
// 			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Aprod123 = A[this->pivot+1][s3].adjoint() * Aprod12;
// 			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Aprod1234 = Aprod123 * A[this->pivot+1][s4];
			
// 			Scalar trace = 0;
// 			for (size_t q=0; q<Aprod1234.dim; ++q)
// 			{
// 				trace += Aprod1234.block[q].trace();
// 			}
			
// 			res += iW12.value() * iW34.value() * trace;
// 		}
// 	}
	
// 	return res;
// }

template<typename Symmetry, typename Scalar>
double Mps<Symmetry,Scalar>::
squaredNorm() const
{
	double res = 0.;
	// exploit canonical form:
	if (this->pivot != -1)
	{
		Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > out = A[this->pivot][0].adjoint().contract(A[this->pivot][0]);
		for (size_t s=1; s<qloc[this->pivot].size(); s++)
		{
			out += A[this->pivot][s].adjoint().contract(A[this->pivot][s]);
		}
		res = out.trace();
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
	assert(Qtot == V.Qtarget() and this->N_sites == V.length());
	
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
	std::swap(this->max_Nsv, V.max_Nsv);
	std::swap(this->min_Nsv, V.min_Nsv);
	std::swap(this->entropy, V.entropy);
	
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
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
Mps<Symmetry,OtherScalar> operator* (const OtherScalar &alpha, const Mps<Symmetry,Scalar> &Vin)
{
	Mps<Symmetry,OtherScalar> Vout = Vin.template cast<OtherScalar>();
	Vout *= alpha;
	return Vout;
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
	Vout.alpha_rsvd = this->alpha_rsvd;
	Vout.max_Nsv = this->max_Nsv;
	Vout.pivot = this->pivot;
	Vout.truncWeight = truncWeight;
	
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
}

template<typename Symmetry, typename Scalar>
Mps<Symmetry,Scalar>& Mps<Symmetry,Scalar>::
operator-= (const Mps<Symmetry,Scalar> &Vin)
{
	addScale(-1.,Vin);
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
void Mps<Symmetry,Scalar>::
add_site (size_t loc, OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin)
{
	if (loc == 0)
	{
		for (size_t s=0; s<qloc[0].size(); ++s)
		for (size_t q=0; q<A[0][s].dim; ++q)
		{
			qarray2<Nq> quple = {A[0][s].in[q], A[0][s].out[q]};
			auto it = Vin.A[0][s].dict.find(quple);
			addRight(alpha*Vin.A[0][s].block[it->second], A[0][s].block[q]);
		}
	}
	else if (loc == this->N_sites-1)
	{
		for (size_t s=0; s<qloc[this->N_sites-1].size(); ++s)
		for (size_t q=0; q<A[this->N_sites-1][s].dim; ++q)
		{
			qarray2<Nq> quple = {A[this->N_sites-1][s].in[q], A[this->N_sites-1][s].out[q]};
			auto it = Vin.A[this->N_sites-1][s].dict.find(quple);
			addBottom(Vin.A[this->N_sites-1][s].block[it->second], A[this->N_sites-1][s].block[q]);
		}
	}
	else
	{
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[loc][s].dim; ++q)
		{
			qarray2<Nq> quple = {A[loc][s].in[q], A[loc][s].out[q]};
			auto it = Vin.A[loc][s].dict.find(quple);
			addBottomRight(Vin.A[loc][s].block[it->second], A[loc][s].block[q]);
		}
	}
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
void Mps<Symmetry,Scalar>::
addScale (OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin, bool SVD_COMPRESS)
{
	assert(Qtot == Vin.Qtarget() and 
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
		if (SVD_COMPRESS == true)
		{
			rightSweepStep(0,DMRG::BROOM::SVD);
		}
		for (size_t l=2; l<this->N_sites; ++l)
		{
			add_site(l,alpha,Vin);
			if (SVD_COMPRESS == true)
			{
				rightSweepStep(l-1,DMRG::BROOM::SVD);
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Mps<Symmetry,Scalar>::
set_A_from_C (size_t loc, const vector<Tripod<Symmetry,MatrixType> > &C, DMRG::BROOM::OPTION TOOL)
{
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
////	outerResize(this->N_sites, qloc, accumulate(conf.begin(),conf.end(),qvacuum<Nq>()));
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
	
	for (size_t s=0; s<qloc[0].size(); ++s)
	for (size_t q=0; q<A[0][s].dim; ++q)
	{
		if (A[0][s].block[q].rows() != 1)
		{
			ss << name << " has wrong dimensions at: l=0: rows=" << A[0][s].block[q].rows() << " != 1" << endl;
		}
	}
	
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
	
	for (size_t s=0; s<qloc[this->N_sites-1].size(); ++s)
	for (size_t q=0; q<A[this->N_sites-1][s].dim; ++q)
	{
		if (A[this->N_sites-1][s].block[q].cols() != 1)
		{
			ss << name << " has wrong dimensions at: l=" << this->N_sites-1 << ": cols=" << A[this->N_sites-1][s].block[q].cols() << " != 1" << endl;
		}
	}
	
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
	string sout = "";
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
			sout += TCOLOR(MAGENTA);
			sout += (l==this->pivot) ? special_token[3] : normal_token[3]; // X
		}
		else if (all_of(A_CHECK.begin(),A_CHECK.end(),[](bool x){return x;}))
		{
			sout += TCOLOR(RED);
			sout += (l==this->pivot) ? special_token[0] : normal_token[0]; // A
		}
		else if (all_of(B_CHECK.begin(),B_CHECK.end(),[](bool x){return x;}))
		{
			sout += TCOLOR(BLUE);
			sout += (l==this->pivot) ? special_token[1] : normal_token[1]; // B
		}
		else
		{
			sout += TCOLOR(GREEN);
			sout += (l==this->pivot) ? special_token[2] : normal_token[2]; // M
		}
	}
	sout += TCOLOR(BLACK);
	return sout;
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
			os << V.A_at(l)[s].formatted();
			os << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != V.length()-1) {os << endl;}
	}
	return os;
}

#endif
