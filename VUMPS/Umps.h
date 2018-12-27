#ifndef VANILLA_Umps
#define VANILLA_Umps


/// \cond
#include <set>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <type_traits>
#include <iostream>
#include <fstream>
/// \endcond

#include "RandomVector.h" // from ALGS

#include "VUMPS/VumpsTypedefs.h"
#include "VumpsContractions.h"
#include "Blocker.h"
#include "VumpsTransferMatrixSF.h"
//include "PolychromaticConsole.h" // from TOOLS
//include "tensors/Biped.h"
//include "LanczosSolver.h" // from ALGS
//include "ArnoldiSolver.h" // from ALGS
//include "VUMPS/VumpsTransferMatrix.h"
//include "Mpo.h"
//include "tensors/DmrgConglutinations.h"

/**
 * \ingroup VUMPS
 * Uniform Matrix Product State. Analogue of the Mps class.
 * \describe_Symmetry
 * \describe_Scalar
 */
template<typename Symmetry, typename Scalar=double>
class Umps
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	static constexpr size_t Nq = Symmetry::Nq;
	
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class VumpsSolver;
	template<typename Symmetry_, typename S1, typename S2> friend class MpsCompressor;

public:
	
	/**Does nothing.*/
	Umps<Symmetry,Scalar>(){};
	
	/**Constructs a Umps with fixed bond dimension with the info from the Hamiltonian.*/
	template<typename Hamiltonian> Umps (const Hamiltonian &H, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax);

	/**Constructs a Umps with fixed bond dimension with a uniform given basis.*/
	Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax);

	/**Constructs a Umps with fixed bond dimension with a given basis.*/
	Umps (const vector<vector<qarray<Symmetry::Nq> > > &qloc_input, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax);

#ifdef USE_HDF5_STORAGE
	/**
	 * Construct from an external HDF5 file named <FILENAME>.h5.
	 * \param filename : The format is fixed to .h5, just enter the name without the format.
	 * \warning This method requires hdf5. For more information see https://www.hdfgroup.org/.
	 */
	Umps (string filename) {load(filename);}
#endif //USE_HDF5_STORAGE

	/**\describe_info*/
	string info() const;
	
	/**Prints a graph. See Mps::graph.*/
	void graph (string filename) const;
	
	/**Tests the orthogonality of the Umps.*/
	string test_ortho (double tol=1e-6) const;
	
	/**Sets all matrices  \f$A_L\f$, \f$A_R\f$, \f$A_C\f$, \f$C\f$) to random using boost's uniform distribution from -1 to 1.*/
	void setRandom();
	
	/**Normalizes the state, so that \f$Tr C^{\dagger} C = 1\f$*/
	void normalize_C();
	
	/**Resizes the bond dimension to \p Dmax and sets \p Nqmax blocks per site.*/
	void resize (size_t Dmax_input, size_t Nqmax_input);

	/**
	 * Determines all subspace quantum numbers and resizes the containers for the blocks. Memory for the matrices remains uninitiated. 
	 * Pulls info from another Mps.
	 * \param V : chain length, local basis and target quantum number will be equal to this umps
	 */
	template<typename OtherMatrixType> void resize (const Umps<Symmetry,OtherMatrixType> &V);

	/**Shorthand to resize all the relevant arrays: \p A, \p inbase, \p outbase, \p truncWeight, \p S.*/
	void resize_arrays();

	/**Calculates \f$A_L\f$ and \f$A_R\f$ from \f$A_C\f$ and \f$C\f$ at site \p loc using SVD (eqs. (19),(20)). 
	* This is supposed to be optimal, but not accurate.*/
	void svdDecompose (size_t loc, GAUGE::OPTION gauge = GAUGE::C);
	
	/**Calculates \f$A_L\f$ and \f$A_R\f$ from \f$A_C\f$ and \f$C\f$ at site \p loc using the polar decomposition (eq. (21),(22)). 
	* This is supposed to be non-optimal, but accurate.*/
	void polarDecompose (size_t loc, GAUGE::OPTION gauge = GAUGE::C);
	
	/**Returns the entropy for all sites.*/
	VectorXd entropy() const {return S;};
	
	/**Return the full entanglement spectrum, resolved by subspace quantum number.*/
	inline vector<map<qarray<Nq>,ArrayXd> > entanglementSpectrum() const {return SVspec;};
	
	/**Return the entanglement spectrum at the site \p loc (values all subspaces merged and sorted).*/
	ArrayXd entanglementSpectrumLoc (size_t loc) const;
	
	/**
	 * Casts the matrices from \p Scalar to \p OtherScalar.
	 */
	template<typename OtherScalar> Umps<Symmetry,OtherScalar> cast() const;
	
	/**
	 * Returns a real Umps containing the real part of this.
	 * \warning Does not check, whether the imaginary part is zero.
	 */
	Umps<Symmetry,double> real() const;
	
	/**Returns the local basis.*/
	inline vector<qarray<Symmetry::Nq> > locBasis (size_t loc) const {return qloc[loc];}
	inline vector<vector<qarray<Symmetry::Nq> > > locBasis()   const {return qloc;}
	
	/**Returns the ingoing basis.*/
	inline Qbasis<Symmetry> inBasis (size_t loc) const {return inbase[loc];}
	inline vector<Qbasis<Symmetry> > inBasis()   const {return inbase;}
	
	/**Returns the outgoing basis.*/
	inline Qbasis<Symmetry> outBasis (size_t loc) const {return outbase[loc];}
	inline vector<Qbasis<Symmetry> > outBasis()   const {return outbase;}
	
	/**Returns the amount of rows of first tensor. Useful for environment tensors in contractions.*/
	size_t get_frst_rows() const {return A[GAUGE::C][0][0].block[0].rows();}
	
	/**Returns the amount of columns of last tensor. Useful for environment tensors in contractions.*/
	size_t get_last_cols() const {return A[GAUGE::C][N_sites-1][0].block[0].cols();}
	
	/**Returns the amount of sites, i.e. the size of the unit cell.*/
	size_t length() const {return N_sites;}
	
	/**Calculates the left and right decomposition error as \f$\epsilon_L=\big|A_C-A_LC\big|^2\f$ and \f$\epsilon_R=\big|A_C-CA_R\big|^2\f$ (eq. (18)).*/
	Scalar calc_epsLRsq (GAUGE::OPTION gauge, size_t loc) const;

#ifdef USE_HDF5_STORAGE
	///\{
	/**
	 * Save all information of the Umps to the file <FILENAME>.h5.
	 * \param filename : The format is fixed to .h5, Just enter the name without the format.
	 * \param info : Additional information about the used model. Enter the info()-method of the used Mpo here.
	 * \warning This method requires hdf5. For more information see https://www.hdfgroup.org/.
	 * \note For the filename you should use the info string of the currently used Mpo.
	 */
	void save (string filename,string info="none");
	
	/**
	 * Reads all information of the Mps from the file <FILENAME>.h5.
	 * \param filename : the format is fixed to .h5. Just enter the name without the format.
	 * \warning This method requires hdf5. For more information visit https://www.hdfgroup.org/.
	 */
	void load (string filename);
#endif //USE_HDF5_STORAGE

	/**
	 * Determines the maximal bond dimension per site (sum of \p A.rows or \p A.cols over all subspaces).
	 */
	size_t calc_Mmax() const;
	
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
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit) const;
	
	/**
	 * Calculates the scalar product with another Umps by finding the dominant eigenvalue of the transfer matrix. 
	 * See arXiv:0804.2509 and Phys. Rev. B 78, 155117.
	 */
	double dot (const Umps<Symmetry,Scalar> &Vket) const;
	
	/**Returns \f$A_L\f$, \f$A_R\f$ or \f$A_C\f$ at site \p loc as const ref.*/
	const vector<Biped<Symmetry,MatrixType> > &A_at (GAUGE::OPTION g, size_t loc) const {return A[g][loc];};
	
	/**quantum number bounds for compatibility in contract_AA*/
	qarray<Symmetry::Nq> Qtop (size_t loc) const;
	qarray<Symmetry::Nq> Qbot (size_t loc) const;
	
	/**Safely calculates \f$l-1 mod L\f$ without overflow for \p size_t.*/
	inline size_t minus1modL (size_t l) const {return (l==0)? N_sites-1 : (l-1);}
	
	/**Returns the total quantum number of the Umps.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
		
	void calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,MatrixType> > &N) const;
	
	/**
	 * Performs a truncation of an Umps by the singular values of the center-matrix C.
	 * Updates AL and AR with the truncated isometries from the SVD and reorthogonalize them afterwards.
	 * \param SET_AC_RANDOM : bool to decide, whether to set AC to random or to C*AR.
	 * Truncation of a converged Umps can be done with \p SET_AC_RANDOM=false to evaluate e.g. observables after the truncation.
	 * Truncation during the variational optimization should be done with \p SET_AC_RANDOM=true because otherwise this would be a bias.
	 */
	void truncate(bool SET_AC_RANDOM=true);
	
	/**
	 * Orthogonalize the tensor with GAUGE \p g to be left-orthonormal using algorithm 2 from https://arxiv.org/abs/1810.07006.
	 * \param g : GAUGE to orthogonalize.
	 * \param G_L : Gauge-Transformation which performs the orthogonalization: \f$A[g] \rightarrow G_L*A[g]*G_L^{-1}\f$
	 * \note : Call this function with GAUGE::L to reorthogonalize AL.
	 */
	void orthogonalize_left (GAUGE::OPTION g, vector<Biped<Symmetry,MatrixType> > &G_L);
	/**
	 * Orthogonalize the tensor with GAUGE \p g to be right-orthonormal using the analogue to algorithm 2 from https://arxiv.org/abs/1810.07006.
	 * \param g : GAUGE to orthogonalize.
	 * \param G_R : Gauge-Transformation which performs the orthogonalization: \f$A[g] \rightarrow G_R^{-1}*A[g]*G_R\f$
	 * \note : Call this function with GAUGE::R to reorthogonalize AR.
	 */
	void orthogonalize_right(GAUGE::OPTION g, vector<Biped<Symmetry,MatrixType> > &G_R);
	
	/**
	 * Calculates either the right or the left fixed point of the transfer-matrix build up with A-tensors in GAUGE \p g.
	 * \param g : GAUGE to orthogonalize.
	 * \param DIR : LEFT or RIGHT fixed point.
	 * \note The return values are of type complex<double>. Can we choose them sometimes to be real?
	 */
	std::pair<complex<double>, Biped<Symmetry,Matrix<complex<double>,Dynamic,Dynamic> > > calc_dominant(GAUGE::OPTION g, DMRG::DIRECTION::OPTION DIR) const;
	
	/**
	 * This functions transforms all quantum numbers in the Umps (Umps::qloc and QN in Umps::A) by \f$q \rightarrow q * N_{cells}\f$.
	 * It is used for avg(Umps V, Mpo O, Umps V) in VumpsLinearAlgebra.h when O.length() > V.length(). 
	 * In this case the quantum numbers in the Umps are transformed in correspondence with V.length()
	 * and this is incompatible with the quantum numbers in O.length() which are transformed in correspondence to O.length().
	 * \param number_cells : \f$N_{cells}\f$
	 */
	void adjustQN (const size_t number_cells);
	
	/**
	 * Sorts the A tensors of a specific gauge. If SORT_ALL_GAUGES is true, then obviously all A tensors get sorted.
	 */
	void sort_A(size_t loc, GAUGE::OPTION g, bool SORT_ALL_GAUGES=false);
	
	/**
	 * Updates the tensor C with zeros if the auxiallary basis has changed, e.g. after an enrichment process
	 * \param loc : location of the C tensor for the update.
	 */
	void updateC(size_t loc);
	/**
	 * Updates the tensor AC with zeros if the auxiallary basis has changed, e.g. after an enrichment process
	 * \param loc : location of the C tensor for the update.
	 * \param g : Pull information about changed dimension from either A[GAUGE::L] or A[GAUGE::R]. 
	 * \warning Do not insert \p g = GAUGE::C here.
	 */
	void updateAC(size_t loc, GAUGE::OPTION g);
	
	/**
	 * Enlarges the tensors of the Umps with an enrichment tensor \p P and resizes everything necessary with zeros.
	 * The tensor \p P needs to be calculated in advance. This is done directly in the VumpsSolver.
	 * \param loc : location of the site to enrich.
	 * \param g : The gauge to enrich. L means, we need to update site tensor at loc+1 accordingly. R means updating site loc-1 with zeros.
	 * \param P : the tensor with the enrichment. It is calculated after Eq. (A31).
	 */
	void enrich(size_t loc, GAUGE::OPTION g, const vector<Biped<Symmetry,MatrixType> > &P);
	
	Array<complex<Scalar>,Dynamic,1> structure_factor (const Mpo<Symmetry,Scalar> &Oalfa, const Mpo<Symmetry,Scalar> &Obeta, 
	                                                   double kmin=0., double kmax=2.*M_PI, int kpoints=51, bool INFO=true);
	
private:
	
	/**parameter*/
	size_t N_sites;
	size_t Dmax, Nqmax;
	double eps_svd = 1e-13;
	size_t max_Nsv=100000ul, min_Nsv=1ul;
	int max_Nrich;
	
	qarray<Nq> Qtot;
	
	/**Calculate entropy at site \p loc.*/
	void calc_entropy (size_t loc, bool PRINT=false);
	
	/**Calculate entropy for all sites.*/
	void calc_entropy (bool PRINT=false) {for (size_t l=0; l<N_sites; ++l) calc_entropy(l,PRINT);};
	
	/**truncated weight*/
	ArrayXd truncWeight;
	
	/**local basis*/
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	
	/**A-tensors in the three gauges \p L, \p R, \p C*/
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> A; // A[L/R/C][l][s].block[q]	
		
	/**center matrix \p C*/
	vector<Biped<Symmetry,MatrixType> >                        C; // zero-site part C[l]
	
	/**null space (see eq. (25) and surrounding text)*/
	// std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> N; // N[L/R/C][l][s].block[q]
	
	VectorXd S;
	
	vector<map<qarray<Nq>,ArrayXd> > SVspec;
	
	/**bases on all ingoing and outgoing legs of the Umps*/
	vector<Qbasis<Symmetry> > inbase;
	vector<Qbasis<Symmetry> > outbase;
	
	/**update basis*/
	void update_inbase  (size_t loc, GAUGE::OPTION g = GAUGE::C);
	void update_outbase (size_t loc, GAUGE::OPTION g = GAUGE::C);
	void update_inbase  (GAUGE::OPTION g = GAUGE::C) {for (size_t l=0; l<this->N_sites; l++) {update_inbase (l,g);}}
	void update_outbase (GAUGE::OPTION g = GAUGE::C) {for (size_t l=0; l<this->N_sites; l++) {update_outbase(l,g);}}
};

template<typename Symmetry, typename Scalar>
string Umps<Symmetry,Scalar>::
info() const
{
	stringstream ss;
	ss << "Umps: ";
	ss << Symmetry::name() << ", ";
	ss << "Lcell=" << N_sites << ", ";
	ss << "Mmax=" << calc_Mmax() << " (";
	if (Symmetry::NON_ABELIAN)
	{
		ss << "full=" << calc_fullMmax() << ", ";
	}
	ss << "Dmax=" << calc_Dmax() << "), ";
	ss << "Nqmax=" << calc_Nqmax() << ", ";
	ss << "S=(" << S.transpose() << "), ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	
	return ss.str();
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
Umps<Symmetry,Scalar>::
Umps (const Hamiltonian &H, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax)
:N_sites(L_input), Qtot(Qtot_input)
{
	qloc = H.locBasis();
	resize(Dmax,Nqmax);
}

template<typename Symmetry, typename Scalar>
Umps<Symmetry,Scalar>::
Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax)
:N_sites(L_input), Qtot(Qtot_input)
{
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l) {qloc[l] = qloc_input;}
	resize(Dmax,Nqmax);
	::transform_base<Symmetry>(qloc,Qtot); // from DmrgExternal.h
}

template<typename Symmetry, typename Scalar>
Umps<Symmetry,Scalar>::
Umps (const vector<vector<qarray<Symmetry::Nq> > > &qloc_input, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax)
:N_sites(L_input), Qtot(Qtot_input)
{
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l) {qloc[l] = qloc_input[l];}
	resize(Dmax,Nqmax);
	::transform_base<Symmetry>(qloc,Qtot); // from DmrgExternal.h
}

template<typename Symmetry, typename Scalar>
double Umps<Symmetry,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<N_sites; ++l)
	{
		res += C[l].memory(memunit);
		for (size_t g=0; g<3; ++g)
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			res += A[g][l][s].memory(memunit);
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
size_t Umps<Symmetry,Scalar>::
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
size_t Umps<Symmetry,Scalar>::
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
size_t Umps<Symmetry,Scalar>::
calc_Mmax() const
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
std::size_t Umps<Symmetry,Scalar>::
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
void Umps<Symmetry,Scalar>::
update_inbase (size_t loc, GAUGE::OPTION g)
{
	inbase[loc].clear();
	inbase[loc].pullData(A[g][loc],0);
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
update_outbase (size_t loc, GAUGE::OPTION g)
{
	outbase[loc].clear();
	outbase[loc].pullData(A[g][loc],1);
}

template<typename Symmetry, typename Scalar>
qarray<Symmetry::Nq> Umps<Symmetry,Scalar>::
Qtop (size_t loc) const
{
	return qplusinf<Symmetry::Nq>();
}

template<typename Symmetry, typename Scalar>
qarray<Symmetry::Nq> Umps<Symmetry,Scalar>::
Qbot (size_t loc) const
{
	return qminusinf<Symmetry::Nq>();
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
resize_arrays ()
{
	truncWeight.resize(N_sites);
	for (size_t g=0; g<3; ++g)
	{
		A[g].resize(N_sites);
		for (size_t l=0; l<N_sites; ++l)
		{
			A[g][l].resize(qloc[l].size());
		}
		
		// N[g].resize(N_sites);
		// for (size_t l=0; l<N_sites; ++l)
		// {
		// 	N[g][l].resize(qloc[l].size());
		// }
	}
	C.resize(N_sites);
	inbase.resize(N_sites);
	outbase.resize(N_sites);
	S.resize(N_sites);
	SVspec.resize(N_sites);
}

template<typename Symmetry, typename Scalar>
template<typename OtherMatrixType> 
void Umps<Symmetry,Scalar>::
resize (const Umps<Symmetry,OtherMatrixType> &V)
{
	N_sites = V.N_sites;
	// N_phys = V.N_phys;
	qloc = V.qloc;
	Qtot = V.Qtot;
	
	inbase = V.inbase;
	outbase = V.outbase;
	
	for (size_t g=0; g<3; g++) {A[g].resize(this->N_sites);}
	C.resize(this->N_sites);
	
	truncWeight.resize(this->N_sites); truncWeight.setZero();
	S.resize(this->N_sites-1); S.setConstant(numeric_limits<double>::quiet_NaN());
	SVspec.resize(N_sites);
	
	for (size_t g=0; g<3; g++)
	for (size_t l=0; l<V.N_sites; ++l)
	{
		A[g][l].resize(qloc[l].size());
		
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			A[g][l][s].in = V.A[g][l][s].in;
			A[g][l][s].out = V.A[g][l][s].out;
			A[g][l][s].block.resize(V.A[g][l][s].dim);
			A[g][l][s].dict = V.A[g][l][s].dict;
			A[g][l][s].dim = V.A[g][l][s].dim;
		}
		C[l].in = V.C[l].in;
		C[l].out = V.C[l].out;
		C[l].block.resize(V.C[l].dim);
		C[l].dict = V.C[l].dict;
		C[l].dim = V.C[l].dim;
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
resize (size_t Dmax_input, size_t Nqmax_input)
{
	Dmax = Dmax_input;
	Nqmax = Nqmax_input;
	if      (Symmetry::IS_TRIVIAL) {Nqmax = 1;}
	else if (Symmetry::IS_MODULAR) {Nqmax = min(static_cast<size_t>(Symmetry::MOD_N),Nqmax_input);}
	
	resize_arrays();
	
	auto take_first_elems = [this] (const vector<qarray<Nq> > &qs) -> vector<qarray<Nq> >
	{
		vector<qarray<Nq> > out = qs;
		// sort the vector first according to the distance to qvacuum
		sort(out.begin(),out.end(),[this] (qarray<Nq> q1, qarray<Nq> q2)
		{
			VectorXd dist_q1(Nq);
			VectorXd dist_q2(Nq);
			for (size_t q=0; q<Nq; q++)
			{
				double Delta = 1.; // QinTop[loc][q] - QinBot[loc][q];
				dist_q1(q) = q1[q] / Delta;
				dist_q2(q) = q2[q] / Delta;
			}
			return (dist_q1.norm() < dist_q2.norm())? true:false;
		});
		return out;
	};
	
	vector<set<qarray<Symmetry::Nq> > > qinset(N_sites);
	vector<set<qarray<Symmetry::Nq> > > qoutset(N_sites);
	qoutset[N_sites-1].insert(Symmetry::qvacuum());
	ArrayXi inSize(N_sites); inSize = 0;
	ArrayXi outSize(N_sites); outSize = 0;
	
	while (not((inSize == Nqmax).all() and (outSize == Nqmax).all()))
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			size_t index = (l==0)? N_sites-1 : (l-1)%N_sites;
			for (const auto &t:qoutset[index])
			{
				if (qinset[l].size() < Nqmax)
				{
					qinset[l].insert(t);
				}
			}
			inSize[l] = qinset[l].size();
			
			vector<qarray<Symmetry::Nq> > qinvec(qinset[l].size());
			copy(qinset[l].begin(), qinset[l].end(), qinvec.begin());
			
			auto tmp = Symmetry::reduceSilent(qinvec, qloc[l], true);
			tmp = take_first_elems(tmp);
			for (const auto &t:tmp)
			{
				if (qoutset[l].size() < Nqmax)
				{
					qoutset[l].insert(t);
				}
			}
			outSize[l] = qoutset[l].size();
		}
	}
	
	// symmetrization
	if (Qtot == Symmetry::qvacuum())
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			auto qinset_tmp = qinset[l];
			for (const auto &q:qinset_tmp)
			{
				if (auto it=qinset_tmp.find(Symmetry::flip(q)); it==qinset_tmp.end())
				{
					qinset[l].insert(Symmetry::flip(q));
				}
			}
			
			auto qoutset_tmp = qoutset[l];
			for (const auto &q:qoutset_tmp)
			{
				if (auto it=qoutset_tmp.find(Symmetry::flip(q)); it==qoutset_tmp.end())
				{
					qoutset[l].insert(Symmetry::flip(q));
				}
			}
		}
	}
	
	MatrixXd Mtmp(Dmax,Dmax); Mtmp.setZero();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		vector<qarray<Symmetry::Nq> > qins(qinset[l].size());
		copy(qinset[l].begin(), qinset[l].end(), qins.begin());
		
		vector<qarray<Symmetry::Nq> > qouts(qoutset[l].size());
		copy(qoutset[l].begin(), qoutset[l].end(), qouts.begin());
		
		for (size_t g=0; g<3; ++g)
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			for (const auto &qin:qins)
			{
				auto qouts = Symmetry::reduceSilent(qloc[l][s], qin);
				for (const auto &qout:qouts)
				{
					if (auto it=qoutset[l].find(qout); it!=qoutset[l].end())
					{
						qarray2<Symmetry::Nq> qinout = {qin,qout};
						A[g][l][s].try_push_back(qinout, Mtmp);
					}
				}
			}
		}
	}
	
//	for (size_t g=0; g<3; ++g)
//	for (size_t l=0; l<N_sites; ++l)
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	for (size_t q=0; q<A[g][l][s].dim; ++q)
//	{
//		A[g][l][s].block[q].resize(Dmax,Dmax);
//	}
	
	update_inbase();
	update_outbase();
	
	for (size_t l=0; l<N_sites; ++l)
	for (size_t qout=0; qout<outbase[l].Nq(); ++qout)
	{
		C[l].try_push_back(qarray2<Symmetry::Nq>{outbase[l][qout], outbase[l][qout]}, Mtmp);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		C[l] = C[l].sorted();
	}	
	graph("init");
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
normalize_C()
{
	// normalize the centre matrices for proper wavefunction norm: Tr(C*C†)=1
	for (size_t l=0; l<N_sites; ++l)
	{
		C[l] = 1./sqrt((C[l].contract(C[l].adjoint())).trace()) * C[l];
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
setRandom()
{
	for (size_t l=0; l<N_sites; ++l)
	for (size_t q=0; q<C[l].dim; ++q)
	{
		for (size_t a1=0; a1<C[l].block[0].rows(); ++a1)
//		for (size_t a2=0; a2<C[l].block[0].cols(); ++a2)
		for (size_t a2=0; a2<=a1; ++a2)
		{
			C[l].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
			C[l].block[q](a2,a1) = C[l].block[q](a1,a2);
		}
	}
	
	normalize_C();
	
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
	for (size_t a1=0; a1<A[GAUGE::C][l][s].block[q].rows(); ++a1)
//	for (size_t a2=0; a2<A[GAUGE::C][l][s].block[q].cols(); ++a2)
	for (size_t a2=0; a2<=a1; ++a2)
	{
		A[GAUGE::C][l][s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		A[GAUGE::C][l][s].block[q](a2,a1) = A[GAUGE::C][l][s].block[q](a1,a2);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		svdDecompose(l);
	}
	
	calc_entropy();
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
graph (string filename) const
{
	stringstream ss;
	
	ss << "#!/usr/bin/dot dot -Tpdf -o " << filename << ".pdf\n\n";
	ss << "digraph G\n{\n";
	ss << "rankdir = LR;\n";
	ss << "labelloc=\"t\";\n";
	ss << "label=\"Umps: cell size=" << N_sites << ", Q=(";
	for (size_t q=0; q<Nq; ++q)
	{
		ss << Symmetry::kind()[q];
		if (q!=Nq-1) {ss << ",";}
	}
	ss << ")" << "\";\n";
	
	// first node
	ss << "\"l=" << 0 << ", " << Sym::format<Symmetry>(Symmetry::qvacuum()) << "\"";
	ss << "[label=" << "\"" << Sym::format<Symmetry>(Symmetry::qvacuum()) << "\"" << "];\n";
	
	// site nodes
	for (size_t l=0; l<N_sites; ++l)
	{
		ss << "subgraph" << " cluster_" << l << "\n{\n";
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
		{
			string qin  = Sym::format<Symmetry>(A[GAUGE::C][l][s].in[q]);
			ss << "\"l=" << l << ", " << qin << "\"";
			ss << "[label=" << "\"" << qin << "\"" << "];\n";
		}
		ss << "label=\"l=" << l << "\"\n";
		ss << "}\n";
	}
	
	// last node
	ss << "subgraph" << " cluster_" << N_sites << "\n{\n";
	for (size_t s=0; s<qloc[N_sites-1].size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][N_sites-1][s].dim; ++q)
	{
		string qout  = Sym::format<Symmetry>(A[GAUGE::C][N_sites-1][s].out[q]);
		ss << "\"l=" << N_sites << ", " << qout << "\"";
		ss << "[label=" << "\"" << qout << "\"" << "];\n";
	}
	ss << "label=\"l=" << N_sites << "=0" << "\"\n";
	ss << "}\n";
	
	// edges
	for (size_t l=0; l<N_sites; ++l)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
		{
			string qin  = Sym::format<Symmetry>(A[GAUGE::C][l][s].in[q]);
			string qout = Sym::format<Symmetry>(A[GAUGE::C][l][s].out[q]);
			ss << "\"l=" << l << ", " << qin << "\"";
			ss << "->";
			ss << "\"l=" << l+1 << ", " << qout << "\"";
			ss << " [label=\"" << A[GAUGE::C][l][s].block[q].rows() << "x" << A[GAUGE::C][l][s].block[q].cols() << "\"";
			ss << "];\n";
		}
	}
	
	ss << "\n}";
	
	ofstream f(filename+".dot");
	f << ss.str();
	f.close();
}

template<typename Symmetry, typename Scalar>
string Umps<Symmetry,Scalar>::
test_ortho (double tol) const
{
	stringstream sout;
	std::array<string,4> normal_token  = {"A","B","M","X"};
	std::array<string,4> special_token = {"\e[4mA\e[0m","\e[4mB\e[0m","\e[4mM\e[0m","\e[4mX\e[0m"};
	Array<Scalar,Dynamic,1> norm(N_sites);
	
	for (int l=0; l<N_sites; ++l)
	{
		// check for A
		Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Test = A[GAUGE::L][l][0].adjoint().contract(A[GAUGE::L][l][0]);
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[GAUGE::L][l][s].adjoint().contract(A[GAUGE::L][l][s]);
		}
		// cout << "AL test:" << endl << Test.print(true) << endl;
		vector<bool> A_CHECK(Test.dim);
		vector<double> A_infnorm(Test.dim);
		for (size_t q=0; q<Test.dim; ++q)
		{
			Test.block[q] -= MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
			A_CHECK[q]     = Test.block[q].norm()<tol ? true : false;
			A_infnorm[q]   = Test.block[q].norm();
			// cout << "q=" << Test.in[q] << ", A_infnorm[q]=" << A_infnorm[q] << endl;
		}
		
		// check for B
		Test.clear();
		Test = A[GAUGE::R][l][0].contract(A[GAUGE::R][l][0].adjoint(), contract::MODE::OORR);
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[GAUGE::R][l][s].contract(A[GAUGE::R][l][s].adjoint(), contract::MODE::OORR);
		}
		// cout << Test.print(true) << endl;
		vector<bool> B_CHECK(Test.dim);
		vector<double> B_infnorm(Test.dim);
		for (size_t q=0; q<Test.dim; ++q)
		{
			Test.block[q] -= MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
			B_CHECK[q]     = Test.block[q].template lpNorm<Infinity>()<tol ? true : false;
			B_infnorm[q]   = Test.block[q].template lpNorm<Infinity>();
//			cout << "q=" << Test.in[q] << ", B_infnorm[q]=" << B_infnorm[q] << endl;
		}
		
		// check for AL*C = AC
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Test = A[GAUGE::L][l][s] * C[l];
			for (size_t q=0; q<Test.dim; ++q)
			{
				qarray2<Symmetry::Nq> quple = {Test.in[q], Test.out[q]};
				auto it = A[GAUGE::C][l][s].dict.find(quple);
				if (it != A[GAUGE::C][l][s].dict.end())
				{
					Test.block[q] -= A[GAUGE::C][l][s].block[it->second];
				}
			}
			vector<double> T_CHECK(Test.dim);
//			cout << "Test.dim=" << Test.dim << endl;
			double normsum = 0;
			for (size_t q=0; q<Test.dim; ++q)
			{
//				cout << "g=L, " << "s=" << s << ", q=" << Test.in[q] << ", " << Test.out[q] 
//				     << ", norm=" << Test.block[q].template lpNorm<Infinity>() << endl;
				normsum += Test.block[q].norm();
				T_CHECK[q] = Test.block[q].norm()<tol ? true : false;
			}
			if (all_of(T_CHECK.begin(),T_CHECK.end(),[](bool x){return x;}))
			{
				cout << "l=" << l << ", s=" << s << ", AL[" << l << "]*C[" << l << "]=AC[" << l << "]=" 
				     << termcolor::green << "true" << termcolor::reset << ", normsum=" << normsum << endl;
			}
			else
			{
				cout << "l=" << l << ", s=" << s << ", AL[" << l << "]*C[" << l << "]=AC[" << l << "]=" 
				     << termcolor::red << "false" << termcolor::reset << ", normsum=" << normsum << endl;
			}
		}
		
		// check for C*AR = AC
		size_t locC = minus1modL(l);
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Test = C[locC] * A[GAUGE::R][l][s];
			for (size_t q=0; q<Test.dim; ++q)
			{
				qarray2<Symmetry::Nq> quple = {Test.in[q], Test.out[q]};
				auto it = A[GAUGE::C][l][s].dict.find(quple);
				if (it != A[GAUGE::C][l][s].dict.end())
				{
					Test.block[q] -= A[GAUGE::C][l][s].block[it->second];
				}
			}
			vector<double> T_CHECK(Test.dim);
//			cout << "Test.dim=" << Test.dim << endl;
			double normsum = 0;
			for (size_t q=0; q<Test.dim; ++q)
			{
//				cout << "g=R, " << "s=" << s << ", q=" << Test.in[q] << ", " << Test.out[q] 
//				     << ", norm=" << Test.block[q].template lpNorm<Infinity>() << endl;
				normsum += Test.block[q].template lpNorm<Infinity>();
				T_CHECK[q] = Test.block[q].template lpNorm<Infinity>()<tol ? true : false;
			}
			if (all_of(T_CHECK.begin(),T_CHECK.end(),[](bool x){return x;}))
			{
				cout << "l=" << l << ", s=" << s << ", C[" << locC << "]*AR[" << l << "]=AC[" << l << "]="
				     << termcolor::green << "true" << termcolor::reset << ", normsum=" << normsum << endl;
			}
			else
			{
				cout << "l=" << l << ", s=" << s << ", C[" << locC << "]*AR[" << l << "]=AC[" << l << "]="
				     << termcolor::red << "false" << termcolor::reset << ", normsum=" << normsum << endl;
			}
		}
				
		norm(l) = (C[l].contract(C[l].adjoint())).trace();
		
		// interpret result
		if (all_of(A_CHECK.begin(),A_CHECK.end(),[](bool x){return x;}))
		{
			sout << TCOLOR(RED);
			sout << normal_token[0]; // A
		}
		else
		{
			// assert(1!=1 and "AL is wrong");
			sout << TCOLOR(GREEN);
			sout << normal_token[2]; // M
		}
		
		if (all_of(B_CHECK.begin(),B_CHECK.end(),[](bool x){return x;}))
		{
			sout << TCOLOR(BLUE);
			sout << normal_token[1]; // B
		}
		else
		{
			// assert(1!=1 and "AR is wrong");
			sout << TCOLOR(GREEN);
			sout << normal_token[2]; // M
		}
	}
	
	sout << TCOLOR(BLACK);
	sout << ", norm=" << norm.transpose();
	return sout.str();
}

template<typename Symmetry, typename Scalar>
double Umps<Symmetry,Scalar>::
dot (const Umps<Symmetry,Scalar> &Vket) const
{
	double outL = calc_LReigen(VMPS::DIRECTION::LEFT,  A[GAUGE::L], Vket.A[GAUGE::L], outBasis(N_sites-1), Vket.outBasis(N_sites-1), qloc).energy;
	// for testing:
	double outR = calc_LReigen(VMPS::DIRECTION::RIGHT, A[GAUGE::R], Vket.A[GAUGE::R], inBasis(0), Vket.inBasis(0), qloc).energy;
	cout << "dot consistency check: from AL: " << outL << ", from AR: " << outR << endl;
	return outL;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_entropy (size_t loc, bool PRINT)
{
	S(loc) = 0;
	SVspec[loc].clear();
	
	if (PRINT)
	{
		lout << termcolor::magenta << "loc=" << loc << termcolor::reset << endl;
	}
	for (size_t q=0; q<C[loc].dim; ++q)
	{
		#ifdef DONT_USE_BDCSVD
		JacobiSVD<MatrixType> Jack; // standard SVD
		#else
		BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
		 #endif
		
		Jack.compute(C[loc].block[q], ComputeThinU|ComputeThinV);
//		Csingular[loc] += Jack.singularValues();
		
		size_t Nnz = (Jack.singularValues().array() > 0.).count();
		double Scontrib = -Symmetry::degeneracy(C[loc].in[q]) * 
		                   (Jack.singularValues().head(Nnz).array().square() * 
		                    Jack.singularValues().head(Nnz).array().square().log()
		                   ).sum();
		
		S(loc) += Scontrib;
		
		SVspec[loc].insert(pair<qarray<Symmetry::Nq>,ArrayXd>(C[loc].in[q],Jack.singularValues()));
		
		if (PRINT)
		{
			lout << termcolor::magenta << "S(" << C[loc].in[q] << "," 
			     << C[loc].out[q] << ")\t=\t" << Scontrib 
			     << ", size=" << C[loc].block[q].rows() << "x" << C[loc].block[q].cols() << termcolor::reset << endl;
		}
	}
	if (PRINT)
	{
		lout << endl;
	}
}

template<typename Symmetry, typename Scalar>
Scalar Umps<Symmetry,Scalar>::
calc_epsLRsq (GAUGE::OPTION gauge, size_t loc) const
{
	for (size_t s=0; s<qloc[loc].size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
	{
		std::array<qarray2<Symmetry::Nq>,3> quple;
		for (size_t g=0; g<3; ++g)
		{
			quple[g] = {A[g][loc][s].in[q], A[g][loc][s].out[q]};
		}
		
		for (size_t g=1; g<3; ++g)
		{
			assert(quple[0] == quple[g]);
		}
	}
	
	Scalar res = 0;
	
	if (gauge == GAUGE::L)
	{
		for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
		{
			qarray2<Symmetry::Nq> quple = {outbase[loc][qout], outbase[loc][qout]};
			auto it = C[loc].dict.find(quple);
			assert(it != C[loc].dict.end());
			size_t qC = it->second;
			
			// Determine how many A's to glue together
			vector<size_t> svec, qvec, Nrowsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
			{
				if (A[GAUGE::C][loc][s].out[q] == outbase[loc][qout])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Nrowsvec.push_back(A[GAUGE::C][loc][s].block[q].rows());
				}
			}
			
			// Do the glue
			size_t Ncols = A[GAUGE::C][loc][svec[0]].block[qvec[0]].cols();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].cols() == Ncols);}
			size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
			
			MatrixType Aclump(Nrows,Ncols);
			MatrixType Acmp(Nrows,Ncols);
			Aclump.setZero();
			Acmp.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
				Acmp.block  (stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::L][loc][svec[i]].block[qvec[i]];
				stitch += Nrowsvec[i];
			}
			
			res += (Aclump-Acmp*C[loc].block[qC]).squaredNorm() * Symmetry::coeff_dot(C[loc].in[qC]);
//			cout << "contrib L, l=" << loc << ", " << (Aclump-Acmp*C[loc].block[qC]).squaredNorm() * Symmetry::coeff_dot(C[loc].in[qC]) << endl;
		}
	}
	else if (gauge == GAUGE::R)
	{
		size_t locC = minus1modL(loc);
		
		for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
		{
			qarray2<Symmetry::Nq> quple = {inbase[loc][qin], inbase[loc][qin]};
			auto it = C[locC].dict.find(quple);
			assert(it != C[locC].dict.end());
			size_t qC = it->second;
			
			// Determine how many A's to glue together
			vector<size_t> svec, qvec, Ncolsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
			{
				if (A[GAUGE::C][loc][s].in[q] == inbase[loc][qin])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Ncolsvec.push_back(A[GAUGE::C][loc][s].block[q].cols());
				}
			}
			
			// Do the glue
			size_t Nrows = A[GAUGE::C][loc][svec[0]].block[qvec[0]].rows();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].rows() == Nrows);}
			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
			MatrixType Aclump(Nrows,Ncols);
			MatrixType Acmp(Nrows,Ncols);
			Aclump.setZero();
			Acmp.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::C][loc][svec[i]].block[qvec[i]]*
					                                        Symmetry::coeff_sign(
					                                         A[GAUGE::C][loc][svec[i]].out[qvec[i]],
					                                         A[GAUGE::C][loc][svec[i]].in[qvec[i]],
					                                         qloc[loc][svec[i]]);
				Acmp.block  (0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::R][loc][svec[i]].block[qvec[i]]*
					                                        Symmetry::coeff_sign(
					                                         A[GAUGE::R][loc][svec[i]].out[qvec[i]],
					                                         A[GAUGE::R][loc][svec[i]].in[qvec[i]],
					                                         qloc[loc][svec[i]]);
				stitch += Ncolsvec[i];
			}
			
			res += (Aclump-C[locC].block[qC]*Acmp).squaredNorm() * Symmetry::coeff_dot(C[locC].in[qC]);
//			cout << "contrib R=" << (Aclump-C[locC].block[qC]*Acmp).squaredNorm() * Symmetry::coeff_dot(C[locC].in[qC]) << endl;
		}
	}
	
	return res;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
polarDecompose (size_t loc, GAUGE::OPTION gauge)
{
	// check that blocks are the same for all gauges
	for (size_t s=0; s<qloc[loc].size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
	{
		std::array<qarray3<Symmetry::Nq>,3> quple;
		for (size_t g=0; g<3; ++g)
		{
			quple[g] = {A[g][loc][s].in[q], A[g][loc][s].out[q]};
		}
		
		for (size_t g=1; g<3; ++g)
		{
			assert(quple[0] == quple[g]);
		}
	}
	
	#ifdef DONT_USE_BDCSVD
	JacobiSVD<MatrixType> Jack; // standard SVD
	#else
	BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
	#endif
	
	if (gauge == GAUGE::L or gauge == GAUGE::C)
	{
//		S(loc) = 0;
		vector<MatrixType> UC(C[loc].dim);
		for (size_t q=0; q<C[loc].dim; ++q)
		{
			Jack.compute(C[loc].block[q], ComputeThinU|ComputeThinV);
			UC[q] = Jack.matrixU() * Jack.matrixV().adjoint();
		}
		
		for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
		{
			qarray2<Symmetry::Nq> quple = {outbase[loc][qout], outbase[loc][qout]};
			
			// Determine how many A's to glue together
			vector<size_t> svec, qvec, Nrowsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
			{
				if (A[GAUGE::C][loc][s].out[q] == outbase[loc][qout])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Nrowsvec.push_back(A[GAUGE::C][loc][s].block[q].rows());
				}
			}
			
			// Do the glue
			size_t Ncols = A[GAUGE::C][loc][svec[0]].block[qvec[0]].cols();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].cols() == Ncols);}
			size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
			
			MatrixType Aclump(Nrows,Ncols);
			Aclump.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
				stitch += Nrowsvec[i];
			}
			
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			MatrixType UL = Jack.matrixU() * Jack.matrixV().adjoint();
			
			auto it = C[loc].dict.find(quple);
			assert(it != C[loc].dict.end());
			size_t qC = it->second;
			
			// Update AL
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[GAUGE::L][loc][svec[i]].block[qvec[i]] = UL.block(stitch,0, Nrowsvec[i],Ncols) * UC[qC].adjoint();
				stitch += Nrowsvec[i];
			}
		}
	}
	
	size_t locC = minus1modL(loc);
	
	if (gauge == GAUGE::R or gauge == GAUGE::C)
	{
		vector<MatrixType> UC(C[locC].dim);
//		cout << "polarDecompose AR from C at " << locC << endl;
		
		for (size_t q=0; q<C[locC].dim; ++q)
		{
			Jack.compute(C[locC].block[q], ComputeThinU|ComputeThinV);
			UC[q] = Jack.matrixU() * Jack.matrixV().adjoint();
		}
		
		for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
		{
			qarray2<Symmetry::Nq> quple = {inbase[loc][qin], inbase[loc][qin]};
			
			// Determine how many A's to glue together
			vector<size_t> svec, qvec, Ncolsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
			{
				if (A[GAUGE::C][loc][s].in[q] == inbase[loc][qin])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Ncolsvec.push_back(A[GAUGE::C][loc][s].block[q].cols());
				}
			}
			
			// Do the glue
			size_t Nrows = A[GAUGE::C][loc][svec[0]].block[qvec[0]].rows();
			for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].rows() == Nrows);}
			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
			MatrixType Aclump(Nrows,Ncols);
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::C][loc][svec[i]].block[qvec[i]]*
					                                         Symmetry::coeff_leftSweep(
					                                          A[GAUGE::C][loc][svec[i]].out[qvec[i]],
					                                          A[GAUGE::C][loc][svec[i]].in[qvec[i]],
					                                          qloc[loc][svec[i]]);;
				stitch += Ncolsvec[i];
			}
			
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			MatrixType UR = Jack.matrixU() * Jack.matrixV().adjoint();
			
			auto it = C[locC].dict.find(quple);
			assert(it != C[locC].dict.end());
			size_t qC = it->second;
			
			// Update AR
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[GAUGE::R][loc][svec[i]].block[qvec[i]] = UC[qC].adjoint() * UR.block(0,stitch, Nrows,Ncolsvec[i])*
					                                       Symmetry::coeff_sign(
					                                        A[GAUGE::C][loc][svec[i]].out[qvec[i]],
					                                        A[GAUGE::C][loc][svec[i]].in[qvec[i]],
					                                        qloc[loc][svec[i]]);
				stitch += Ncolsvec[i];
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
svdDecompose (size_t loc, GAUGE::OPTION gauge)
{
	// check that blocks are the same for all gauges
	for (size_t s=0; s<qloc[loc].size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
	{
		std::array<qarray2<Symmetry::Nq>,3> quple;
		for (size_t g=0; g<3; ++g)
		{
			quple[g] = {A[g][loc][s].in[q], A[g][loc][s].out[q]};
		}
		
		for (size_t g=1; g<3; ++g)
		{
			assert(quple[0] == quple[g]);
		}
	}
	
	if (gauge == GAUGE::L or gauge == GAUGE::C)
	{
		ArrayXd truncWeightSub(outbase[loc].Nq()); truncWeightSub.setZero();

		vector<Biped<Symmetry,MatrixType> > Atmp(qloc[loc].size());
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			Atmp[s] = A[GAUGE::C][loc][s] * C[loc].adjoint();
		}
		
//		cout << "svdDecompose AL from C at " << loc << endl;
		for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
		{
			// qarray2<Symmetry::Nq> quple = {outbase[loc][qout], outbase[loc][qout]};
			// auto it = C[loc].dict.find(quple);
			// assert(it != C[loc].dict.end());
			// size_t qC = it->second;
			
			// Determine how many A's to glue together
			vector<size_t> svec, qvec, Nrowsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<Atmp[s].dim; ++q)
			{
				if (Atmp[s].out[q] == outbase[loc][qout])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Nrowsvec.push_back(Atmp[s].block[q].rows());
				}
			}
		
			// Do the glue
			size_t Ncols = Atmp[svec[0]].block[qvec[0]].cols();
			for (size_t i=1; i<svec.size(); ++i) {assert(Atmp[svec[i]].block[qvec[i]].cols() == Ncols);}
			size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
		
			MatrixType Aclump(Nrows,Ncols);
			Aclump.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(stitch,0, Nrowsvec[i],Ncols) = Atmp[svec[i]].block[qvec[i]];
				stitch += Nrowsvec[i];
			}
			
//			Aclump *= C[loc].block[qC].adjoint();
			
			#ifdef DONT_USE_BDCSVD
			JacobiSVD<MatrixType> Jack; // standard SVD
			#else
			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
			#endif
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			VectorXd SV = Jack.singularValues();
			
			//Here is probably the place for truncations of the Mps by taking Nret dependent on singluarValues() < eps_svd
			size_t Nret = Jack.singularValues().rows();
			// size_t Nret = (SV.array() > this->eps_svd).count();
			// Nret = max(Nret, this->min_Nsv);
			// Nret = min(Nret, this->max_Nsv);
			// cout << "L: Nret=" << Nret << ", full=" << Jack.singularValues().rows() << endl;
			// truncWeightSub(qout) = Symmetry::degeneracy(outbase[loc][qout]) * SV.tail(SV.rows()-Nret).cwiseAbs2().sum();

			// Update AL
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[GAUGE::L][loc][svec[i]].block[qvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret) * 
				                                           Jack.matrixV().adjoint().topRows(Nret);
				stitch += Nrowsvec[i];
			}
		}
		truncWeight(loc) = truncWeightSub.sum();
	}
	
	size_t locC = minus1modL(loc);
	
	if (gauge == GAUGE::R or gauge == GAUGE::C)
	{
		ArrayXd truncWeightSub(inbase[loc].Nq()); truncWeightSub.setZero();
		vector<Biped<Symmetry,MatrixType> > Atmp(qloc[loc].size());
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			// Atmp[s] = C[locC].adjoint().contract(A[GAUGE::C][loc][s]);
			Atmp[s] = C[locC].adjoint() * A[GAUGE::C][loc][s];
		}
		
//		cout << "svdDecompose AR from C at " << locC << endl;
		
		for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
		{
//			qarray2<Symmetry::Nq> quple = {inbase[loc][qin], inbase[loc][qin]};
//			auto it = C[locC].dict.find(quple);
//			assert(it != C[locC].dict.end());
//			size_t qC = it->second;
			
			// Determine how many A's to glue together
			vector<size_t> svec, qvec, Ncolsvec;
			for (size_t s=0; s<qloc[loc].size(); ++s)
			for (size_t q=0; q<Atmp[s].dim; ++q)
			{
				if (Atmp[s].in[q] == inbase[loc][qin])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Ncolsvec.push_back(Atmp[s].block[q].cols());
				}
			}
			
			// Do the glue
			size_t Nrows = Atmp[svec[0]].block[qvec[0]].rows();
			for (size_t i=1; i<svec.size(); ++i) {assert(Atmp[svec[i]].block[qvec[i]].rows() == Nrows);}
			size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
			
			MatrixType Aclump(Nrows,Ncols);
			Aclump.setZero();
			size_t stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = Atmp[svec[i]].block[qvec[i]]*
				                                            Symmetry::coeff_leftSweep(
				                                             Atmp[svec[i]].out[qvec[i]],
				                                             Atmp[svec[i]].in[qvec[i]],
				                                             qloc[loc][svec[i]]);
				stitch += Ncolsvec[i];
			}
			
//			Aclump = C[locC].block[qC].adjoint() * Aclump;
			
			#ifdef DONT_USE_BDCSVD
			JacobiSVD<MatrixType> Jack; // standard SVD
			#else
			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
			#endif
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			VectorXd SV = Jack.singularValues();
			
			//Here is probably the place for truncations of the Mps by taking Nret dependent on singluarValues() < eps_svd
			size_t Nret = Jack.singularValues().rows();
			// size_t Nret = (SV.array() > this->eps_svd).count();
			// Nret = max(Nret, this->min_Nsv);
			// Nret = min(Nret, this->max_Nsv);
			// cout << "R: Nret=" << Nret << ", full=" << Jack.singularValues().rows() << endl;
			// truncWeightSub(qin) = Symmetry::degeneracy(inbase[loc][qin]) * SV.tail(SV.rows()-Nret).cwiseAbs2().sum();

			// Update AR
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[GAUGE::R][loc][svec[i]].block[qvec[i]] = Jack.matrixU().leftCols(Nret) * 
				                                           Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i])
				                                           *
				                                           Symmetry::coeff_sign(
				                                            Atmp[svec[i]].out[qvec[i]],
				                                            Atmp[svec[i]].in[qvec[i]],
				                                            qloc[loc][svec[i]]);
				stitch += Ncolsvec[i];
			}
		}
		truncWeight(loc) = truncWeightSub.sum();
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &N) const
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
			for (size_t q=0; q<A[GAUGE::R][loc][s].dim; ++q)
			{
				if (A[GAUGE::R][loc][s].in[q] == inbase[loc][qin])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Ncolsvec.push_back(A[GAUGE::R][loc][s].block[q].cols());
				}
			}
			
			if (Ncolsvec.size() > 0)
			{
				// do the glue
				size_t Nrows = A[GAUGE::R][loc][svec[0]].block[qvec[0]].rows();
				for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::R][loc][svec[i]].block[qvec[i]].rows() == Nrows);}
				size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
				
				MatrixType Aclump(Nrows,Ncols);
				size_t stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::R][loc][svec[i]].block[qvec[i]]*
					                                            Symmetry::coeff_leftSweep(
					                                            A[GAUGE::R][loc][svec[i]].out[qvec[i]],
					                                            A[GAUGE::R][loc][svec[i]].in[qvec[i]],
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
						                  A[GAUGE::R][loc][svec[i]].out[qvec[i]],
						                  A[GAUGE::R][loc][svec[i]].in[qvec[i]],
						                  qloc[loc][svec[i]]);
						N[svec[i]].try_push_back(A[GAUGE::R][loc][svec[i]].in[qvec[i]], A[GAUGE::R][loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Ncolsvec[i];
				}
			}
		}

		Qbasis<Symmetry> qloc_(qloc[loc]);
		Qbasis<Symmetry> qcomb = outbase[loc].combine(qloc_,true);

		for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			auto qfulls = Symmetry::reduceSilent(outbase[loc][qout], Symmetry::flip(qloc[loc][s]));
			for (const auto &qfull:qfulls)
			{
				qarray2<Symmetry::Nq> quple = {qfull,outbase[loc][qout]};
				auto it = A[GAUGE::R][loc][s].dict.find(quple);
				if (it == A[GAUGE::R][loc][s].dict.end())
				{
					MatrixType Mtmp(qcomb.inner_dim(qfull), outbase[loc].inner_dim(outbase[loc][qout]));
					Mtmp.setZero();
					Index down=qcomb.leftAmount(qfull,{outbase[loc][qout], Symmetry::flip(qloc[loc][s])});

					size_t source_dim;
					auto it = qcomb.history.find(qfull);
					for (size_t i=0; i<(it->second).size(); i++)
					{
						if ((it->second)[i].source == qarray2<Nq>{outbase[loc][qout], Symmetry::flip(qloc[loc][s])})
						{
							source_dim = (it->second)[i].dim;
						}
					}
					Mtmp.block(down,0,source_dim,outbase[loc].inner_dim(outbase[loc][qout])).setIdentity();
					Mtmp.block(down,0,source_dim,outbase[loc].inner_dim(outbase[loc][qout])) *= Symmetry::coeff_sign( outbase[loc][qout], qfull, qloc[loc][s] );
					// Mtmp.setIdentity();
					// cout << "push_back with q=" << quple[0] << "," << quple[1] << endl;
					N[s].push_back(quple, Mtmp);

					// Mtmp.setIdentity();
					// Mtmp *= Symmetry::coeff_sign( outbase[loc][qout], qfull, qloc[loc][s] );
					// N[s].push_back(quple, Mtmp);
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
			for (size_t q=0; q<A[GAUGE::L][loc][s].dim; ++q)
			{
				if (A[GAUGE::L][loc][s].out[q] == outbase[loc][qout])
				{
					svec.push_back(s);
					qvec.push_back(q);
					Nrowsvec.push_back(A[GAUGE::L][loc][s].block[q].rows());
				}
			}
			
			if (Nrowsvec.size() > 0)
			{
				// do the glue
				size_t Ncols = A[GAUGE::L][loc][svec[0]].block[qvec[0]].cols();
				for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::L][loc][svec[i]].block[qvec[i]].cols() == Ncols);}
				size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
				
				MatrixType Aclump(Nrows,Ncols);
				Aclump.setZero();
				size_t stitch = 0;
				for (size_t i=0; i<svec.size(); ++i)
				{
					Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::L][loc][svec[i]].block[qvec[i]];
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
						MatrixType Mtmp = Qmatrix.block(stitch,Nret, Nrowsvec[i],Nnull);
						N[svec[i]].try_push_back(A[GAUGE::L][loc][svec[i]].in[qvec[i]], A[GAUGE::L][loc][svec[i]].out[qvec[i]], Mtmp);
					}
					stitch += Nrowsvec[i];
				}
			}
		}
		
		Qbasis<Symmetry> qloc_(qloc[loc]);
		Qbasis<Symmetry> qcomb = inbase[loc].combine(qloc_);

		for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			auto qfulls = Symmetry::reduceSilent(inbase[loc][qin], qloc[loc][s]);
			for (const auto &qfull:qfulls)
			{
				qarray2<Symmetry::Nq> quple = {inbase[loc][qin], qfull};
				auto it = A[GAUGE::L][loc][s].dict.find(quple);
				if (it == A[GAUGE::L][loc][s].dict.end())
				{
					MatrixType Mtmp(inbase[loc].inner_dim(inbase[loc][qin]), qcomb.inner_dim(qfull));
					Mtmp.setZero();
					Index left=qcomb.leftAmount(qfull,{inbase[loc][qin], qloc[loc][s]});

					size_t source_dim;
					auto it = qcomb.history.find(qfull);
					for (size_t i=0; i<(it->second).size(); i++)
					{
						if ((it->second)[i].source == qarray2<Nq>{inbase[loc][qin], qloc[loc][s]})
						{
							source_dim = (it->second)[i].dim;
						}
					}
					Mtmp.block(0,left,inbase[loc].inner_dim(inbase[loc][qin]),source_dim).setIdentity();
					// Mtmp.setIdentity();
					N[s].push_back(quple, Mtmp);
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
Umps<Symmetry,OtherScalar> Umps<Symmetry,Scalar>::
cast() const
{
	Umps<Symmetry,OtherScalar> Vout;
	Vout.resize(*this);
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[g][l][s].dim; ++q)
	{
		Vout.A[g][l][s].block[q] = A[g][l][s].block[q].template cast<OtherScalar>();
	}
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t q=0; q<C[l].dim; ++q)
	{
		Vout.C[l].block[q] = C[l].block[q].template cast<OtherScalar>();
	}
	Vout.eps_svd = this->eps_svd;
	Vout.truncWeight = truncWeight;
	
	return Vout;
}

template<typename Symmetry, typename Scalar>
Umps<Symmetry,double> Umps<Symmetry,Scalar>::
real() const
{
	Umps<Symmetry,double> Vout;
	Vout.resize(*this);
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[g][l][s].dim; ++q)
	{
		Vout.A[g][l][s].block[q] = A[g][l][s].block[q].real();
	}
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t q=0; q<C[l].dim; ++q)
	{
		Vout.C[l].block[q] = C[l].block[q].real();
	}
	Vout.eps_svd = this->eps_svd;
	Vout.truncWeight = truncWeight;
	
	return Vout;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
adjustQN (const size_t number_cells)
{
	//transform quantum number in all Bipeds
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[g][l][s] = A[g][l][s].adjustQN(number_cells);
	}

	//transform physical quantum numbers
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		qloc[l][s] = ::adjustQN<Symmetry>(qloc[l][s],number_cells);
	}
	update_inbase();
	update_outbase();
};

#ifdef USE_HDF5_STORAGE
template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
save (string filename, string info)
{
	filename+=".h5";
	HDF5Interface target(filename, WRITE);
	target.create_group("As");
	target.create_group("Cs");
	target.create_group("qloc");
	target.create_group("Qtot");
	
	string add_infoLabel = "add_info";

	//save scalar values
	target.save_scalar(this->N_sites,"L");
	for (size_t q=0; q<Nq; q++)
	{
		stringstream ss; ss << "q=" << q;
		target.save_scalar(this->Qtot[q],ss.str(),"Qtot");
	}
	target.save_scalar(this->calc_Dmax(),"Dmax");
	target.save_scalar(this->calc_Nqmax(),"Nqmax");	
	target.save_scalar(this->min_Nsv,"min_Nsv");
	target.save_scalar(this->max_Nsv,"max_Nsv");
	target.save_scalar(this->eps_svd,"eps_svd");
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
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		stringstream tt; tt << "g=" << g << ",l=" << l << ",s=" << s;
		target.save_scalar(A[g][l][s].dim,tt.str());
		for (size_t q=0; q<A[g][l][s].dim; ++q)
		{
			for (size_t p=0; p<Nq; p++)
			{
				stringstream in; in << "in,g=" << g << ",l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				stringstream out; out << "out,g=" << g << ",l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				target.save_scalar((A[g][l][s].in[q])[p],in.str(),"As");
				target.save_scalar((A[g][l][s].out[q])[p],out.str(),"As");
			}
			stringstream ss;			
			ss << g << "_" << l << "_" << s << "_" << "(" << A[g][l][s].in[q] << "," << A[g][l][s].out[q] << ")";
			label = ss.str();
			target.save_matrix(A[g][l][s].block[q],label,"As");
		}
	}

	//save the C-matrices
	for (size_t l=0; l<this->N_sites; ++l)
	{
		stringstream tt; tt << "l=" << l;
		target.save_scalar(C[l].dim,tt.str());
		for (size_t q=0; q<C[l].dim; ++q)
		{
			for (size_t p=0; p<Nq; p++)
			{
				stringstream in; in << "l=" << l << ",q=" << q << ",p=" << p;
				target.save_scalar((C[l].in[q])[p],in.str(),"Cs");
			}
			stringstream ss;			
			ss << l << "_"  << "(" << C[l].in[q] << ")";
			label = ss.str();
			target.save_matrix(C[l].block[q],label,"Cs");			
		}
	}
	target.close();
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
load (string filename)
{
	filename+=".h5";
	HDF5Interface source(filename, READ);
	
	//load the scalars
	source.load_scalar(this->N_sites,"L");
	for (size_t q=0; q<Nq; q++)
	{
		stringstream ss; ss << "q=" << q;
		source.load_scalar(this->Qtot[q],ss.str(),"Qtot");
	}
	source.load_scalar(this->eps_svd,"eps_svd");
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
	this->resize_arrays();

	//load the A-matrices
	string label;
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		size_t Asize;
		stringstream tt; tt << "g=" << g << ",l=" << l << ",s=" << s;
		source.load_scalar(Asize,tt.str());
		for (size_t q=0; q<Asize; ++q)
		{
			qarray<Nq> qin,qout;
			for (size_t p=0; p<Nq; p++)
			{
				stringstream in; in << "in,g=" << g << ",l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				stringstream out; out << "out,g=" << g << ",l=" << l << ",s=" << s << ",q=" << q << ",p=" << p;
				source.load_scalar(qin[p],in.str(),"As");
				source.load_scalar(qout[p],out.str(),"As");
			}
			stringstream ss;
			ss << g << "_" << l << "_" << s << "_" << "(" << qin << "," << qout << ")";
			label = ss.str();
			MatrixType mat;
			source.load_matrix(mat, label, "As");
			A[g][l][s].push_back(qin,qout,mat);
		}
	}

	//load the C-matrices
	label.clear();
	for (size_t l=0; l<this->N_sites; ++l)
	{
		size_t Asize;
		stringstream tt; tt << "l=" << l;
		source.load_scalar(Asize,tt.str());
		for (size_t q=0; q<Asize; ++q)
		{
			qarray<Nq> qVal;
			for (size_t p=0; p<Nq; p++)
			{
				stringstream qq; qq << "l=" << l << ",q=" << q << ",p=" << p;
				source.load_scalar(qVal[p],qq.str(),"Cs");
			}
			stringstream ss;
			ss << l << "_" << "(" << qVal << ")";
			label = ss.str();
			MatrixType mat;
			source.load_matrix(mat, label, "Cs");
			C[l].push_back(qVal,qVal,mat);
		}
	}
	source.close();
	update_inbase();
	update_outbase();
}
#endif //USE_HDF5_STORAGE

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
sort_A(size_t loc, GAUGE::OPTION g, bool SORT_ALL_GAUGES)
{
	if (SORT_ALL_GAUGES)
	{
		for (size_t gP=0; gP<3; ++gP)
		for (size_t s=0; s<locBasis(loc).size(); ++s)
		{
			A[gP][loc][s] = A[gP][loc][s].sorted();
		}
	}
	else
	{
		for (size_t s=0; s<locBasis(loc).size(); ++s)
		{
			A[g][loc][s] = A[g][loc][s].sorted();
		}
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
updateC(size_t loc)
{
	for (size_t q=0; q<outBasis(loc).Nq(); ++q)
	{
		qarray2<Symmetry::Nq> quple = {outBasis(loc)[q], outBasis(loc)[q]};
		auto qC = C[loc].dict.find(quple);
		size_t r = outBasis(loc).inner_dim(outBasis(loc)[q]);
		size_t c = r;
		if (qC != C[loc].dict.end())
		{
			int dr = r-C[loc].block[qC->second].rows();
			int dc = c-C[loc].block[qC->second].cols();
			
			C[loc].block[qC->second].conservativeResize(r,c);
			
			C[loc].block[qC->second].bottomRows(dr).setZero();
			C[loc].block[qC->second].rightCols(dc).setZero();
		}
		else
		{
			MatrixType Mtmp(r,c);
			Mtmp.setZero();
			C[loc].push_back(quple, Mtmp);
		}
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
updateAC(size_t loc, GAUGE::OPTION g)
{
	assert(g != GAUGE::C and "Oouuhh.. you tried to update AC with itself, but we have no bootstrap implemented ;). Use AL or AR.");
	for (size_t s=0; s<locBasis(loc).size(); ++s)
	for (size_t q=0; q<A[g][loc][s].size(); ++q)
	{
		qarray2<Symmetry::Nq> quple = {A[g][loc][s].in[q], A[g][loc][s].out[q]};
		auto it = A[GAUGE::C][loc][s].dict.find(quple);
		if (it != A[GAUGE::C][loc][s].dict.end())
		{
			int dr = A[g][loc][s].block[q].rows() - A[GAUGE::C][loc][s].block[it->second].rows();
			int dc = A[g][loc][s].block[q].cols() - A[GAUGE::C][loc][s].block[it->second].cols();
			assert(dr >= 0 and dc >= 0 and "Something went wrong in expand_basis during the VUMPS Algorithm.");
			MatrixType Mtmp(dr,A[GAUGE::C][loc][s].block[it->second].cols()); Mtmp.setZero();
			addBottom(Mtmp, A[GAUGE::C][loc][s].block[it->second]);
			Mtmp.resize(A[GAUGE::C][loc][s].block[it->second].rows(),dc); Mtmp.setZero();
			addRight(Mtmp, A[GAUGE::C][loc][s].block[it->second]);
		}
		else
		{
			MatrixType Mtmp(A[g][loc][s].block[q].rows(), A[g][loc][s].block[q].cols());
			Mtmp.setZero();
			A[GAUGE::C][loc][s].push_back(quple,Mtmp);
		}
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
enrich(size_t loc, GAUGE::OPTION g, const vector<Biped<Symmetry,MatrixType> > &P)
{
	size_t loc1,loc2;
	if (g == GAUGE::L) {loc1 = loc; loc2 = (loc+1)%N_sites;}
	else if (g == GAUGE::R) {loc1 = (loc+1)%N_sites; loc2 = loc;}
	
	for (size_t s=0; s<locBasis(loc1).size(); ++s)
	for (size_t qP=0; qP<P[s].size(); ++qP)
	{
		qarray2<Symmetry::Nq> quple = {P[s].in[qP], P[s].out[qP]};
		auto qA = A[g][loc1][s].dict.find(quple);
		
		if (qA != A[g][loc1][s].dict.end())
		{
			if (g == GAUGE::L) {addRight(P[s].block[qP], A[g][loc1][s].block[qA->second]);}
			else if (g == GAUGE::R) {addBottom(P[s].block[qP], A[g][loc1][s].block[qA->second]);}			
		}
		else
		{
			A[g][loc1][s].push_back(quple, P[s].block[qP]);
		}
	}

	// update the inleg from AL at site loc2 with zeros
	Qbasis<Symmetry> ExpandedBasis;
	if (g == GAUGE::L) {ExpandedBasis.pullData(P,1);}
	else if (g == GAUGE::R) {ExpandedBasis.pullData(P,0);}
	
	update_inbase(loc1,g);
	update_outbase(loc1,g);
		
	for (const auto &[qval,qdim,plain]:ExpandedBasis)
	for (size_t s=0; s<locBasis(loc2).size(); ++s)
	{
		std::vector<qarray<Symmetry::Nq> > qins_outs;
		if (g == GAUGE::L) {qins_outs = Symmetry::reduceSilent(qval, locBasis(loc2)[s]);}
		else if (g == GAUGE::R) {qins_outs = Symmetry::reduceSilent(qval, Symmetry::flip(locBasis(loc2)[s]));}
		for (const auto &qin_out:qins_outs)
		{
			qarray2<Symmetry::Nq> quple;
			if (g == GAUGE::L)
			{
				if (outBasis(loc2).find(qin_out) == false) {continue;}
				quple = {qval, qin_out};
			}
			else if (g == GAUGE::R)
			{
				if (inBasis(loc2).find(qin_out) == false) {continue;}
				quple = {qin_out, qval};
			}
			auto it = A[g][loc2][s].dict.find(quple);
			if (it != A[g][loc2][s].dict.end())
			{
				if (g == GAUGE::L)
				{
					MatrixType Mtmp(ExpandedBasis.inner_dim(qval), 
									A[g][loc2][s].block[it->second].cols());
					Mtmp.setZero();
					addBottom(Mtmp, A[g][loc2][s].block[it->second]);
				}
				else if (g == GAUGE::R)
				{
					MatrixType Mtmp(A[g][loc2][s].block[it->second].rows(),
									ExpandedBasis.inner_dim(qval));
					Mtmp.setZero();
					addRight(Mtmp, A[g][loc2][s].block[it->second]);
				}
			}
			else
			{
				MatrixType Mtmp;
				if (g == GAUGE::L)      {Mtmp.resize(ExpandedBasis.inner_dim(qval), outBasis(loc2).inner_dim(qin_out));}
				else if (g == GAUGE::R) {Mtmp.resize(inBasis(loc2).inner_dim(qin_out), ExpandedBasis.inner_dim(qval));}
				Mtmp.setZero();
				A[g][loc2][s].push_back(quple, Mtmp);
			}
		}
	}

}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
orthogonalize_right(GAUGE::OPTION g, vector<Biped<Symmetry,MatrixType> > &G_R)
{
	vector<vector<Biped<Symmetry,MatrixType> > > A_ortho(N_sites);
	for (size_t l=0; l<N_sites; l++)
	{
		A_ortho[l].resize(qloc[l].size());
	}

	vector<Biped<Symmetry,MatrixType> > Rprev(N_sites);
	Biped<Symmetry,MatrixType> Xnext;
	Rprev[N_sites-1].setRandom(outbase[N_sites-1],outbase[N_sites-1]);
	Rprev[0] = 1./sqrt(Rprev[0].squaredNorm().sum()) * Rprev[0];
	
	vector<vector<Biped<Symmetry,MatrixType> > > AxR(N_sites);
	for (size_t loc=0; loc<N_sites; loc++)
	{
		AxR[loc].resize(qloc[loc].size());
	}
	double tol = 1.e-10, err=1.;
	size_t i=0, imax = 10001;
	while (err > tol and i < imax)
	{
		for (size_t loc=N_sites-1; loc!=-1; --loc)
		{
			for (size_t s=0; s<qloc[loc].size(); s++)
			{
				AxR[loc][s] = A[g][loc][s] * Rprev[loc];
			}
			Blocker<Symmetry,Scalar> Jim(AxR[loc], qloc[loc], inbase[loc], outbase[loc]);
			auto A_blocked = Jim.Aclump(DMRG::DIRECTION::RIGHT);
			Biped<Symmetry,MatrixType> Q,R;
			for (size_t q=0; q<A_blocked.dim; ++q)
			{
				HouseholderQR<MatrixType> Quirinus;
				Quirinus.compute(A_blocked.block[q].adjoint());

				MatrixType Qmat = Quirinus.householderQ() * MatrixType::Identity(A_blocked.block[q].cols(),A_blocked.block[q].rows());
				MatrixType Rmat = MatrixType::Identity(A_blocked.block[q].rows(),A_blocked.block[q].cols()) * Quirinus.matrixQR().template triangularView<Upper>();
				//make the QR decomposition unique by enforcing the diagonal of R to be positive.
				DiagonalMatrix<Scalar,Dynamic> Sign = Rmat.diagonal().cwiseSign().matrix().asDiagonal();

				Rmat = Sign*Rmat;
				Qmat = Qmat*Sign;

				Q.push_back(A_blocked.in[q], A_blocked.out[q], (Qmat.adjoint()));
				R.push_back(A_blocked.in[q], A_blocked.out[q], Rmat.adjoint());
			}
			R = 1./R.operatorNorm(false) * R;
			if (loc>0) {Rprev[loc-1] = R;}
			else {Xnext = R;}
			A_ortho[loc] = Jim.reblock(Q, DMRG::DIRECTION::RIGHT);
		}
		err = (Xnext - Rprev[N_sites-1]).norm();
		Rprev[N_sites-1] = Xnext;
		i++;
		// cout << "iteration number=" << i << ", err=" << err << endl << endl;
	}
	G_R = Rprev;
	cout << "iteration number=" << i << ", err=" << err << endl << endl;

	A[g] = A_ortho;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
orthogonalize_left(GAUGE::OPTION g, vector<Biped<Symmetry,MatrixType> > &G_L)
{
	vector<vector<Biped<Symmetry,MatrixType> > > A_ortho(N_sites);
	for (size_t l=0; l<N_sites; l++)
	{
		A_ortho[l].resize(qloc[l].size());
	}
	
	vector<Biped<Symmetry,MatrixType> > Lprev(N_sites);
	Biped<Symmetry,MatrixType> Lnext;
	Lprev[0].setRandom(inbase[0],inbase[0]);
	Lprev[0] = 1./sqrt(Lprev[0].squaredNorm().sum()) * Lprev[0];
	vector<vector<Biped<Symmetry,MatrixType> > > LxA(N_sites);
	for (size_t loc=0; loc<N_sites; loc++)
	{
		LxA[loc].resize(qloc[loc].size());
	}
	double tol = 1.e-10, err=1.;
	size_t i=0, imax = 10001;
	while (err > tol and i < imax)
	{
		for (size_t loc=0; loc<N_sites; loc++)
		{
			for (size_t s=0; s<qloc[loc].size(); s++)
			{
				LxA[loc][s] = Lprev[loc] * A[g][loc][s];
			}
			Blocker<Symmetry,Scalar> Jim(LxA[loc], qloc[loc], inbase[loc], outbase[loc]);
			auto A_blocked = Jim.Aclump(DMRG::DIRECTION::LEFT);
			Biped<Symmetry,MatrixType> Q,R;
			for (size_t q=0; q<A_blocked.dim; ++q)
			{
				HouseholderQR<MatrixType> Quirinus;
				Quirinus.compute(A_blocked.block[q]);

				MatrixType Qmat = Quirinus.householderQ() * MatrixType::Identity(A_blocked.block[q].rows(),A_blocked.block[q].cols());
				MatrixType Rmat = MatrixType::Identity(A_blocked.block[q].cols(),A_blocked.block[q].rows()) * Quirinus.matrixQR().template triangularView<Upper>();
				//make the QR decomposition unique by enforcing the diagonal of R to be positive.
				DiagonalMatrix<Scalar,Dynamic> Sign = Rmat.diagonal().cwiseSign().matrix().asDiagonal();

				Rmat = Sign*Rmat;
				Qmat = Qmat*Sign;

				Q.push_back(A_blocked.in[q], A_blocked.out[q], Qmat);
				R.push_back(A_blocked.in[q], A_blocked.out[q], Rmat);
			}
			R = 1./R.operatorNorm(false) * R;
			if (loc<N_sites-1) {Lprev[loc+1] = R;}
			else {Lnext = R;}
			A_ortho[loc] = Jim.reblock(Q, DMRG::DIRECTION::LEFT);
		}
		err = (Lnext - Lprev[0]).norm();
		Lprev[0] = Lnext;
		i++;
		// cout << "iteration number=" << i << ", err=" << err << endl << endl;
	}
	G_L = Lprev;
	A[g] = A_ortho;
	cout << "iteration number=" << i << ", err=" << err << endl << endl;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
truncate(bool SET_AC_RANDOM)
{
	//isometries from the truncated SVD from the center-matrix C
	vector<Biped<Symmetry,MatrixType> > U(N_sites);
	vector<Biped<Symmetry,MatrixType> > Vdag(N_sites);

	//decompose C by SVD and write isometries to U and Vdag and the singular (Schmidt) values into C.
	for (size_t l=0; l<N_sites; ++l)
	{
		cout << "**********************l=" << l << "*************************" << endl;
		for (size_t q=0; q<C[l].dim; ++q)
		{
			JacobiSVD<MatrixType> Jack(C[l].block[q], ComputeFullU|ComputeFullV);
			size_t Nret = (Jack.singularValues().array() > eps_svd).count();
			cout << "q=" << C[l].in[q] << ", Nret=" << Nret  << " states from " << Jack.singularValues().rows() << endl;
			C[l].block[q] = Jack.singularValues().head(Nret).asDiagonal();
			U[l].push_back(C[l].in[q], C[l].out[q], Jack.matrixU().leftCols(Nret));
			Vdag[l].push_back(C[l].in[q], C[l].out[q], Jack.matrixV().adjoint().topRows(Nret));
		}
	}

	//update AL and AR
	for (size_t l=0; l<N_sites; ++l)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		{			
			A[GAUGE::L][l][s] = U[minus1modL(l)].adjoint() * A[GAUGE::L][l][s] * U[l];
			A[GAUGE::R][l][s] = Vdag[minus1modL(l)] * A[GAUGE::R][l][s] * Vdag[l].adjoint();
		}
	}
	update_outbase(GAUGE::L);
	update_inbase(GAUGE::L);

	//Orthogonalize AL and AR again and safe gauge transformation into L and R. (AL -> L*AL*Linv, AR -> Rinv*AR*R)
	//L and R need to be multiplied into the center matrix afterwards
	vector<Biped<Symmetry,MatrixType> > L(N_sites),R(N_sites);
	orthogonalize_right(GAUGE::R,R);
	orthogonalize_left(GAUGE::L,L);
	for (size_t l=0; l<N_sites; ++l)
	{
		C[l] = L[(l+1)%N_sites] * C[l] * R[l];
		C[l] = C[l].sorted();
	}
	
	//normalize the state to get rid off small norm changes due to the truncation
	normalize_C();

	//Update AC so that it has the truncated sizes and sort the A tensors.
	for (size_t l=0; l<N_sites; l++)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			A[GAUGE::C][l][s] = C[minus1modL(l)] * A[GAUGE::R][l][s];
			if (SET_AC_RANDOM) { A[GAUGE::C][l][s].setRandom(); }
		}
		sort_A(l,GAUGE::L,true); //true means sort all GAUGES. First parameter has no consequences
	}
	calc_entropy();
	// cout << test_ortho() << endl;
}

template<typename Symmetry, typename Scalar>
std::pair<complex<double>,Biped<Symmetry,Matrix<complex<double>,Dynamic,Dynamic> > > Umps<Symmetry,Scalar>::
calc_dominant (GAUGE::OPTION g, DMRG::DIRECTION::OPTION DIR) const
{
	Umps<Symmetry,complex<double> > Compl = this->template cast<complex<double> > ();
	complex<double> lambda;
	
	TransferMatrix<Symmetry,complex<double> > T;
	if (DIR == DMRG::DIRECTION::LEFT)
	{
		T = TransferMatrix<Symmetry,complex<double> >(GAUGE::R, Compl.A[g], Compl.A[g], Compl.locBasis());
	}
	else
	{
		T = TransferMatrix<Symmetry,complex<double> >(GAUGE::L, Compl.A[g], Compl.A[g], Compl.locBasis());		
	}
	
	Biped<Symmetry,Matrix<complex<double>, Dynamic,Dynamic> > RandBiped;
	if (DIR == DMRG::DIRECTION::LEFT)
	{
		RandBiped.setRandom(Compl.inBasis(0), Compl.inBasis(0));
	}
	else
	{
		RandBiped.setRandom(Compl.outBasis(N_sites-1), Compl.outBasis(N_sites-1));
	}
	RandBiped = 1./RandBiped.norm() * RandBiped;
	TransferVector<Symmetry,complex<double> > x(RandBiped);
	
	ArnoldiSolver<TransferMatrix<Symmetry,complex<double> >,TransferVector<Symmetry,complex<double> > > John(T,x,lambda);
	
	lout << "fixed point: " << John.info() << endl;
	//Normalize the Fixed point and try to make it real.
	x.data[0] = exp(-1i*arg(x.data[0].block[0](0,0))) * (1./x.data[0].norm()) * x.data[0];
	
	return std::make_pair(lambda,x.data[0]);
}

template<typename Symmetry, typename Scalar>
ArrayXd Umps<Symmetry,Scalar>::
entanglementSpectrumLoc (size_t loc) const
{
	vector<double> Svals;
	for (const auto &x : SVspec[loc])
	for (int i=0; i<x.second.size(); ++i)
	{
		Svals.push_back(x.second(i));
	}
	sort(Svals.begin(), Svals.end());
	reverse(Svals.begin(), Svals.end());
	
	ArrayXd out(Svals.size());
	for (int i=0; i<Svals.size(); ++i)
	{
		out(i) = Svals[i];
	}
	return out;
}

template<typename Symmetry, typename Scalar>
Array<complex<Scalar>,Dynamic,1> Umps<Symmetry,Scalar>::
structure_factor (const Mpo<Symmetry,Scalar> &Oalfa, const Mpo<Symmetry,Scalar> &Obeta, double kmin, double kmax, int kpoints, bool INFO)
{
	double t_tot=0.;
	double t_LReigen=0.;
	double t_GMRES=0.;
	double t_contraction=0.;
	
	Stopwatch<> TotTimer;
	
	Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > Reigen_LR, Leigen_LR, Reigen_RL, Leigen_RL;
	
	Stopwatch<> LReigenTimer;
	
	// T_L^R, right eigenvector
	Reigen_LR = calc_LReigen(VMPS::DIRECTION::RIGHT, A[GAUGE::L], A[GAUGE::R], outBasis(N_sites-1), outBasis(N_sites-1), qloc).state;
	// T_L^R, left eigenvector
	Leigen_LR = calc_LReigen(VMPS::DIRECTION::LEFT,  A[GAUGE::L], A[GAUGE::R], inBasis(0), inBasis(0), qloc).state;
	// T_R^L, right eigenvector
	Reigen_RL = calc_LReigen(VMPS::DIRECTION::RIGHT, A[GAUGE::R], A[GAUGE::L], outBasis(N_sites-1), outBasis(N_sites-1), qloc).state;
	// T_R^L, left eigenvector
	Leigen_RL = calc_LReigen(VMPS::DIRECTION::LEFT,  A[GAUGE::R], A[GAUGE::L], inBasis(0), inBasis(0), qloc).state;
	
	t_LReigen += LReigenTimer.time();
	
	// b (edge tensor of contraction) for alfa, beta and exp(-i*k), exp(+i*k)
	
	Stopwatch<> ContractionTimer;
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Lid; Lid.setIdentity(1,1,inBasis(0));
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Rid; Rid.setIdentity(1,1,outBasis(N_sites-1));
	
	// term exp(-i*k), alfa
	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > bmalfaTripod(N_sites);
	contract_L(Lid, A[GAUGE::L][0], Oalfa.W_at(0), Oalfa.IS_HAMILTONIAN(), A[GAUGE::C][0], 
	           Oalfa.locBasis(0), Oalfa.opBasis(0), bmalfaTripod[0]);
		// shift forward in cell
		for (size_t l=1; l<N_sites; ++l)
		{
			contract_L(bmalfaTripod[l-1], A[GAUGE::L][l], Oalfa.W_at(l), Oalfa.IS_HAMILTONIAN(), A[GAUGE::R][l], 
			           Oalfa.locBasis(l), Oalfa.opBasis(l), bmalfaTripod[l]);
		}
	
	// term exp(+i*k), alfa
	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > bpalfaTripod(N_sites);
	contract_R(Rid, A[GAUGE::R][N_sites-1], Oalfa.W_at(N_sites-1), Oalfa.IS_HAMILTONIAN(), A[GAUGE::C][N_sites-1], 
	           Oalfa.locBasis(N_sites-1), Oalfa.opBasis(N_sites-1), bpalfaTripod[N_sites-1]);
		// shift backward in cell
		for (int l=N_sites-2; l>=0; --l)
		{
			contract_R(bpalfaTripod[l+1], A[GAUGE::R][l], Oalfa.W_at(l), Oalfa.IS_HAMILTONIAN(), A[GAUGE::L][l], 
			           Oalfa.locBasis(l), Oalfa.opBasis(l), bpalfaTripod[l]);
		}
	
	// term exp(-i*k), beta
	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > bmbeta(N_sites);
	contract_R(Rid, A[GAUGE::C][N_sites-1], Obeta.W_at(N_sites-1), Obeta.IS_HAMILTONIAN(), A[GAUGE::R][N_sites-1], 
	           Obeta.locBasis(N_sites-1), Obeta.opBasis(N_sites-1), bmbeta[N_sites-1]);
		// shift backward in cell
		for (int l=N_sites-2; l>=0; --l)
		{
			contract_R(bmbeta[l+1], A[GAUGE::L][l], Oalfa.W_at(l), Oalfa.IS_HAMILTONIAN(), A[GAUGE::R][l], 
			           Obeta.locBasis(l), Obeta.opBasis(l), bmbeta[l]);
		}
	
	// term exp(+i*k), beta
	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > bpbeta(N_sites);
	contract_L(Lid, A[GAUGE::C][0], Obeta.W_at(0), Obeta.IS_HAMILTONIAN(), A[GAUGE::L][0], 
	           Obeta.locBasis(0), Obeta.opBasis(0), bpbeta[0]);
		// shift forward in cell
		for (size_t l=1; l<N_sites; ++l)
		{
			contract_L(bpbeta[l-1], A[GAUGE::R][l], Obeta.W_at(l), Obeta.IS_HAMILTONIAN(), A[GAUGE::L][l], 
			           Obeta.locBasis(l), Obeta.opBasis(l), bpbeta[l]);
		}
	
	t_contraction += ContractionTimer.time();
	
	// convert bmalfa, bpalfa to Biped wrapped by TransferVector for GMRES
	TransferVector<Symmetry,complex<Scalar> > bmalfa(bmalfaTripod[N_sites-1].BipedSlice().template cast<MatrixXcd >());
	TransferVector<Symmetry,complex<Scalar> > bpalfa(bpalfaTripod[0].BipedSlice().template cast<MatrixXcd>());
	
	Array<complex<Scalar>,Dynamic,1> out(kpoints);
	if (kmin==kmax) {out.resize(1);} // only one k-point needed in case of kmin=kmax
	
	// solve linear systems
	Stopwatch<> GMRES_Timer;
	#pragma omp parallel for
	for (int i=0; i<out.rows(); ++i)
	{
		double k = (kmin==kmax)? kmin : kmin + i*(kmax-kmin)/(kpoints-1);
		
		GMResSolver<TransferMatrixSF<Symmetry,Scalar>,TransferVector<Symmetry,complex<Scalar> > > Gimli;
		
		// term exp(-i*k)
		TransferMatrixSF<Symmetry,Scalar> Tm(VMPS::DIRECTION::LEFT, A[GAUGE::L], A[GAUGE::R], Leigen_LR, Reigen_LR, qloc, k);
		Gimli.set_dimK(min(100ul,dim(bmalfa)));
		TransferVector<Symmetry,complex<Scalar> > Fmalfa;
		Gimli.solve_linear(Tm, bmalfa, Fmalfa, 1e-14, true);
//		lout << "i=" << i << "\t" << Gimli.info() << endl;
		
		// term exp(+i*k)
		TransferMatrixSF<Symmetry,Scalar> Tp(VMPS::DIRECTION::RIGHT, A[GAUGE::R], A[GAUGE::L], Leigen_RL, Reigen_RL, qloc, k);
		Gimli.set_dimK(min(100ul,dim(bpalfa)));
		TransferVector<Symmetry,complex<Scalar> > Fpalfa;
		Gimli.solve_linear(Tp, bpalfa, Fpalfa, 1e-14, true);
//		lout << "i=" << i << "\t" << Gimli.info() << endl;
		
		// result
		out(i) = exp(-1.i*k) * contract_LR(0, Fmalfa.data, bmbeta[0].template cast<MatrixXcd>()) 
		        +exp(+1.i*k) * contract_LR(0, bpbeta[N_sites-1].template cast<MatrixXcd>(), Fpalfa.data); // contract Biped & Tripod (qmid=0)
	}
	
	t_GMRES += GMRES_Timer.time();
	
	t_tot = TotTimer.time();
	
	if (INFO)
	{
		lout << TotTimer.info("StructureFactor")
			 << " (LReigen=" << round(t_LReigen/t_tot*100.,0) << "%, "
			 << "GMRES=" << round(t_GMRES/t_tot*100.,0) << "%, "
			 << "contractions=" << round(t_contraction/t_tot*100.,0) << "%)"
			 << ", kmin/π=" << kmin/M_PI << ", kmax/π=" << kmax/M_PI << ", kpoints=" << out.rows() << endl;
		lout << "\t" << Oalfa.info() << endl;
		lout << "\t" << Obeta.info() << endl;
	}
	
	return out;
}

#endif //VANILLA_Umps
