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

//include "PolychromaticConsole.h" // from TOOLS
//include "tensors/Biped.h"
//include "LanczosSolver.h" // from ALGS
//include "ArnoldiSolver.h" // from ALGS
//include "VUMPS/VumpsTransferMatrix.h"
//include "Mpo.h"
//include "tensors/DmrgConglutinations.h"


/**Uniform Matrix Product State. Analogue of the Mps class.
\ingroup VUMPS
\describe_Symmetry
\describe_Scalar*/
template<typename Symmetry, typename Scalar=double>
class Umps
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	static constexpr size_t Nq = Symmetry::Nq;
	
	template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class VumpsSolver;
	
public:
	
	/**Does nothing.*/
	Umps<Symmetry,Scalar>(){};
	
	/**Constructs a Umps with fixed bond dimension with the info from the Hamiltonian.*/
	template<typename Hamiltonian> Umps (const Hamiltonian &H, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax);
	
	/**Constructs a Umps with fixed bond dimension with a given basis.*/
	Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, qarray<Nq> Qtot_input, size_t L_input, size_t Dmax, size_t Nqmax);

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
	
//	void expand_basis (size_t DeltaD, const boost::multi_array<Scalar,4> &h2site, double e);
	
	void calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,MatrixType> > &N) const;
	
	void truncate();

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

//private:
	/**parameter*/
	size_t N_sites;
	size_t Dmax, Nqmax;
	double eps_svd = 1e-12;
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
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> N; // N[L/R/C][l][s].block[q]
	
	VectorXd S;
	
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
		
		N[g].resize(N_sites);
		for (size_t l=0; l<N_sites; ++l)
		{
			N[g][l].resize(qloc[l].size());
		}
	}
	C.resize(N_sites);
	inbase.resize(N_sites);
	outbase.resize(N_sites);
	S.resize(N_sites);
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
resize (size_t Dmax_input, size_t Nqmax_input)
{
	Dmax = Dmax_input;
	Nqmax = Nqmax_input;
	if (Symmetry::IS_TRIVIAL) {Nqmax = 1;}

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
	ArrayXd norm(N_sites);
	
	for (int l=0; l<N_sites; ++l)
	{
		// check for A
		Biped<Symmetry,MatrixType> Test = A[GAUGE::L][l][0].adjoint().contract(A[GAUGE::L][l][0]);
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[GAUGE::L][l][s].adjoint().contract(A[GAUGE::L][l][s]);
		}
		cout << "l=" << l << endl << Test.print(true) << endl << endl; 
		// cout << Test.print(true) << endl;
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
			Biped<Symmetry,MatrixType> Test = A[GAUGE::L][l][s] * C[l];
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
			Biped<Symmetry,MatrixType> Test = C[locC] * A[GAUGE::R][l][s];
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
			assert(1!=1 and "AL is wrong");
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
			assert(1!=1 and "AR is wrong");
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
	double outL = calc_LReigen(GAUGE::L, A[GAUGE::L], Vket.A[GAUGE::L], outBasis(N_sites-1), Vket.outBasis(N_sites-1), qloc).energy;
	// for testing:
	double outR = calc_LReigen(GAUGE::R, A[GAUGE::R], Vket.A[GAUGE::R], inBasis(0), Vket.inBasis(0), qloc).energy;
	cout << "dot consistency check: from AL: " << outL << ", from AR: " << outR << endl;
	return outL;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_entropy (size_t loc, bool PRINT)
{
	S(loc) = 0;
	
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
		
		if (PRINT)
		{
			lout << termcolor::magenta << "S(" << C[loc].in[q] << "," << C[loc].out[q] << ")\t=\t" << Scontrib << ", size=" << C[loc].block[q].rows() << "x" << C[loc].block[q].cols() << termcolor::reset << endl;
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
truncate()
{
	vector<Biped<Symmetry,MatrixType> > U(N_sites);
	vector<Biped<Symmetry,MatrixType> > Vdag(N_sites);
	cout << "eps_svd=" << eps_svd << endl;
	//decompose C by SVD and write isometries to U and V and the Schmidt values into C.
	for (size_t l=0; l<N_sites; ++l)
	{
		for (size_t q=0; q<C[l].dim; ++q)
		{
			JacobiSVD<MatrixType> Jack(C[l].block[q], ComputeThinU|ComputeThinV);
			size_t Nret = (Jack.singularValues().array() > eps_svd).count();
			// size_t Nret = Jack.singularValues().rows();
			Nret = max(Nret,1ul);
			cout << "q=" << C[l].in[q] << ", Nret=" << Nret << endl;
	//		C[l].block[q] = Jack.matrixU().leftCols(Nret) * 
	//		                  Jack.singularValues().head(Nret).asDiagonal() * 
	//		                  Jack.matrixV().adjoint().topRows(Nret);
			if (Nret > 0)
			{
				C[l].block[q] = Jack.singularValues().head(Nret).asDiagonal();
				U[l].push_back(C[l].in[q], C[l].out[q], Jack.matrixU().leftCols(Nret));
				Vdag[l].push_back(C[l].in[q], C[l].out[q], Jack.matrixV().adjoint().topRows(Nret));
			}
		}
		
		// cout << "U=" << endl;
		// cout << (U[l].adjoint().contract(U[l])).print(true) << endl;
		// cout << endl;
		
		// cout << "Vdag=" << endl;
		// cout << (Vdag[l].contract(Vdag[l].adjoint())).print(true) << endl;
		// cout << endl;
	}

	//update AL and AR
	for (size_t l=0; l<N_sites; ++l)
	{
		cout << "cutting A_LR, l=" << l << ", minus1modL(l)=" << minus1modL(l) << endl;
		for (size_t s=0; s<qloc[l].size(); ++s)
		{
			A[GAUGE::L][l][s] = U[minus1modL(l)].adjoint() * A[GAUGE::L][l][s] * U[l];
			A[GAUGE::R][l][s] = Vdag[minus1modL(l)] * A[GAUGE::R][l][s] * Vdag[l].adjoint();
			
			A[GAUGE::C][l][s] = A[GAUGE::L][l][s] * C[l];
			// A[GAUGE::C][l][s].setRandom();
		}
	}
	update_outbase();
	update_inbase();
	cout << test_ortho() << endl;
}
#endif //VANILLA_Umps
