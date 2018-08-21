#ifndef VANILLA_Umps
#define VANILLA_Umps

// #if !defined DONT_USE_LAPACK_SVD || !defined DONT_USE_LAPACK_QR
// 	#include "LapackWrappers.h"
// #endif

#include <set>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <type_traits>
#include <iostream>
#include <fstream>

#include "VUMPS/VumpsTypedefs.h"
#include "VUMPS/VumpsTransferMatrix.h"
#include "tensors/Biped.h"
#include "LanczosSolver.h" // from LANCZOS
#include "ArnoldiSolver.h" // from LANCZOS
#include "Mpo.h"
#include "tensors/DmrgConglutinations.h"
// #if !defined DONT_USE_LAPACK_SVD || !defined DONT_USE_LAPACK_QR
// 	#include "LapackWrappers.h"
// #endif
#include "PolychromaticConsole.h" // from HELPERS
#include "RandomVector.h" // from LANCZOS

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
	
	/**Resizes all containers to \p N_sites, the bond dimension to \p Dmax and sets \p Nqmax blocks per site.*/
	void resize (size_t Dmax_input, size_t Nqmax_input);
	
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
	
	void calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,MatrixType> > &N);
	
	void truncate();

	/**
	 * This functions transforms all quantum numbers in the Umps (Umps::qloc and QN in Umps::A) by \f$q \rightarrow q * N_{cells}\f$.
	 * It is used for avg(Umps V, Mpo O, Umps V) in VumpsLinearAlgebra.h when O.length() > V.length(). 
	 * In this case the quantum numbers in the Umps are transformed in correspondence with V.length()
	 * and this is incompatible with the quantum numbers in O.length() which are transformed in correspondence to O.length().
	 * \param number_cells : \f$N_{cells}\f$
	 */
	void adjustQN (const size_t number_cells);

//private:
	
	/**parameter*/
	size_t N_sites;
	size_t Dmax, Nqmax;
	double eps_svd = 1e-12;
	size_t N_sv;
	qarray<Nq> Qtot;
	
	/**Calculate entropy at site \p loc.*/
	void calc_entropy (size_t loc, bool PRINT=false);
	
	/**Calculate entropy for all sites.*/
	void calc_entropy (bool PRINT=false) {for (size_t l=0; l<N_sites; ++l) calc_entropy(l,PRINT);};
	
	/**Sets of all unique incoming & outgoing indices for convenience*/
	vector<vector<qarray<Symmetry::Nq> > > inset;
	vector<vector<qarray<Symmetry::Nq> > > outset;
	
	/**local basis*/
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	
	/**A-tensors in the three gauges \p L, \p R, \p C*/
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> A; // A[L/R/C][l][s].block[q]
	
	/**Contracted and saved A-tensors (\p L and \p R) of the whole unit cell.*/
	std::array<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > >,2> Acell;
	
	/**Basis of the whole unit cell.*/
	vector<qarray<Symmetry::Nq> > qlocCell;
	
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
//	qarray<Symmetry::Nq> res = Symmetry::qvacuum();
//	for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
//	{
//		if (outbase[loc][qout] > res) {res = outbase[loc][qout];}
//	}
//	return res;
	return qplusinf<Symmetry::Nq>();
}

template<typename Symmetry, typename Scalar>
qarray<Symmetry::Nq> Umps<Symmetry,Scalar>::
Qbot (size_t loc) const
{
//	qarray<Symmetry::Nq> res = Symmetry::qvacuum();
//	for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
//	{
//		if (outbase[loc][qout] < res) {res = outbase[loc][qout];}
//	}
//	return res;
	return qminusinf<Symmetry::Nq>();
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
resize (size_t Dmax_input, size_t Nqmax_input)
{
	Dmax = Dmax_input;
	Nqmax = Nqmax_input;
	if (Symmetry::IS_TRIVIAL) {Nqmax = 1;}
	
//	C.clear();
//	inbase.clear();
//	outbase.clear();
//	for (size_t g=0; g<3; ++g)
//	{
//		A[g].clear();
//	}
	
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
	// check later for particles!
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
	
	S.resize(N_sites);
	
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
		vector<bool> A_CHECK(Test.dim);
		vector<double> A_infnorm(Test.dim);
		for (size_t q=0; q<Test.dim; ++q)
		{
			Test.block[q] -= MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
			A_CHECK[q]     = Test.block[q].norm()<tol ? true : false;
			A_infnorm[q]   = Test.block[q].norm();
//			cout << "q=" << Test.in[q] << ", A_infnorm[q]=" << A_infnorm[q] << endl;
		}
		
		// check for B
		Test.clear();
		Test = A[GAUGE::R][l][0].contract(A[GAUGE::R][l][0].adjoint(), contract::MODE::OORR);
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[GAUGE::R][l][s].contract(A[GAUGE::R][l][s].adjoint(), contract::MODE::OORR);
		}
		
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
//	assert(N_sites == Vket.length());
//	assert (N_sites==1 or N_sites==2 and "Only Lcell=1 and Lcell=2 implemented in dot product!");
//	
//	MatrixType LRdummy;
//	size_t Mbra = A[GAUGE::R][0][0].block[0].rows();
//	size_t Mket = Vket.A[GAUGE::R][0][0].block[0].rows();
//	size_t D0 = qloc[0].size();
//	
//	TransferMatrix<Symmetry,double> TR;
//	
//	if (N_sites == 1)
//	{
//		TR = TransferMatrix<Symmetry,double>(GAUGE::R, A[GAUGE::R][0], Vket.A[GAUGE::R][0], LRdummy, {}, {D0});
//	}
//	else if (N_sites == 2)
//	{
//		// Pre-contract the A-tensors
//		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairR;
//		contract_AA(A[GAUGE::R][0], qloc[0], A[GAUGE::R][1], qloc[1], ApairR);
//		
//		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairKetR;
//		contract_AA(Vket.A[GAUGE::R][0], qloc[0], Vket.A[GAUGE::R][1], qloc[1], ApairKetR);
//		
//		// The transfer matrix requires an MPO. Set up a dummy which is equal to unity.
//		size_t D1 = qloc[1].size();
//		boost::multi_array<double,4> WarrayDummy(boost::extents[D0][D0][D1][D1]);
//		for (size_t s1=0; s1<D0; ++s1)
//		for (size_t s2=0; s2<D0; ++s2)
//		for (size_t s3=0; s3<D1; ++s3)
//		for (size_t s4=0; s4<D1; ++s4)
//		{
//			WarrayDummy[s1][s2][s3][s4] = (s1==s2 and s3==s4)? 1.:0.;
//		}
//		
//		TR.Warray.resize(boost::extents[D0][D0][D1][D1]); // This resize is necessary. I hate boost::multi_array. :-(
//		TR = TransferMatrix<Symmetry,double>(GAUGE::R, ApairR, ApairKetR, LRdummy, WarrayDummy, {D0,D1});
//	}
//	
//	// Calculate dominant eigenvalue
//	TransferVector<complex<double> > Reigen;
//	Reigen.A.resize(Mket,Mbra);
//	ArnoldiSolver<TransferMatrix<Symmetry,double>,TransferVector<complex<double> > > Arnie;
//	Arnie.set_dimK(min(30ul,Mbra*Mket));
//	complex<double> lambda;
//	
//	Arnie.calc_dominant(TR,Reigen,lambda);
//	lout << Arnie.info() << endl;
//	
//	return lambda;
	
	double outL = calc_LReigen(GAUGE::L, Acell[GAUGE::L], Vket.Acell[GAUGE::L], C[N_sites-1], qlocCell).energy;
	cout << "from AL: " << outL << endl;
	double out = calc_LReigen(GAUGE::R, Acell[GAUGE::R], Vket.Acell[GAUGE::R], C[N_sites-1], qlocCell).energy;
	return out;
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
		vector<Biped<Symmetry,MatrixType> > Atmp(qloc[loc].size());
		for (size_t s=0; s<qloc[loc].size(); ++s)
		{
			// Atmp[s] = A[GAUGE::C][loc][s].contract(C[loc].adjoint());
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
			
			size_t Nret = Jack.singularValues().rows();
			
			// Update AL
			stitch = 0;
			for (size_t i=0; i<svec.size(); ++i)
			{
				A[GAUGE::L][loc][svec[i]].block[qvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret) * 
				                                           Jack.matrixV().adjoint().topRows(Nret);
				stitch += Nrowsvec[i];
			}
		}
	}
	
	size_t locC = minus1modL(loc);
	
	if (gauge == GAUGE::R or gauge == GAUGE::C)
	{
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
			
			size_t Nret = Jack.singularValues().rows();
			
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
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_N (DMRG::DIRECTION::OPTION DIR, size_t loc, vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &N)
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
					Mtmp.setIdentity();
					Mtmp *= Symmetry::coeff_sign( outbase[loc][qout], qfull, qloc[loc][s] );
					N[s].push_back(quple, Mtmp);
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
					Mtmp.setIdentity();
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

//template<typename Symmetry, typename Scalar>
//void Umps<Symmetry,Scalar>::
//truncate()
//{
//	vector<Biped<Symmetry,MatrixType> > U(N_sites);
//	vector<Biped<Symmetry,MatrixType> > Vdag(N_sites);
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		for (size_t q=0; q<C[l].dim; ++q)
//		{
//			JacobiSVD<MatrixType> Jack(C[l].block[q], ComputeThinU|ComputeThinV);
////			size_t Nret = (Jack.singularValues().array() > eps_svd).count();
//			size_t Nret = Jack.singularValues().rows();
//			Nret = max(Nret,1ul);
//			cout << "q=" << C[l].in[q] << ", Nret=" << Nret << endl;
//	//		C[l].block[q] = Jack.matrixU().leftCols(Nret) * 
//	//		                  Jack.singularValues().head(Nret).asDiagonal() * 
//	//		                  Jack.matrixV().adjoint().topRows(Nret);
//			C[l].block[q] = Jack.singularValues().head(Nret).asDiagonal();
//			U[l].push_back(C[l].in[q], C[l].out[q], Jack.matrixU().leftCols(Nret));
//			Vdag[l].push_back(C[l].in[q], C[l].out[q], Jack.matrixV().adjoint().topRows(Nret));
//		}
//		
////		cout << "U=" << endl;
////		cout << (U[l].adjoint().contract(U[l])).print(true) << endl;
////		cout << endl;
////		
////		cout << "Vdag=" << endl;
////		cout << (Vdag[l].contract(Vdag[l].adjoint())).print(true) << endl;
////		cout << endl;
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		cout << "cutting A_LR, l=" << l << endl;
//		for (size_t s=0; s<qloc[l].size(); ++s)
//		{
//			A[GAUGE::L][l][s] = Vdag[minus1modL(l)] * A[GAUGE::L][l][s] * U[l];
//			A[GAUGE::R][l][s] = Vdag[minus1modL(l)] * A[GAUGE::R][l][s] * U[l];
//			
//			A[GAUGE::C][l][s] = A[GAUGE::L][l][s];
//			A[GAUGE::C][l][s].setRandom();
//		}
//	}
//	
//	cout << test_ortho() << endl;
//}

//template<typename Symmetry, typename Scalar>
//void Umps<Symmetry,Scalar>::
//expand_basis (size_t DeltaD, const boost::multi_array<Scalar,4> &h2site, double e)
//{
//	vector<Biped<Symmetry,MatrixType> > NL;
//	vector<Biped<Symmetry,MatrixType> > NR;
//	
//	size_t l = 0;
//	size_t loc = l;
//	
//	calc_N(DMRG::DIRECTION::RIGHT, l, NL);
//	calc_N(DMRG::DIRECTION::LEFT,  l, NR);
//	
//	Biped<Symmetry,MatrixType> TestR = A[GAUGE::R][l][0].contract(NR[0].adjoint());
//	Biped<Symmetry,MatrixType> TestL = NL[0].adjoint().contract(A[GAUGE::L][l][0]);
//	for (size_t s=1; s<qloc[l].size(); ++s)
//	{
//		TestR += A[GAUGE::R][l][s].contract(NR[s].adjoint());
//		TestL += NL[s].adjoint().contract(A[GAUGE::L][l][s]);
//	}
//	
//	for (size_t q=0; q<TestL.dim; ++q)
//	{
//		cout << "q=" << q << ", TestLR.block[q].norm()=\t" << TestR.block[q].norm() << "\t" << TestL.block[q].norm() << endl;
//	}
//	
////	cout << "NL=" << endl;
////	for (size_t s=0; s<qloc[l].size(); ++s)
////	{
////		cout << "s=" << s << endl;
////		cout << NL[s].print(false,15) << endl;
////	}
////	
////	cout << "NR=" << endl;
////	for (size_t s=0; s<qloc[l].size(); ++s)
////	{
////		cout << "s=" << s << endl;
////		cout << NR[s].print(false,15) << endl;
////	}
//	
//	Biped<Symmetry,MatrixType> NAAN;
//	
//	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > A2C;
//	contract_AA (Vout.state.A[GAUGE::L][0], H.locBasis(0), 
//	             Vout.state.A[GAUGE::C][0], H.locBasis(0), 
//	             Vout.state.Qtop(0), Vout.state.Qbot(0),
//	             A2C);
//	PivotMatrix2
//	
////	for (size_t s1=0; s1<qloc[l].size(); ++s1)
////	for (size_t s2=0; s2<qloc[l].size(); ++s2)
////	for (size_t s3=0; s3<qloc[l].size(); ++s3)
////	for (size_t s4=0; s4<qloc[l].size(); ++s4)
////	for (size_t qNL=0; qNL<NL[s1].dim; ++qNL)
////	{
////		auto qALouts = Symmetry::reduceSilent(NL[s1].in[qNL], qloc[l][s2]);
////		for (const auto &qALout : qALouts)
////		{
////			auto itAL = A[GAUGE::L][l][s2].dict.find(qarray2<Symmetry::Nq>{NL[s1].in[qNL], qALout});
////			if (itAL != A[GAUGE::L][l][s2].dict.end())
////			{
////				auto itC = C[l].dict.find(qarray2<Symmetry::Nq>{qALout, qALout});
////				if (itC != C[l].dict.end())
////				{
////					auto qARouts = Symmetry::reduceSilent(qALout, qloc[l][s4]);
////					for (const auto &qARout : qARouts)
////					{
////						auto itAR = A[GAUGE::R][l][s4].dict.find(qarray2<Symmetry::Nq>{qALout, qARout});
////						if (itAR != A[GAUGE::R][l][s4].dict.end())
////						{
////							auto qNRins = Symmetry::reduceSilent(qARout, Symmetry::flip(qloc[l][s3]));
////							for (const auto &qNRin : qNRins)
////							{
////								auto itNR = NR[s3].dict.find(qarray2<Symmetry::Nq>{qNRin, qARout});
////								if (itNR != NR[s3].dict.end())
////								{
////									size_t r = s1 + qloc[l].size()*s3;
////									size_t c = s2 + qloc[l].size()*s4;
////									
////									double factor = (r==c)? h2site[s1][s2][s3][s4]-e : h2site[s1][s2][s3][s4];
////									MatrixType Mtmp = factor * 
////									                  NL[s1].block[qNL].adjoint() *
////									                  A[GAUGE::L][l][s2].block[itAL->second] * 
////									                  C[l].block[itC->second] * 
////									                  A[GAUGE::R][l][s4].block[itAR->second] * 
////									                  NR[s3].block[itNR->second].adjoint();
////									
////									qarray2<Symmetry::Nq> quple = {NL[s1].out[qNL], NR[s3].in[itNR->second]};
////									
////									if (Mtmp.size() != 0 and NL[s1].out[qNL] == NR[s3].in[itNR->second])
////									{
////										auto it = NAAN.dict.find(quple);
////										
////										if (it != NAAN.dict.end())
////										{
////											if (NAAN.block[it->second].rows() != Mtmp.rows() and
////											    NAAN.block[it->second].cols() != Mtmp.cols())
////											{
////												NAAN.block[it->second] = Mtmp;
////											}
////											else
////											{
////												NAAN.block[it->second] += Mtmp;
////											}
////										}
////										else
////										{
////											NAAN.push_back(quple, Mtmp);
////										}
////									}
////								}
////							}
////						}
////					}
////				}
////			}
////		}
////	}
//	
//	for (size_t s1=0; s1<qloc[l].size(); ++s1)
//	for (size_t s2=0; s2<qloc[l].size(); ++s2)
//	for (size_t s3=0; s3<qloc[l].size(); ++s3)
//	for (size_t s4=0; s4<qloc[l].size(); ++s4)
//	for (size_t qNL=0; qNL<NL[s1].dim; ++qNL)
//	{
//		auto qALouts = Symmetry::reduceSilent(NL[s1].in[qNL], qloc[l][s2]);
//		for (const auto &qALout : qALouts)
//		{
//			auto itAL = A[GAUGE::L][l][s2].dict.find(qarray2<Symmetry::Nq>{NL[s1].in[qNL], qALout});
//			if (itAL != A[GAUGE::L][l][s2].dict.end())
//			{
//				auto qACouts = Symmetry::reduceSilent(qALout, qloc[l][s4]);
//				for (const auto &qACout : qACouts)
//				{
//					auto itAC = A[GAUGE::C][l][s4].dict.find(qarray2<Symmetry::Nq>{qALout, qACout});
//					if (itAC != A[GAUGE::C][l][s4].dict.end())
//					{
//						auto qNRins = Symmetry::reduceSilent(qACout, Symmetry::flip(qloc[l][s3]));
//						for (const auto &qNRin : qNRins)
//						{
//							auto itNR = NR[s3].dict.find(qarray2<Symmetry::Nq>{qNRin, qACout});
//							if (itNR != NR[s3].dict.end())
//							{
//								size_t r = s1 + qloc[l].size()*s3;
//								size_t c = s2 + qloc[l].size()*s4;
//								
//								double factor = (r==c)? h2site[s1][s2][s3][s4]-e : h2site[s1][s2][s3][s4];
//								MatrixType Mtmp = factor * 
//								                  NL[s1].block[qNL].adjoint() *
//								                  A[GAUGE::L][l][s2].block[itAL->second] * 
//								                  A[GAUGE::C][l][s4].block[itAC->second] * 
//								                  NR[s3].block[itNR->second].adjoint();
//								
//								qarray2<Symmetry::Nq> quple = {NL[s1].out[qNL], NR[s3].in[itNR->second]};
//								
//								if (Mtmp.size() != 0 and NL[s1].out[qNL] == NR[s3].in[itNR->second])
//								{
//									auto it = NAAN.dict.find(quple);
//									
//									if (it != NAAN.dict.end())
//									{
//										if (NAAN.block[it->second].rows() != Mtmp.rows() and
//										    NAAN.block[it->second].cols() != Mtmp.cols())
//										{
//											NAAN.block[it->second] = Mtmp;
//										}
//										else
//										{
//											NAAN.block[it->second] += Mtmp;
//										}
//									}
//									else
//									{
//										NAAN.push_back(quple, Mtmp);
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	
////	cout << "NAAN=" << endl;
////	cout << NAAN.print(true,15) << endl;
//	double normsum = 0;
//	for (size_t q=0; q<NAAN.dim; ++q)
//	{
//		cout << "q=" << NAAN.in[q] << ", " << NAAN.out[q] << ", norm=" << NAAN.block[q].norm() << endl;
//		normsum += NAAN.block[q].norm();
//	}
//	cout << "normsum=" << normsum << ", sqrt(normsum)=" << sqrt(normsum) << endl;
//	
//	Biped<Symmetry,MatrixType> U, Vdag;
//	for (size_t q=0; q<NAAN.dim; ++q)
//	{
//		JacobiSVD<MatrixType> Jack(NAAN.block[q],ComputeThinU|ComputeThinV);
//		
////		size_t Nret = (Jack.singularValues().array() > this->eps_svd).count();
//		size_t Nret = Jack.singularValues().rows();
//		Nret = min(DeltaD, Nret);
//		
//		U.push_back(NAAN.in[q], NAAN.out[q], Jack.matrixU().leftCols(Nret));
//		Vdag.push_back(NAAN.in[q], NAAN.out[q], Jack.matrixV().adjoint().topRows(Nret));
//	}
//	
//	vector<Biped<Symmetry,MatrixType> > P(qloc[l].size());
//	
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		P[s] = Vdag.contract(NR[s]);
//	}
//	
//	for (size_t s=0; s<qloc[loc].size(); ++s)
//	for (size_t qP=0; qP<P[s].size(); ++qP)
//	{
//		qarray2<Symmetry::Nq> quple = {P[s].in[qP], P[s].out[qP]};
//		auto qA = A[GAUGE::R][l][s].dict.find(quple);
//		
//		if (qA != A[GAUGE::R][l][s].dict.end())
//		{
//			addBottom(P[s].block[qP], A[GAUGE::R][l][s].block[qA->second]);
//		}
//		else
//		{
//			if (inbase[loc].find(P[s].in[qP]))
//			{
//				MatrixType Mtmp(inbase[loc].inner_dim(P[s].in[qP]), P[s].block[qP].cols());
//				Mtmp.setZero();
//				addBottom(P[s].block[qP], Mtmp);
//				A[GAUGE::R][l][s].push_back(quple, Mtmp);
//			}
//			else
//			{
////				if (loc != 0)
////				{
////					bool BLOCK_INSERTED_AT_LOC = false;
////					
////					for (size_t qin=0; qin<inbase[loc-1].Nq(); ++qin)
////					for (size_t sprev=0; sprev<qloc[loc-1].size(); ++sprev)
////					{
////						auto qCandidates = Symmetry::reduceSilent(inbase[loc-1][qin], qloc[loc-1][sprev]);
////						auto it = find(qCandidates.begin(), qCandidates.end(), P[s].in[qP]);
////						
////						if (it != qCandidates.end())
////						{
////							if (!BLOCK_INSERTED_AT_LOC)
////							{
////								A[GAUGE::R][l][s].push_back(quple, P[s].block[qP]);
////								BLOCK_INSERTED_AT_LOC = true;
////							}
////							MatrixType Mtmp(inbase[loc-1].inner_dim(inbase[loc-1][qin]), P[s].block[qP].rows());
////							Mtmp.setZero();
////							A[loc-1][sprev].try_push_back(inbase[loc-1][qin], P[s].in[qP], Mtmp);
////						}
////					}
////				}
////				else
//				{
//					if (P[s].in[qP] == Symmetry::qvacuum())
//					{
//						A[GAUGE::R][l][s].push_back(quple, P[s].block[qP]);
//					}
//				}
//			}
//		}
//	}
//	
//	P.clear();
//	P.resize(qloc[l].size());
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		P[s] = NL[s].contract(U);
//	}
//	
//	for (size_t s=0; s<qloc[loc].size(); ++s)
//	for (size_t qP=0; qP<P[s].size(); ++qP)
//	{
//		qarray2<Symmetry::Nq> quple = {P[s].in[qP], P[s].out[qP]};
//		auto qA = A[GAUGE::L][l][s].dict.find(quple);
//		
//		if (qA != A[GAUGE::L][l][s].dict.end())
//		{
//			addRight(P[s].block[qP], A[GAUGE::L][l][s].block[qA->second]);
//		}
//		else
//		{
//			if (outbase[loc].find(P[s].out[qP]))
//			{
//				MatrixType Mtmp(P[s].block[qP].rows(), outbase[loc].inner_dim(P[s].out[qP]));
//				Mtmp.setZero();
//				addRight(P[s].block[qP], Mtmp);
//				A[GAUGE::L][l][s].push_back(quple, Mtmp);
//			}
//			else
//			{
////				if (loc != this->N_sites-1)
////				{
////					bool BLOCK_INSERTED_AT_LOC = false;
////					
////					for (size_t qout=0; qout<outbase[loc+1].Nq(); ++qout)
////					for (size_t snext=0; snext<qloc[loc+1].size(); ++snext)
////					{
////						auto qCandidates = Symmetry::reduceSilent(outbase[loc+1][qout], Symmetry::flip(qloc[loc+1][snext]));
////						auto it = find(qCandidates.begin(), qCandidates.end(), P[s].out[qP]);
////						
////						if (it != qCandidates.end())
////						{
////							if (!BLOCK_INSERTED_AT_LOC)
////							{
////								A[GAUGE::L][l][s].push_back(quple, P[s].block[qP]);
////								BLOCK_INSERTED_AT_LOC = true;
////							}
////							MatrixType Mtmp(P[s].block[qP].cols(), outbase[loc+1].inner_dim(outbase[loc+1][qout]));
////							Mtmp.setZero();
////							A[loc+1][snext].try_push_back(P[s].out[qP], outbase[loc+1][qout], Mtmp);
////						}
////					}
////				}
////				else
//				{
//					if (P[s].out[qP] == Qtarget())
//					{
//						A[GAUGE::L][l][s].push_back(quple, P[s].block[qP]);
//					}
//				}
//			}
//		}
//	}
//	
//	map<qarray<Symmetry::Nq>,int> ALcols;
//	map<qarray<Symmetry::Nq>,int> ARrows;
//	
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	for (size_t q=0; q<A[GAUGE::L][l][s].dim; ++q)
//	{
//		if (A[GAUGE::L][l][s].block[q].cols() > A[GAUGE::L][l][s].block[q].rows())
//		{
//			size_t Delta = A[GAUGE::L][l][s].block[q].cols() - A[GAUGE::L][l][s].block[q].rows();
//			cout << "Delta=" << Delta << endl;
//			A[GAUGE::L][l][s].block[q].conservativeResize(A[GAUGE::L][l][s].block[q].cols(), A[GAUGE::L][l][s].block[q].cols());
//			A[GAUGE::L][l][s].block[q].bottomRows(Delta).setZero();
//		}
//		
//		ALcols[A[GAUGE::L][l][s].out[q]] = A[GAUGE::L][l][s].block[q].cols();
//	}
//	
//	cout << "AL=" << endl;
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		cout << "s=" << s << endl;
//		cout << A[GAUGE::L][l][s].print(false) << endl;
//	}
//	
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	for (size_t q=0; q<A[GAUGE::R][l][s].dim; ++q)
//	{
//		if (A[GAUGE::R][l][s].block[q].rows() > A[GAUGE::R][l][s].block[q].cols())
//		{
//			size_t Delta = A[GAUGE::R][l][s].block[q].rows() - A[GAUGE::R][l][s].block[q].cols();
//			cout << "Delta=" << Delta << endl;
//			A[GAUGE::R][l][s].block[q].conservativeResize(A[GAUGE::R][l][s].block[q].rows(), A[GAUGE::R][l][s].block[q].rows());
//			A[GAUGE::R][l][s].block[q].rightCols(Delta).setZero();
//		}
//		
//		ARrows[A[GAUGE::R][l][s].in[q]] = A[GAUGE::R][l][s].block[q].rows();
//	}
//	
//	cout << "AR=" << endl;
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		cout << "s=" << s << endl;
//		cout << A[GAUGE::R][l][s].print(false) << endl;
//	}
//	
//	for (size_t q=0; q<C[l].dim; ++q)
//	{
//		qarray<Symmetry::Nq> qC = C[l].in[q];
//		int r = ALcols[qC];
//		int c = ARrows[qC];
//		int dr = r-C[l].block[q].rows();
//		int dc = c-C[l].block[q].cols();
//		
//		cout << "q=" << C[l].in[q] << ", r=" << r << ", c=" << c << ", dr=" << dr << ", dc=" << dc << endl;
//		C[l].block[q].conservativeResize(r,c);
//		
//		C[l].block[q].bottomRows(dr).setZero();
//		C[l].block[q].rightCols(dc).setZero();
//	}
//	
//	cout << "C=" << endl;
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		cout << C[l].print(false) << endl;
//	}
//	
////	for (size_t s=0; s<qloc[l].size(); ++s)
////	for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
////	{
////		int r = ALcols[A[GAUGE::C][l][s].in[q]];
////		int c = ARrows[A[GAUGE::C][l][s].out[q]];
////		int dr = r-A[GAUGE::C][l][s].block[q].rows();
////		int dc = c-A[GAUGE::C][l][s].block[q].cols();
////		
////		A[GAUGE::C][l][s].block[q].conservativeResize(r,c);
////		A[GAUGE::C][l][l].block[q].bottomRows(dr).setZero();
////		A[GAUGE::C][l][l].block[q].rightCols(dc).setZero();
////	}
//	
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		A[GAUGE::C][l][s] = A[GAUGE::L][l][s].contract(C[l]);
//	}
//	
//	cout << "AC=" << endl;
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		cout << "s=" << s << endl;
//		cout << A[GAUGE::C][l][s].print(false) << endl;
//	}
//	
//	update_inbase();
//	update_outbase();
//}

#endif
