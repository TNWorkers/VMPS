#ifndef VANILLA_UMPS
#define VANILLA_UMPS

#if !defined DONT_USE_LAPACK_SVD || !defined DONT_USE_LAPACK_QR
	#include "LapackWrappers.h"
#endif

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
#if !defined DONT_USE_LAPACK_SVD || !defined DONT_USE_LAPACK_QR
	#include "LapackWrappers.h"
#endif
#include "PolychromaticConsole.h" // from HELPERS
#include "RandomVector.h" // from LANCZOS

/**Uniform Matrix Product State. Analogue of the Mps class. Currently without symmetries, the template parameter \p Symmetry can only be \p Sym::U0.
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
	
	Umps<Symmetry,Scalar>(){};
	
	/**Constructs a UMPS with fixed bond dimension with the info from the Hamiltonian.*/
	template<typename Hamiltonian> Umps (const Hamiltonian &H, size_t L_input, size_t Dmax, size_t Nqmax);
	
	/**Constructs a UMPS with fixed bond dimension with a given basis.*/
	Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, size_t L_input, size_t Dmax, size_t Nqmax);
	
	/**\describe_info*/
	string info() const;
	
	void graph (string filename) const;
	
	/**Tests the orthogonality of the UMPS.*/
	string test_ortho (double tol=1e-10) const;
	
	/**Sets all matrices  \f$A_L\f$, \f$A_R\f$, \f$A_C\f$, \f$C\f$) to random using boost's uniform distribution from -1 to 1.*/
	void setRandom();
	
	/**Resizes all containers to \p N_sites, the bond dimension to \p Dmax and sets all quantum numbers to vacuum.*/
	void resize (size_t Dmax_input, size_t Nqmax_input);
	
	/**Calculates \f$A_L\f$ and \f$A_R\f$ from \f$A_C\f$ and \f$C\f$ at site \p loc using SVD (eq. 19,20). Is supposed to be optimal, but not accurate. Calculates the singular values along the way.*/
	void svdDecompose (size_t loc);
	
	/**Calculates \f$A_L\f$ and \f$A_R\f$ from \f$A_C\f$ and \f$C\f$ at site \p loc using the polar decomposition (eq. 21,22). Is supposed to be non-optimal, but accurate.*/
	void polarDecompose (size_t loc);
	
	/**Returns the singular values at site \p loc.*/
	VectorXd singularValues (size_t loc=0);
	
	/**Returns the entropy at site \p loc.*/
	double entropy (size_t loc=0);
	
	/**Returns the local basis at site \p loc.*/
	inline vector<qarray<Symmetry::Nq> > locBasis (size_t loc) const {return qloc[loc];}
	
	/**Returns the whole local basis at site \p loc.*/
	inline vector<vector<qarray<Symmetry::Nq> > > locBasis()   const {return qloc;}
	
	/**Returns the amount of rows of first tensor. Useful for environment tensors in contractions.*/
	size_t get_frst_rows() const {return A[GAUGE::C][0][0].block[0].rows();}
	
	/**Returns the amount of columns of last tensor. Useful for environment tensors in contractions.*/
	size_t get_last_cols() const {return A[GAUGE::C][N_sites][0].block[0].cols();}
	
	/**Returns the amount of sites, i.e. the size of the unit cell.*/
	size_t length() const {return N_sites;}
	
	/**Calculates the left and right decomposition error as \f$\epsilon_L=\big|A_C-A_LC\big|^2\f$ and \f$\epsilon_R=\big|A_C-CA_R\big|^2\f$ (eq. 18).*/
	void calc_epsLRsq (size_t loc, double &epsL, double &epsR);
	
	size_t calc_Dmax() const;
	size_t calc_Mmax() const;
	size_t calc_fullMmax() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit) const;
	
	/**Calculates the scalar product with another UMPS by finding the dominant eigenvalue of the transfer matrix. 
	See arXiv:0804.2509 and Phys. Rev. B 78, 155117.*/
	complex<double> dot (const Umps<Symmetry,Scalar> &Vket) const;
	
	/**Returns \f$A_L\f$, \f$A_R\f$ or \f$A_C\f$ at site \p loc as const ref.*/
	const vector<Biped<Symmetry,MatrixType> > &A_at (GAUGE::OPTION g, size_t loc) const {return A[g][loc];};
	
private:
	
	size_t N_sites;
	size_t Dmax, Nqmax;
	double eps_svd = 1e-7;
	size_t N_sv;
	
	void calc_singularValues (size_t loc=0);
	
	// sets of all unique incoming & outgoing indices for convenience
	vector<vector<qarray<Symmetry::Nq> > > inset;
	vector<vector<qarray<Symmetry::Nq> > > outset;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	std::array<string,Symmetry::Nq> qlabel = {};
	
	// UMPS-tensors in the three gauges L,R,C
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> A; // A[L/R/C][l][s].block[q]
	
	// center matrix
	vector<Biped<Symmetry,MatrixType> >                        C; // zero-site part C[l]
	
	// null space (see eq. 25 and surrounding text)
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> N; // N[L/R/C][l][s].block[q]
	
//	vector<VectorXd> Csingular;
	VectorXd S;
	
	// Bases on all ingoing and outgoing legs of the MPS
	vector<Qbasis<Symmetry> > inbase;
	vector<Qbasis<Symmetry> > outbase;
	
	void update_inbase  (size_t loc);
	void update_outbase (size_t loc);
	void update_inbase()  { for(size_t l=0; l<this->N_sites; l++) {update_inbase(l); } }
	void update_outbase() { for(size_t l=0; l<this->N_sites; l++) {update_outbase(l); } }
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
	ss << "S=(" << S.transpose() << "), ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	
	return ss.str();
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
Umps<Symmetry,Scalar>::
Umps (const Hamiltonian &H, size_t L_input, size_t Dmax, size_t Nqmax)
{
	N_sites = L_input;
	qloc = H.locBasis();
	resize(Dmax,Nqmax);
}

template<typename Symmetry, typename Scalar>
Umps<Symmetry,Scalar>::
Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, size_t L_input, size_t Dmax, size_t Nqmax)
{
	N_sites = L_input;
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l) {qloc[l] = qloc_input;}
	resize(Dmax,Nqmax);
}

template<typename Symmetry, typename Scalar>
size_t Umps<Symmetry,Scalar>::
calc_Dmax() const
{
//	size_t res = 0;
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		for (size_t s=0; s<qloc[l].size(); ++s)
//		for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
//		{
//			if (A[GAUGE::C][l][s].block[q].rows()>res) {res = A[GAUGE::C][l][s].block[q].rows();}
//			if (A[GAUGE::C][l][s].block[q].cols()>res) {res = A[GAUGE::C][l][s].block[q].cols();}
//		}
//	}
//	return res;
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (inbase[l].Dmax()  > res) {res = inbase[l].Dmax();}
		if (outbase[l].Dmax() > res) {res = outbase[l].Dmax();}
	}
	return res;
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
calc_Mmax() const
{
//	size_t res = 0;
//	for (size_t l=0; l<N_sites; ++l)
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		size_t Mrows = 0;
//		size_t Mcols = 0;
//		for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
//		{
//			Mrows += A[GAUGE::C][l][s].block[q].rows();
//			Mcols += A[GAUGE::C][l][s].block[q].cols();
//		}
//		if (Mrows>res) {res = Mrows;}
//		if (Mcols>res) {res = Mcols;}
//	}
//	return res;
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
update_inbase (size_t loc)
{
	inbase[loc].clear();
	inbase[loc].pullData(A[GAUGE::C][loc],0);
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
update_outbase (size_t loc)
{
	outbase[loc].clear();
	outbase[loc].pullData(A[GAUGE::C][loc],1);
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
resize (size_t Dmax_input, size_t Nqmax_input)
{
//	Dmax = Dmax_input;
//	
//	for (size_t g=0; g<3; ++g)
//	{
//		A[g].resize(N_sites);
//		for (size_t l=0; l<N_sites; ++l)
//		{
//			A[g][l].resize(qloc[l].size());
//		}
//		
//		N[g].resize(N_sites);
//		for (size_t l=0; l<N_sites; ++l)
//		{
//			N[g][l].resize(qloc[l].size());
//		}
//	}
//	C.resize(N_sites);
//	inset.resize(N_sites);
//	outset.resize(N_sites);
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		inset[l].push_back(Symmetry::qvacuum());
//		outset[l].push_back(Symmetry::qvacuum());
//	}
//	
//	for (size_t g=0; g<3; ++g)
//	for (size_t l=0; l<N_sites; ++l)
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		A[g][l][s].in.push_back(Symmetry::qvacuum());
//		A[g][l][s].out.push_back(Symmetry::qvacuum());
//		A[g][l][s].dict.insert({qarray2<Symmetry::Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, A[g][l][s].dim});
//		A[g][l][s].dim = 1;
//		A[g][l][s].block.resize(1);
//		
//		N[g][l][s].in.push_back(Symmetry::qvacuum());
//		N[g][l][s].out.push_back(Symmetry::qvacuum());
//		N[g][l][s].dict.insert({qarray2<Symmetry::Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, A[g][l][s].dim});
//		N[g][l][s].dim = 1;
//		N[g][l][s].block.resize(1);
//	}
//	
//	for (size_t g=0; g<3; ++g)
//	for (size_t l=0; l<N_sites; ++l)
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		A[g][l][s].block[0].resize(Dmax,Dmax);
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		C[l].in.push_back(Symmetry::qvacuum());
//		C[l].out.push_back(Symmetry::qvacuum());
//		C[l].dict.insert({qarray2<Symmetry::Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, C[l].dim});
//		C[l].dim = 1;
//		C[l].block.resize(1);
//		C[l].block[0].resize(Dmax,Dmax);
//	}
//	
//	Csingular.clear();
//	Csingular.resize(N_sites);
//	S.resize(N_sites);
	
	Dmax = Dmax_input;
	Nqmax = Nqmax_input;
	
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
		// sort the vector first according to the distance to mean
		sort(out.begin(),out.end(),[this] (qarray<Nq> q1, qarray<Nq> q2)
		{
			VectorXd dist_q1(Nq);
			VectorXd dist_q2(Nq);
			for (size_t q=0; q<Nq; q++)
			{
				double Delta = 1.; // QinTop[loc][q] - QinBot[loc][q];
				dist_q1(q) = (q1[q]) / Delta;
				dist_q2(q) = (q2[q]) / Delta;
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
			for (const auto &t:qoutset[(l-1)%N_sites])
			{
				if (qinset[l].size() < Nqmax)
				{
					qinset[l].insert(t);
				}
			}
			inSize[l] = qinset[l].size();
			
			vector<qarray<Symmetry::Nq> > qinvec(qinset[l].size());
			copy(qinset[l].begin(), qinset[l].end(), qinvec.begin());
			
			auto tmp = Symmetry::reduceSilent(qinvec, qloc[l]);
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
				auto qouts = Symmetry::reduceSilent(qloc[l], qin);
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
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[g][l][s].dim; ++q)
	{
		A[g][l][s].block[q].resize(Dmax,Dmax);
	}
//	for (size_t l=0; l<N_sites; ++l)
//	for (size_t s=0; s<qloc[l].size(); ++s)
//	{
//		cout << "l=" << l << ", s=" << s << endl;
//		cout << A[GAUGE::C][l][s].print(false) << endl;
//	}
	
	update_inbase();
	update_outbase();
	
	for (size_t l=0; l<N_sites; ++l)
	for (size_t qout=0; qout<outbase[l].Nq(); ++qout)
	{
		C[l].push_back(qarray2<Symmetry::Nq>{outbase[l][qout], outbase[l][qout]}, Mtmp);
	}
	
//	Csingular.clear();
//	Csingular.resize(N_sites);
	S.resize(N_sites);
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
setRandom()
{
	for (size_t l=0; l<N_sites; ++l)
	for (size_t q=0; q<C[l].dim; ++q)
	for (size_t a1=0; a1<C[l].block[q].rows(); ++a1)
	for (size_t a2=0; a2<C[l].block[q].cols(); ++a2)
	{
		C[l].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
	}
	
	// normalize the centre matrices for proper wavefunction norm: Tr(C*C†)=1
	for (size_t l=0; l<N_sites; ++l)
	{
		C[l] = 1./sqrt((C[l].contract(C[l].adjoint())).trace()) * C[l];
	}
	
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
	for (size_t a1=0; a1<A[GAUGE::C][l][s].block[q].rows(); ++a1)
	for (size_t a2=0; a2<A[GAUGE::C][l][s].block[q].cols(); ++a2)
	{
		A[GAUGE::C][l][s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
	}
	
	calc_singularValues();
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
	ss << "label=\"UMPS: cell size=" << N_sites << ", Q=(";
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
			A_CHECK[q]     = Test.block[q].template lpNorm<Infinity>()<tol ? true : false;
			A_infnorm[q]   = Test.block[q].template lpNorm<Infinity>();
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
			sout << TCOLOR(GREEN);
			sout << normal_token[2]; // M
		}
	}
	
	sout << TCOLOR(BLACK);
	sout << ", norm=" << norm.transpose();
	return sout.str();
}

//template<typename Symmetry, typename Scalar>
//complex<double> Umps<Symmetry,Scalar>::
//dot (const Umps<Symmetry,Scalar> &Vket) const
//{
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
//}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_singularValues (size_t loc)
{
	S(loc) = 0;
	
	for (size_t q=0; q<C[loc].dim; ++q)
	{
		JacobiSVD<MatrixType> Jack(C[loc].block[q]);
//		Csingular[loc] += Jack.singularValues();
		size_t Nnz = (Jack.singularValues().array() > 0.).count();
		
		double Scontr = -Symmetry::degeneracy(C[loc].in[q]) * (Jack.singularValues().head(Nnz).array().square() 
		                                              * Jack.singularValues().head(Nnz).array().square().log()).sum();
		cout << "q=" << q << ", in=" << C[loc].in[q] << ", out=" << C[loc].out[q] << ", S=" << Scontr << endl;
		
		S(loc) += -Symmetry::degeneracy(C[loc].in[q]) * (Jack.singularValues().head(Nnz).array().square() 
		                                              * Jack.singularValues().head(Nnz).array().square().log()).sum();
	}
}

//template<typename Symmetry, typename Scalar>
//VectorXd Umps<Symmetry,Scalar>::
//singularValues (size_t loc)
//{
//	assert(loc<N_sites);
//	return Csingular[loc];
//}

template<typename Symmetry, typename Scalar>
double Umps<Symmetry,Scalar>::
entropy (size_t loc)
{
	assert(loc<N_sites);
	return S(loc);
}

// wtf is that?
//MatrixXd gauge (const MatrixXd &U)
//{
//	MatrixXd Mout;
//	Mout.setIdentity(U.rows(),U.cols());
//	for (int i=0; i<U.rows(); ++i)
//	{
//		MatrixXd::Index imax;
//		U.row(i).maxCoeff(&imax);
//		Mout(i,i) *= (U.row(i)(imax)>=0)? 1:-1;
//	}
//	return Mout;
//}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
polarDecompose (size_t loc)
{
	#ifdef DONT_USE_BDCSVD
	JacobiSVD<MatrixType> Jack; // standard SVD
	#else
	BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
	#endif
	
	S(loc) = 0;
	
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
		MatrixType Acmp(Nrows,Ncols);
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
		size_t qC = it->second;
		
		// Check if possible to move outside the loop:
		vector<MatrixType> UC;
		for (size_t q=0; q<C[loc].dim; ++q)
		{
			Jack.compute(C[loc].block[q], ComputeThinU|ComputeThinV);
			UC.push_back(Jack.matrixU() * Jack.matrixV().adjoint());
			
			// Get the singular values and the entropy while at it (C[loc].dim=1 assumed):
//			Csingular[loc] = Jack.singularValues();
			size_t Nnz = (Jack.singularValues().array() > 0).count();
			S(loc) += -Symmetry::degeneracy(C[loc].in[q]) * (Jack.singularValues().head(Nnz).array().square() 
			                                             * Jack.singularValues().head(Nnz).array().square().log()).sum();
		}
		
		// Update AL
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			A[GAUGE::L][loc][svec[i]].block[qvec[i]] = UL.block(stitch,0, Nrowsvec[i],Ncols) * UC[qC].adjoint();
			stitch += Nrowsvec[i];
		}
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
		
		size_t locC = (N_sites==1)? 0 : (loc-1)%N_sites;
		auto it = C[locC].dict.find(quple);
		size_t qC = it->second;
		
		vector<MatrixType> UC;
		
		for (size_t q=0; q<C[locC].dim; ++q)
		{
			Jack.compute(C[locC].block[q], ComputeThinU|ComputeThinV);
			UC.push_back(Jack.matrixU() * Jack.matrixV().adjoint());
		}
		
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

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_epsLRsq (size_t loc, double &epsLsq, double &epsRsq)
{
	epsLsq = 0;
	epsRsq = 0;
	
	for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
	{
		qarray2<Symmetry::Nq> quple = {outbase[loc][qout], outbase[loc][qout]};
		auto it = C[loc].dict.find(quple);
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
		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
			Acmp.block  (stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::L][loc][svec[i]].block[qvec[i]];
			stitch += Nrowsvec[i];
		}
		
		epsLsq += (Aclump-Acmp*C[loc].block[qC]).squaredNorm() * Symmetry::coeff_dot(C[loc].in[qC]);
		
//		BDCSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
////		JacobiSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
//		MatrixType NullSpace = Jack.matrixV().rightCols((qloc[loc].size()-1) * A[GAUGE::C][loc][0].block[0].rows());
//		
//		double epsL_ = (NullSpace.adjoint() * Aclump).norm();
////		double epsL__ = sqrt(Aclump.squaredNorm() - (Acmp*C[loc].block[qC]).squaredNorm());
//		
//		MatrixType B = Aclump-Acmp*C[loc].block[qC];
//		size_t D = qloc[loc].size();
//		size_t M = A[GAUGE::C][loc][0].block[0].rows();
//		
////		double epsL___ = 0;
////		for (size_t s=0; s<qloc[loc].size(); ++s)
////		{
////			epsL___ += (A[GAUGE::C][loc][s].block[0] - A[GAUGE::L][loc][s].block[0] * C[loc].block[0]).squaredNorm();
////		}
////		epsL___ = sqrt(epsL___);
//		
//		cout << "nullspace test: " << (Acmp.adjoint() * NullSpace).norm() << endl;
//		cout << "ortho test: " << (NullSpace.adjoint() * NullSpace - MatrixType::Identity(NullSpace.cols(),NullSpace.cols())).norm() << endl;
//		cout << "norm(NullSpace)=" << NullSpace.norm() << endl;
//		
//		stitch = 0;
//		for (size_t i=0; i<svec.size(); ++i)
//		{
//			N[GAUGE::L][loc][svec[i]].block[0] = NullSpace.block(stitch,0, Nrowsvec[i],NullSpace.cols());
//			stitch += Nrowsvec[i];
//		}
		
//		MatrixType Ntest = N[GAUGE::L][loc][0].block[0].adjoint() * N[GAUGE::L][loc][0].block[0];
//		for (size_t s=1; s<qloc[loc].size(); ++s)
//		{
//			Ntest += N[GAUGE::L][loc][s].block[0].adjoint() * N[GAUGE::L][loc][s].block[0];
//		}
//		cout << "NtestL1=" << (Ntest-MatrixType::Identity(Ntest.rows(),Ntest.cols())).norm() << endl;
//		
//		Ntest = N[GAUGE::L][loc][0].block[0].adjoint() * A[GAUGE::L][loc][0].block[0];
//		for (size_t s=1; s<qloc[loc].size(); ++s)
//		{
//			Ntest += N[GAUGE::L][loc][s].block[0].adjoint() * A[GAUGE::L][loc][s].block[0];
//		}
//		cout << "NtestL2=" << Ntest.norm() << endl;
//		
//		MatrixType U(D*M,D*M);
//		U.leftCols(M) = Acmp;
//		U.rightCols((D-1)*M) = NullSpace;
//		
//		cout << "Utest1=" << (U.adjoint() * U - MatrixType::Identity(D*M,D*M)).norm() << endl;
//		cout << "Utest2=" << (U * U.adjoint() - MatrixType::Identity(D*M,D*M)).norm() << endl;
//		cout << "Bnorm=" << B.norm() << endl;
//		cout << "UBnorm=" << (U.adjoint()*B).norm() << "\t" << (U*B).norm() << endl;
//		cout << (Acmp.adjoint()*B).norm() << endl;
//		cout << (NullSpace.adjoint() * B).norm() << endl;
//		
//		MatrixType Cprime = A[GAUGE::L][loc][0].block[0].adjoint() * A[GAUGE::C][loc][0].block[0];
//		for (size_t s=1; s<D; ++s)
//		{
//			Cprime += A[GAUGE::L][loc][s].block[0].adjoint() * A[GAUGE::C][loc][s].block[0];
//		}
//		cout << "Ctest=" << (Cprime-C[loc].block[0]).norm() << endl;
//		
//		cout << endl << "epsL = " << sqrt(epsLsq) << "\t" << epsL_ << endl << endl;
	}
	
	for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
	{
		qarray2<Symmetry::Nq> quple = {inbase[loc][qin], inbase[loc][qin]};
		size_t locC = (N_sites==1)? 0 : (loc-1)%N_sites;
		auto it = C[locC].dict.find(quple);
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
		
		epsRsq += (Aclump-C[locC].block[qC]*Acmp).squaredNorm() * Symmetry::coeff_dot(C[locC].in[qC]);
		
//		BDCSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
////		JacobiSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
//		MatrixType NullSpace = Jack.matrixU().adjoint().bottomRows((qloc[loc].size()-1) * A[GAUGE::C][loc][0].block[0].rows());
		
//		stitch = 0;
//		for (size_t i=0; i<svec.size(); ++i)
//		{
//			N[GAUGE::R][loc][svec[i]].block[0] = NullSpace.block(0,stitch, NullSpace.rows(),Ncolsvec[i]);
//			stitch += Ncolsvec[i];
//		}
		
//		MatrixType Ntest = N[GAUGE::R][loc][0].block[0] * N[GAUGE::R][loc][0].block[0].adjoint();
//		for (size_t s=1; s<qloc[loc].size(); ++s)
//		{
//			Ntest += N[GAUGE::R][loc][s].block[0] * N[GAUGE::R][loc][s].block[0].adjoint();
//		}
//		cout << "NtestR1=" << (Ntest-MatrixType::Identity(Ntest.rows(),Ntest.cols())).norm() << endl;
//		
//		Ntest = A[GAUGE::R][loc][0].block[0] * N[GAUGE::R][loc][0].block[0].adjoint();
//		for (size_t s=1; s<qloc[loc].size(); ++s)
//		{
//			Ntest += A[GAUGE::R][loc][s].block[0] * N[GAUGE::R][loc][s].block[0].adjoint();
//		}
//		cout << "NtestR2=" << Ntest.norm() << endl;
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
svdDecompose (size_t loc)
{
	for (size_t qout=0; qout<outbase[loc].Nq(); ++qout)
	{
		qarray2<Symmetry::Nq> quple = {outbase[loc][qout], outbase[loc][qout]};
		auto it = C[loc].dict.find(quple);
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
		Aclump.setZero();
		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			Aclump.block(stitch,0, Nrowsvec[i],Ncols) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
			stitch += Nrowsvec[i];
		}
		
		Aclump *= C[loc].block[qC].adjoint();
		
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
	
	for (size_t qin=0; qin<inbase[loc].Nq(); ++qin)
	{
		qarray2<Symmetry::Nq> quple = {inbase[loc][qin], inbase[loc][qin]};
		size_t locC = (N_sites==1)? 0 : (loc-1)%N_sites;
		auto it = C[locC].dict.find(quple);
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
		Aclump.setZero();
		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::C][loc][svec[i]].block[qvec[i]]*
					                                    Symmetry::coeff_leftSweep(
					                                         A[GAUGE::C][loc][svec[i]].out[qvec[i]],
					                                         A[GAUGE::C][loc][svec[i]].in[qvec[i]],
					                                         qloc[loc][svec[i]]);
			stitch += Ncolsvec[i];
		}
		
		Aclump = C[locC].block[qC].adjoint() * Aclump;
		
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
			                                            A[GAUGE::C][loc][svec[i]].out[qvec[i]],
			                                            A[GAUGE::C][loc][svec[i]].in[qvec[i]],
			                                            qloc[loc][svec[i]]);
			stitch += Ncolsvec[i];
		}
	}
	
	calc_singularValues(loc);
}

#endif
