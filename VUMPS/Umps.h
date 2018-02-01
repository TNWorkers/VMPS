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
typedef Matrix<complex<double>,Dynamic,Dynamic> CMatrixType;
typedef Matrix<Scalar,Dynamic,1>       VectorType;

template<typename Symmetry_, typename MpHamiltonian, typename Scalar_> friend class VumpsSolver;

public:
	
	/**Does nothing.*/
	Umps<Symmetry,Scalar>(){};
	
	/**Constructs a UMPS with fixed bond dimension with the info from the Hamiltonian.*/
	template<typename Hamiltonian> Umps (const Hamiltonian &H, size_t L_input, size_t Dmax, qarray<Symmetry::Nq> Qtot_input);
	
	/**Constructs a UMPS with fixed bond dimension with a given basis.*/
	Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, size_t L_input, size_t Dmax, qarray<Symmetry::Nq> Qtot_input);
	
	/**\describe_info*/
	string info() const;
	
	/**Tests the orthogonality of the UMPS.*/
	string test_ortho (double tol=1e-10) const;
	
	/**Sets all matrices  \f$A_L\f$, \f$A_R\f$, \f$A_C\f$, \f$C\f$) to random using boost's uniform distribution from -1 to 1.*/
	void setRandom();
	
	/**Resizes all containers to \p N_sites, the bond dimension to \p Dmax and sets all quantum numbers to vacuum.*/
	void resize (size_t Dmax);
	
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
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit) const;
	
	/**Calculates the scalar product with another UMPS by finding the dominant eigenvalue of the transfer matrix. 
	See arXiv:0804.2509 and Phys. Rev. B 78, 155117.*/
	complex<double> dot (const Umps<Symmetry,Scalar> &Vket) const;
	
	/**Returns \f$A_L\f$, \f$A_R\f$ or \f$A_C\f$ at site \p loc as const ref.*/
	const vector<Biped<Symmetry,MatrixType> > &A_at (GAUGE::OPTION g, size_t loc) const {return A[g][loc];};
	
private:
	
	size_t N_sites;
	size_t Dmax;
	double eps_svd = 1e-7;
	size_t N_sv;
	
	void calc_singularValues (size_t loc=0);
	
	// sets of all unique incoming & outgoing indices for convenience
	vector<vector<qarray<Symmetry::Nq> > > inset;
	vector<vector<qarray<Symmetry::Nq> > > outset;
	
	vector<vector<qarray<Symmetry::Nq> > > qloc;
	std::array<string,Symmetry::Nq> qlabel = {};
	qarray<Symmetry::Nq> Qtot;
	
	// UMPS-tensors in the three gauges
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> A; // A[L/R/C][l][s].block[q]
	
	// center matrix
	vector<Biped<Symmetry,MatrixType> >                        C; // zero-site part C[l]
	
	// null space (eq. 25 and surrounding text)
	std::array<vector<vector<Biped<Symmetry,MatrixType> > >,3> N; // N[L/R/C][l][s].block[q]
	
	vector<VectorXd> Csingular;
	VectorXd S;
};

template<typename Symmetry, typename Scalar>
string Umps<Symmetry,Scalar>::
info() const
{
	stringstream ss;
	ss << "Umps: ";
	
//	if (Nq != 0)
//	{
//		ss << "(";
//		for (size_t q=0; q<Symmetry; ++q)
//		{
//			ss << qlabel[q];
//			if (q!=Nq-1) {ss << ",";}
//		}
//		ss << ")=" << format(Qtot) << ", ";
//	}
//	else
//	{
		ss << "no symmetries, ";
//	}
	
	ss << "Lcell=" << N_sites << ", ";
	ss << "Mmax=" << calc_Mmax() << " (Dmax=" << calc_Dmax() << "), ";
//	ss << "Nqmax=" << calc_Nqmax() << ", ";
//	ss << "trunc_weight=" << truncWeight.sum() << ", ";
	ss << "S=(" << S.transpose() << "), ";
	ss << "mem=" << round(memory(GB),3) << "GB";
//	"overhead=" << round(overhead(MB),3) << "MB";
	
	return ss.str();
}

template<typename Symmetry, typename Scalar>
template<typename Hamiltonian>
Umps<Symmetry,Scalar>::
Umps (const Hamiltonian &H, size_t L_input, size_t Dmax, qarray<Symmetry::Nq> Qtot_input)
{
//	format = H.format;
//	qlabel = H.qlabel;
//	N_legs = H.width();
//	outerResize<typename Hamiltonian::qarrayIterator>(H.length(), H.locBasis(), Qtot_input);
	N_sites = L_input;
	qloc = H.locBasis();
	
//	outerResize(H.length(), H.locBasis(), Qtot_input);
	resize(Dmax);
}

template<typename Symmetry, typename Scalar>
Umps<Symmetry,Scalar>::
Umps (const vector<qarray<Symmetry::Nq> > &qloc_input, size_t L_input, size_t Dmax, qarray<Symmetry::Nq> Qtot_input)
{
	N_sites = L_input;
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l) {qloc[l] = qloc_input;}
	resize(Dmax);
}

template<typename Symmetry, typename Scalar>
size_t Umps<Symmetry,Scalar>::
calc_Dmax() const
{
	size_t res = 0;
	for (size_t l=0; l<N_sites; ++l)
	{
		for (size_t s=0; s<qloc[l].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
		{
			if (A[GAUGE::C][l][s].block[q].rows()>res) {res = A[GAUGE::C][l][s].block[q].rows();}
			if (A[GAUGE::C][l][s].block[q].cols()>res) {res = A[GAUGE::C][l][s].block[q].cols();}
		}
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
	size_t res = 0;
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		size_t Mrows = 0;
		size_t Mcols = 0;
		for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
		{
			Mrows += A[GAUGE::C][l][s].block[q].rows();
			Mcols += A[GAUGE::C][l][s].block[q].cols();
		}
		if (Mrows>res) {res = Mrows;}
		if (Mcols>res) {res = Mcols;}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
resize (size_t Dmax_input)
{
	Dmax = Dmax_input;
	
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
	inset.resize(N_sites);
	outset.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		inset[l].push_back(Symmetry::qvacuum());
		outset[l].push_back(Symmetry::qvacuum());
	}
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[g][l][s].in.push_back(Symmetry::qvacuum());
		A[g][l][s].out.push_back(Symmetry::qvacuum());
		A[g][l][s].dict.insert({qarray2<Symmetry::Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, A[g][l][s].dim});
		A[g][l][s].dim = 1;
		A[g][l][s].block.resize(1);
		
		N[g][l][s].in.push_back(Symmetry::qvacuum());
		N[g][l][s].out.push_back(Symmetry::qvacuum());
		N[g][l][s].dict.insert({qarray2<Symmetry::Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, A[g][l][s].dim});
		N[g][l][s].dim = 1;
		N[g][l][s].block.resize(1);
	}
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[g][l][s].block[0].resize(Dmax,Dmax);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		C[l].in.push_back(Symmetry::qvacuum());
		C[l].out.push_back(Symmetry::qvacuum());
		C[l].dict.insert({qarray2<Symmetry::Nq>{Symmetry::qvacuum(),Symmetry::qvacuum()}, C[l].dim});
		C[l].dim = 1;
		C[l].block.resize(1);
		C[l].block[0].resize(Dmax,Dmax);
	}
	
	Csingular.clear();
	Csingular.resize(N_sites);
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
	for (size_t q=0; q<C[l].dim; ++q)
	{
		C[l].block[q] /= sqrt((C[l].block[q] * C[l].block[q].adjoint()).trace());
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
string Umps<Symmetry,Scalar>::
test_ortho (double tol) const
{
	string sout = "";
	std::array<string,4> normal_token  = {"A","B","M","X"};
	std::array<string,4> special_token = {"\e[4mA\e[0m","\e[4mB\e[0m","\e[4mM\e[0m","\e[4mX\e[0m"};
	
	for (int l=0; l<this->N_sites; ++l)
	{
		// check for A
		Biped<Symmetry,MatrixType> Test = A[GAUGE::L][l][0].adjoint() * A[GAUGE::L][l][0];
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[GAUGE::L][l][s].adjoint() * A[GAUGE::L][l][s];
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
		Test = A[GAUGE::R][l][0] * A[GAUGE::R][l][0].adjoint();
		for (size_t s=1; s<qloc[l].size(); ++s)
		{
			Test += A[GAUGE::R][l][s] * A[GAUGE::R][l][s].adjoint();
		}
		
		vector<bool> B_CHECK(Test.dim);
		vector<double> B_infnorm(Test.dim);
		for (size_t q=0; q<Test.dim; ++q)
		{
			Test.block[q] -= MatrixType::Identity(Test.block[q].rows(), Test.block[q].cols());
			B_CHECK[q]     = Test.block[q].template lpNorm<Infinity>()<tol ? true : false;
			B_infnorm[q]   = Test.block[q].template lpNorm<Infinity>();
		}
		
		vector<double> norms;
		norms.resize(C[l].dim);
		for (size_t q=0; q<C[l].dim; ++q)
		{
			norms[q] = (C[l].block[q] * C[l].block[q].adjoint()).trace();
		}
		
		// interpret result
//		if (all_of(A_CHECK.begin(),A_CHECK.end(),[](bool x){return x;}) and 
//		    all_of(B_CHECK.begin(),B_CHECK.end(),[](bool x){return x;}))
//		{
//			sout += TCOLOR(MAGENTA);
//			sout += normal_token[3]; // X
//		}
		if (all_of(A_CHECK.begin(),A_CHECK.end(),[](bool x){return x;}))
		{
			sout += TCOLOR(RED);
			sout += normal_token[0]; // A
		}
		else
		{
			sout += TCOLOR(GREEN);
			sout += normal_token[2]; // M
		}
		
		if (all_of(B_CHECK.begin(),B_CHECK.end(),[](bool x){return x;}))
		{
			sout += TCOLOR(BLUE);
			sout += normal_token[1]; // B
		}
		else
		{
			sout += TCOLOR(GREEN);
			sout += normal_token[2]; // M
		}
	}
	
	sout += TCOLOR(BLACK);
	return sout;
}

template<typename Symmetry, typename Scalar>
complex<double> Umps<Symmetry,Scalar>::
dot (const Umps<Symmetry,Scalar> &Vket) const
{
	assert(N_sites == Vket.length());
	
	MatrixType LRdummy;
	size_t Mbra = A[GAUGE::R][0][0].block[0].rows();
	size_t Mket = Vket.A[GAUGE::R][0][0].block[0].rows();
	size_t D0 = qloc[0].size();
	
	TransferMatrix<Symmetry,double> TR;
	
	if (N_sites == 1)
	{
		TR = TransferMatrix<Symmetry,double>(GAUGE::R, A[GAUGE::R][0], Vket.A[GAUGE::R][0], LRdummy, {}, {D0});
	}
	else if (N_sites == 2)
	{
		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairR;
		contract_AA(A[GAUGE::R][0], qloc[0], A[GAUGE::R][1], qloc[1], ApairR);
		
		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > ApairKetR;
		contract_AA(Vket.A[GAUGE::R][0], qloc[0], Vket.A[GAUGE::R][1], qloc[1], ApairKetR);
		
		size_t D1 = qloc[1].size();
		boost::multi_array<double,4> WarrayDummy(boost::extents[D0][D0][D1][D1]);
		for (size_t s1=0; s1<D0; ++s1)
		for (size_t s2=0; s2<D0; ++s2)
		for (size_t s3=0; s3<D1; ++s3)
		for (size_t s4=0; s4<D1; ++s4)
		{
			WarrayDummy[s1][s2][s3][s4] = (s1==s2 and s3==s4)? 1.:0.;
		}
		
		TR.Warray.resize(boost::extents[D0][D0][D1][D1]); // This resize is necessary. I hate boost::multi_array. :-(
		TR = TransferMatrix<Symmetry,double>(GAUGE::R, ApairR, ApairKetR, LRdummy, WarrayDummy, {D0,D1});
	}
//	CMatrixType Reigen(Mket,Mbra);
	
//	ArnoldiSolver<TransferMatrix<Symmetry,double>,CMatrixType> Arnie;
	TransferVector<complex<double> > Reigen;
	Reigen.A.resize(Mket,Mbra);
	ArnoldiSolver<TransferMatrix<Symmetry,double>,TransferVector<complex<double> > > Arnie;
	Arnie.set_dimK(min(30ul,Mbra*Mket));
	complex<double> lambda;
	
	Arnie.calc_dominant(TR,Reigen,lambda);
	lout << Arnie.info() << endl;
	
	return lambda;
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_singularValues (size_t loc)
{
//	BDCSVD<MatrixType> Jack(C[loc].block[0]);
	JacobiSVD<MatrixType> Jack(C[loc].block[0]);
	Csingular[loc] = Jack.singularValues();
	size_t Nnz = (Jack.singularValues().array() > 0).count();
	S(loc) = -(Csingular[loc].head(Nnz).array().square() * Csingular[loc].head(Nnz).array().square().log()).sum();
}

template<typename Symmetry, typename Scalar>
VectorXd Umps<Symmetry,Scalar>::
singularValues (size_t loc)
{
	assert(loc<N_sites);
	return Csingular[loc];
}

template<typename Symmetry, typename Scalar>
double Umps<Symmetry,Scalar>::
entropy (size_t loc)
{
	assert(loc<N_sites);
	return S(loc);
}

MatrixXd gauge (const MatrixXd &U)
{
	MatrixXd Mout;
	Mout.setIdentity(U.rows(),U.cols());
	for (int i=0; i<U.rows(); ++i)
	{
		MatrixXd::Index imax;
		U.row(i).maxCoeff(&imax);
		Mout(i,i) *= (U.row(i)(imax)>=0)? 1:-1;
	}
	return Mout;
}

// creates AL, AR from AC, C
template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
polarDecompose (size_t loc)
{
	#ifdef DONT_USE_LAPACK_SVD
	BDCSVD<MatrixType> Jack;
//	JacobiSVD<MatrixType> Jack;
	#else
	LapackSVD<Scalar> Jack;
	#endif
	
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		qarray2<Symmetry::Nq> quple = {outset[loc][qout], outset[loc][qout]};
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].out[q] == outset[loc][qout])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Nrowsvec.push_back(A[GAUGE::C][loc][s].block[q].rows());
			}
		}
		
		// do the glue
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
		
		#ifdef DONT_USE_LAPACK_SVD
		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
		MatrixType UL = Jack.matrixU() * Jack.matrixV().adjoint();
		#else
		Jack.compute(Aclump);
		MatrixType UL = Jack.matrixU() * Jack.matrixVT();
		#endif
		
		auto it = C[loc].dict.find(quple);
		size_t qC = it->second;
		
		vector<MatrixType> UC;
		
		for (size_t q=0; q<C[loc].dim; ++q)
		{
			#ifdef DONT_USE_LAPACK_SVD
			Jack.compute(C[loc].block[q],ComputeThinU|ComputeThinV);
			UC.push_back(Jack.matrixU() * Jack.matrixV().adjoint());
			#else
			Jack.compute(C[loc].block[q]);
			UC.push_back(Jack.matrixU() * Jack.matrixVT());
			#endif
			
			// get the singular values and the entropy while at it (C[loc].dim=1 assumed):
			Csingular[loc] = Jack.singularValues();
			size_t Nnz = (Jack.singularValues().array() > 0).count();
			S(loc) = -(Csingular[loc].head(Nnz).array().square() * Csingular[loc].head(Nnz).array().square().log()).sum();
		}
		
		// update AL
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			A[GAUGE::L][loc][svec[i]].block[qvec[i]] = UL.block(stitch,0, Nrowsvec[i],Ncols) * UC[qC].adjoint();
			stitch += Nrowsvec[i];
		}
	}
	
	for (size_t qin=0; qin<inset[loc].size(); ++qin)
	{
		qarray2<Symmetry::Nq> quple = {inset[loc][qin], inset[loc][qin]};
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Ncolsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].in[q] == inset[loc][qin])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Ncolsvec.push_back(A[GAUGE::C][loc][s].block[q].cols());
			}
		}
		
		// do the glue
		size_t Nrows = A[GAUGE::C][loc][svec[0]].block[qvec[0]].rows();
		for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].rows() == Nrows);}
		size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
		
		MatrixType Aclump(Nrows,Ncols);
		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
			stitch += Ncolsvec[i];
		}
		
		#ifdef DONT_USE_LAPACK_SVD
		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
		MatrixType UR = Jack.matrixU() * Jack.matrixV().adjoint();
		#else
		Jack.compute(Aclump);
		MatrixType UR = Jack.matrixU() * Jack.matrixVT();
		#endif
		
		size_t locC = (N_sites==1)? 0 : (loc-1)%N_sites;
		auto it = C[locC].dict.find(quple);
		size_t qC = it->second;
		
		vector<MatrixType> UC;
		
		for (size_t q=0; q<C[locC].dim; ++q)
		{
			#ifdef DONT_USE_LAPACK_SVD
			Jack.compute(C[locC].block[q],ComputeThinU|ComputeThinV);
			UC.push_back(Jack.matrixU() * Jack.matrixV().adjoint());
			#else
			Jack.compute(C[locC].block[q]);
			UC.push_back(Jack.matrixU() * Jack.matrixVT());
			#endif
		}
		
		// update AR
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			A[GAUGE::R][loc][svec[i]].block[qvec[i]] = UC[qC].adjoint() * UR.block(0,stitch, Nrows,Ncolsvec[i]);
			stitch += Ncolsvec[i];
		}
	}
}

template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
calc_epsLRsq (size_t loc, double &epsLsq, double &epsRsq)
{
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		qarray2<Symmetry::Nq> quple = {outset[loc][qout], outset[loc][qout]};
		auto it = C[loc].dict.find(quple);
		size_t qC = it->second;
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].out[q] == outset[loc][qout])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Nrowsvec.push_back(A[GAUGE::C][loc][s].block[q].rows());
			}
		}
		
		// do the glue
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
		
		epsLsq = (Aclump-Acmp*C[loc].block[qC]).squaredNorm();
		
		BDCSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
//		JacobiSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
		MatrixType NullSpace = Jack.matrixV().rightCols((qloc[loc].size()-1) * A[GAUGE::C][loc][0].block[0].rows());
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
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			N[GAUGE::L][loc][svec[i]].block[0] = NullSpace.block(stitch,0, Nrowsvec[i],NullSpace.cols());
			stitch += Nrowsvec[i];
		}
		
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
	
	for (size_t qin=0; qin<inset[loc].size(); ++qin)
	{
		qarray2<Symmetry::Nq> quple = {inset[loc][qin], inset[loc][qin]};
		size_t locC = (N_sites==1)? 0 : (loc-1)%N_sites;
		auto it = C[locC].dict.find(quple);
		size_t qC = it->second;
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Ncolsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].in[q] == inset[loc][qin])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Ncolsvec.push_back(A[GAUGE::C][loc][s].block[q].cols());
			}
		}
		
		// do the glue
		size_t Nrows = A[GAUGE::C][loc][svec[0]].block[qvec[0]].rows();
		for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].rows() == Nrows);}
		size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
		
		MatrixType Aclump(Nrows,Ncols);
		MatrixType Acmp(Nrows,Ncols);
		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
			Acmp.block  (0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::R][loc][svec[i]].block[qvec[i]];
			stitch += Ncolsvec[i];
		}
		
		epsRsq = (Aclump-C[locC].block[qC]*Acmp).squaredNorm();
		
		BDCSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
//		JacobiSVD<MatrixType> Jack(Acmp.adjoint(),ComputeFullU|ComputeFullV);
		MatrixType NullSpace = Jack.matrixU().adjoint().bottomRows((qloc[loc].size()-1) * A[GAUGE::C][loc][0].block[0].rows());
		
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			N[GAUGE::R][loc][svec[i]].block[0] = NullSpace.block(0,stitch, NullSpace.rows(),Ncolsvec[i]);
			stitch += Ncolsvec[i];
		}
		
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

// creates AL, AR from AC, C
template<typename Symmetry, typename Scalar>
void Umps<Symmetry,Scalar>::
svdDecompose (size_t loc)
{
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		qarray2<Symmetry::Nq> quple = {outset[loc][qout], outset[loc][qout]};
		auto it = C[loc].dict.find(quple);
		size_t qC = it->second;
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].out[q] == outset[loc][qout])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Nrowsvec.push_back(A[GAUGE::C][loc][s].block[q].rows());
			}
		}
		
		// do the glue
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
		
		#ifdef DONT_USE_LAPACK_SVD
//		BDCSVD<MatrixType> Jack(Aclump,ComputeThinU|ComputeThinV);
		JacobiSVD<MatrixType> Jack(Aclump,ComputeThinU|ComputeThinV);
		#else
		LapackSVD<Scalar> Jack;
		Jack.compute(Aclump);
		#endif
		size_t Nret = Jack.singularValues().rows();
		
		// update AL
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			#ifdef DONT_USE_LAPACK_SVD
			A[GAUGE::L][loc][svec[i]].block[qvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret) * 
			                                           Jack.matrixV().adjoint().topRows(Nret);
			#else
			A[GAUGE::L][loc][svec[i]].block[qvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret) * 
			                                           Jack.matrixVT().topRows(Nret);
			#endif
			stitch += Nrowsvec[i];
		}
	}
	
	for (size_t qin=0; qin<inset[loc].size(); ++qin)
	{
		qarray2<Symmetry::Nq> quple = {inset[loc][qin], inset[loc][qin]};
		size_t locC = (N_sites==1)? 0 : (loc-1)%N_sites;
		auto it = C[locC].dict.find(quple);
		size_t qC = it->second;
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Ncolsvec;
		for (size_t s=0; s<qloc[loc].size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].in[q] == inset[loc][qin])
			{
				svec.push_back(s);
				qvec.push_back(q);
				Ncolsvec.push_back(A[GAUGE::C][loc][s].block[q].cols());
			}
		}
		
		// do the glue
		size_t Nrows = A[GAUGE::C][loc][svec[0]].block[qvec[0]].rows();
		for (size_t i=1; i<svec.size(); ++i) {assert(A[GAUGE::C][loc][svec[i]].block[qvec[i]].rows() == Nrows);}
		size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
		
		MatrixType Aclump(Nrows,Ncols);
		Aclump.setZero();
		size_t stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = A[GAUGE::C][loc][svec[i]].block[qvec[i]];
			stitch += Ncolsvec[i];
		}
		
		Aclump = C[locC].block[qC].adjoint() * Aclump;
		
		#ifdef DONT_USE_LAPACK_SVD
//		BDCSVD<MatrixType> Jack(Aclump,ComputeThinU|ComputeThinV);
		JacobiSVD<MatrixType> Jack(Aclump,ComputeThinU|ComputeThinV);
		#else
		LapackSVD<Scalar> Jack;
		Jack.compute(Aclump);
		#endif
		size_t Nret = Jack.singularValues().rows();
		
		// update AR
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			#ifdef DONT_USE_LAPACK_SVD
			A[GAUGE::R][loc][svec[i]].block[qvec[i]] = Jack.matrixU().leftCols(Nret) * 
			                                           Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i]);
			#else
			A[GAUGE::R][loc][svec[i]].block[qvec[i]] = Jack.matrixU().leftCols(Nret) * 
			                                           Jack.matrixVT().block(0,stitch, Nret,Ncolsvec[i]);
			#endif
			stitch += Ncolsvec[i];
		}
	}
	
	calc_singularValues(loc);
}

#endif
