#ifndef VANILLA_UMPS
#define VANILLA_UMPS

#include <set>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <type_traits>
#include <iostream>
#include <fstream>

struct GAUGE
{
	enum OPTION {L=0, R=1, C=2};
};

#include "VumpsTransferMatrix.h"
#include "LanczosSolver.h"
#include "ArnoldiSolver.h"

//template<typename MatrixType>
//void unique_QR (const MatrixType &M, MatrixType &Qmatrix, MatrixType &Rmatrix)
//{
//	#ifdef DONT_USE_EIGEN_QR
//	LapackQR<Scalar> Quirinus; // Lapack QR
//	#else
//	HouseholderQR<MatrixType> Quirinus; // Eigen QR
//	#endif
//	
//	Quirinus.compute(M);
//	
//	#ifdef DONT_USE_EIGEN_QR
//	Qmatrix = Quirinus.Qmatrix();
//	Rmatrix = Quirinus.Rmatrix();
//	#else
//	Qmatrix = Quirinus.householderQ() * MatrixType::Identity(M.rows(),M.cols());
//	Rmatrix = MatrixType::Identity(M.cols(),M.rows()) 
//	        * Quirinus.matrixQR().template triangularView<Upper>();
//	#endif
//	
//	// signs of the diagonal of Rmatrix in order to make the QR decomposition unique
//	VectorXd Signum = (Rmatrix.diagonal().array()/Rmatrix.diagonal().array().abs()).matrix();
//	Rmatrix = Signum.asDiagonal() * Rmatrix;
//	Qmatrix = Qmatrix * Signum.asDiagonal();
//}

//template<typename MatrixType>
//void unique_RQ (const MatrixType &M, MatrixType &Qmatrix, MatrixType &Rmatrix)
//{
//	#ifdef DONT_USE_EIGEN_QR
//	LapackQR<Scalar> Quirinus; // Lapack QR
//	#else
//	HouseholderQR<MatrixType> Quirinus; // Eigen QR
//	#endif
//	
//	Quirinus.compute(M.adjoint());
//	
//	#ifdef DONT_USE_EIGEN_QR
//	Qmatrix = Quirinus.Qmatrix().adjoint();
//	Rmatrix = Quirinus.Rmatrix().adjoint();
//	#else
//	Qmatrix = (Quirinus.householderQ() * MatrixType::Identity(M.cols(),M.rows())).adjoint();
//	Rmatrix = (MatrixType::Identity(M.rows(),M.cols()) 
//	        * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
//	#endif
//	
//	VectorXd Signum = (Rmatrix.diagonal().array()/Rmatrix.diagonal().array().abs()).matrix();
//	Rmatrix = Rmatrix * Signum.asDiagonal();
//	Qmatrix = Signum.asDiagonal() * Qmatrix;
//}

#include "Biped.h"
#include "Multipede.h"
#include "MpoQ.h"
#include "DmrgConglutinations.h"
#if !defined DONT_USE_LAPACK_SVD || !defined DONT_USE_LAPACK_QR
	#include "LapackWrappers.h"
#endif
#include "PolychromaticConsole.h"
#include "RandomVector.h"

/**Uniform Matrix Product State.
\describe_Nq
\describe_Scalar*/
template<size_t Nq, typename Scalar=double>
class UmpsQ
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
typedef Matrix<complex<double>,Dynamic,Dynamic> CMatrixType;
typedef Matrix<Scalar,Dynamic,1>       VectorType;
	
public:
	
	/**Does nothing.*/
	UmpsQ<Nq,Scalar>(){};
	
	template<typename Hamiltonian> UmpsQ (const Hamiltonian &H, size_t L_input, size_t Dmax, qarray<Nq> Qtot_input);
	
	UmpsQ (const vector<qarray<Nq> > &qloc_input, size_t L_input, size_t Dmax, qarray<Nq> Qtot_input);
	
	string info() const;
	string test_ortho (double tol=1e-10) const;
	
	void setRandom();
	
	void resize (size_t Dmax);
	
	/**Resizes all block matrices with the same forced dimensions. Useful for iDMRG.*/
	void forcedResize (size_t Dmax);
	
	void decompose (size_t loc, const vector<vector<Biped<Nq,MatrixType> > > &Apair);
	void svdDecompose (size_t loc);
	void polarDecompose (size_t loc);
	
	VectorXd singularValues (size_t loc=0);
	double entropy (size_t loc=0);
	
	inline vector<qarray<Nq> > locBasis (size_t loc) const {return qloc[loc];}
	inline vector<vector<qarray<Nq> > > locBasis()   const {return qloc;}
	
	/**Returns the amount of rows of first tensor without symmetries. Useful for iDMRG.*/
	size_t get_frst_rows() const {return A[GAUGE::C][0][0].block[0].rows();}
	
	/**Returns the amount of columns of last tensor without symmetries. Useful for iDMRG.*/
	size_t get_last_cols() const {return A[GAUGE::C][N_sites][0].block[0].cols();}
	
	size_t length() const {return N_sites;}
	
	void calc_epsLR (size_t loc, double &epsL, double &epsR);
	
	size_t calc_Dmax() const;
	size_t calc_Mmax() const;
	double memory (MEMUNIT memunit) const;
	
	complex<double> dot (const UmpsQ<Nq,Scalar> &Vket) const;
	
//private:
	
	size_t N_sites;
	size_t Dmax;
	double eps_svd = 1e-7;
	size_t N_sv;
	
	void calc_singularValues (size_t loc=0);
	
	// sets of all unique incoming & outgoing indices for convenience
	vector<vector<qarray<Nq> > > inset;
	vector<vector<qarray<Nq> > > outset;
	
	vector<vector<qarray<Nq> > > qloc;
	std::array<string,Nq> qlabel = {};
	qarray<Nq> Qtot;
	
	std::array<vector<vector<Biped<Nq,MatrixType> > >,3> A; // A[L/R/C][l][s].block[q]
	vector<Biped<Nq,MatrixType> >                        C; // zero-site part C[l]
	vector<vector<VectorType> >                          Sigma;
	
	bool SCHMIDT_SPECTRUM_CALCULATED = false;
	vector<VectorXd> Csingular;
	VectorXd S;
};

template<size_t Nq, typename Scalar>
string UmpsQ<Nq,Scalar>::
info() const
{
	stringstream ss;
	ss << "Umps: ";
	
//	if (Nq != 0)
//	{
//		ss << "(";
//		for (size_t q=0; q<Nq; ++q)
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
	ss << "S=" << S.transpose() << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
//	"overhead=" << round(overhead(MB),3) << "MB";
	
	return ss.str();
}

template<size_t Nq, typename Scalar>
template<typename Hamiltonian>
UmpsQ<Nq,Scalar>::
UmpsQ (const Hamiltonian &H, size_t L_input, size_t Dmax, qarray<Nq> Qtot_input)
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

template<size_t Nq, typename Scalar>
UmpsQ<Nq,Scalar>::
UmpsQ (const vector<qarray<Nq> > &qloc_input, size_t L_input, size_t Dmax, qarray<Nq> Qtot_input)
{
	N_sites = L_input;
	qloc.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l) {qloc[l] = qloc_input;}
	resize(Dmax);
}

template<size_t Nq, typename Scalar>
size_t UmpsQ<Nq,Scalar>::
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

template<size_t Nq, typename Scalar>
double UmpsQ<Nq,Scalar>::
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

template<size_t Nq, typename Scalar>
size_t UmpsQ<Nq,Scalar>::
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

template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
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
	}
	C.resize(N_sites);
	Sigma.resize(N_sites);
	inset.resize(N_sites);
	outset.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		inset[l].push_back(qvacuum<Nq>());
		outset[l].push_back(qvacuum<Nq>());
		Sigma[l].resize(outset[l].size());
	}
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[g][l][s].in.push_back(qvacuum<Nq>());
		A[g][l][s].out.push_back(qvacuum<Nq>());
		A[g][l][s].dict.insert({qarray2<Nq>{qvacuum<Nq>(),qvacuum<Nq>()}, A[g][l][s].dim});
		A[g][l][s].dim = 1;
		A[g][l][s].block.resize(1);
	}
	
	for (size_t g=0; g<3; ++g)
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc[l].size(); ++s)
	{
		A[g][l][s].block[0].resize(Dmax,Dmax);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		C[l].in.push_back(qvacuum<Nq>());
		C[l].out.push_back(qvacuum<Nq>());
		C[l].dict.insert({qarray2<Nq>{qvacuum<Nq>(),qvacuum<Nq>()}, C[l].dim});
		C[l].dim = 1;
		C[l].block.resize(1);
		C[l].block[0].resize(Dmax,Dmax);
	}
	
	Csingular.clear();
	Csingular.resize(N_sites);
	S.resize(N_sites);
}

template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
forcedResize (size_t Dmax)
{
	for (size_t l=0; l<N_sites; ++l)
	for (size_t s=0; s<qloc.size(); ++s)
	for (size_t q=0; q<A[GAUGE::C][l][s].dim; ++q)
	{
		A[GAUGE::C][l][s].block[q].resize(Dmax,Dmax);
	}
}

template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
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

template<size_t Nq, typename Scalar>
string UmpsQ<Nq,Scalar>::
test_ortho (double tol) const
{
	string sout = "";
	std::array<string,4> normal_token  = {"A","B","M","X"};
	std::array<string,4> special_token = {"\e[4mA\e[0m","\e[4mB\e[0m","\e[4mM\e[0m","\e[4mX\e[0m"};
	
	for (int l=0; l<this->N_sites; ++l)
	{
		// check for A
		Biped<Nq,MatrixType> Test = A[GAUGE::L][l][0].adjoint() * A[GAUGE::L][l][0];
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
			cout << "q=" << q << ", norm=" << norms[q] << endl;
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

template<size_t Nq, typename Scalar>
complex<double> UmpsQ<Nq,Scalar>::
dot (const UmpsQ<Nq,Scalar> &Vket) const
{
	MatrixType LRdummy;
	size_t Mbra = A[GAUGE::R][0][0].block[0].rows();
	size_t Mket = Vket.A[GAUGE::R][0][0].block[0].rows();
	
	TransferMatrix<Nq,double> TR(GAUGE::R, A[GAUGE::R][0], Vket.A[GAUGE::R][0], LRdummy, {});
	CMatrixType Reigen(Mket,Mbra);
	
	ArnoldiSolver<TransferMatrix<Nq,double>,CMatrixType> Arnie;
	Arnie.set_dimK(min(100ul,Mbra*Mket));
	complex<double> lambda;
	
	Arnie.calc_dominant(TR,Reigen,lambda);
	
	return lambda;
}

template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
calc_singularValues (size_t loc)
{
	BDCSVD<MatrixType> Jack(C[loc].block[0]);
	Csingular[loc] = Jack.singularValues();
	size_t Nnz = (Jack.singularValues().array() > 0.).count();
	S(loc) = -(Csingular[loc].head(Nnz).array().square() * Csingular[loc].head(Nnz).array().square().log()).sum();
}

template<size_t Nq, typename Scalar>
VectorXd UmpsQ<Nq,Scalar>::
singularValues (size_t loc)
{
	assert(loc<N_sites);
	return Csingular[loc];
}

template<size_t Nq, typename Scalar>
double UmpsQ<Nq,Scalar>::
entropy (size_t loc)
{
	assert(loc<N_sites);
	return S(loc);
}

// creates AL, AR from AC, C
template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
polarDecompose (size_t loc)
{
	BDCSVD<MatrixType> Jack;
	
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		qarray2<Nq> quple = {outset[loc][qout], outset[loc][qout]};
		
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
		
		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
		MatrixType UL = Jack.matrixU() * Jack.matrixV().adjoint();
//		MatrixType UL, Rmatrix;
//		unique_QR(Aclump, UL, Rmatrix);
		
		auto it = C[loc].dict.find(quple);
		size_t qC = it->second;
		
		vector<MatrixType> UC;
		
		for (size_t q=0; q<C[loc].dim; ++q)
		{
			Jack.compute(C[loc].block[q],ComputeThinU|ComputeThinV);
			UC.push_back(Jack.matrixU()*Jack.matrixV().adjoint());
			
			// get the singular values and the entropy while at it (C[loc].dim=1 assumed):
			Csingular[loc] = Jack.singularValues();
			size_t Nnz = (Jack.singularValues().array() > 0.).count();
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
		qarray2<Nq> quple = {inset[loc][qin], inset[loc][qin]};
		
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
		
		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
		MatrixType UR = Jack.matrixU() * Jack.matrixV().adjoint();
//		MatrixType UR, Rmatrix;
//		unique_RQ(Aclump, UR, Rmatrix);
		
		size_t locC = (N_sites==1)? 0 : (loc-1)%2;
		auto it = C[locC].dict.find(quple);
		size_t qC = it->second;
		
		vector<MatrixType> UC;
		
		for (size_t q=0; q<C[locC].dim; ++q)
		{
			Jack.compute(C[locC].block[q],ComputeThinU|ComputeThinV);
			UC.push_back(Jack.matrixU()*Jack.matrixV().adjoint());
			
			// get the singular values and the entropy while at it (C[loc].dim=1 assumed):
			Csingular[loc] = Jack.singularValues();
			size_t Nnz = (Jack.singularValues().array() > 0.).count();
			S(loc) = -(Csingular[loc].head(Nnz).array().square() * Csingular[loc].head(Nnz).array().square().log()).sum();
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

template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
calc_epsLR (size_t loc, double &epsL, double &epsR)
{
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		qarray2<Nq> quple = {outset[loc][qout], outset[loc][qout]};
		auto it = C[loc].dict.find(quple);
		size_t qC = it->second;
		
		// determine how many A's to glue together
		vector<size_t> svec, qvec, Nrowsvec;
		for (size_t s=0; s<qloc.size(); ++s)
		for (size_t q=0; q<A[GAUGE::C][loc][s].dim; ++q)
		{
			if (A[GAUGE::C][loc][s].out[q] == outset[0][qout])
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
			Acmp.block(stitch,0, Nrowsvec[i],Ncols)   = A[GAUGE::L][loc][svec[i]].block[qvec[i]];
			stitch += Nrowsvec[i];
		}
		
		epsL = (Aclump-Acmp*C[loc].block[qC]).norm();
	}
	
	for (size_t qin=0; qin<inset[loc].size(); ++qin)
	{
		qarray2<Nq> quple = {inset[loc][qin], inset[loc][qin]};
		size_t locC = (N_sites==1)? 0 : (loc-1)%2;
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
		
		epsR = (Aclump-C[locC].block[qC]*Acmp).norm();
	}
}

// creates AL, AR from AC, C
template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
svdDecompose (size_t loc)
{
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		qarray2<Nq> quple = {outset[loc][qout], outset[loc][qout]};
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
		
		BDCSVD<MatrixType> Jack(Aclump,ComputeThinU|ComputeThinV);
		size_t Nret = Jack.singularValues().rows();
		
		// update AL
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			A[GAUGE::L][loc][svec[i]].block[qvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret) * 
			                                           Jack.matrixV().adjoint().topRows(Nret);
			stitch += Nrowsvec[i];
		}
	}
	
	for (size_t qin=0; qin<inset[loc].size(); ++qin)
	{
		qarray2<Nq> quple = {inset[loc][qin], inset[loc][qin]};
		size_t locC = (N_sites==1)? 0 : (loc-1)%2;
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
		
		BDCSVD<MatrixType> Jack(Aclump,ComputeThinU|ComputeThinV);
		size_t Nret = Jack.singularValues().rows();
		
		// update AR
		stitch = 0;
		for (size_t i=0; i<svec.size(); ++i)
		{
			A[GAUGE::R][loc][svec[i]].block[qvec[i]] = Jack.matrixU().leftCols(Nret) * 
			                                           Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i]);
			stitch += Ncolsvec[i];
		}
	}
	
	calc_singularValues(loc);
}

template<size_t Nq, typename Scalar>
void UmpsQ<Nq,Scalar>::
decompose (size_t loc, const vector<vector<Biped<Nq,MatrixType> > > &Apair)
{
	ArrayXd truncWeightSub(outset[loc].size()); truncWeightSub.setZero();
	ArrayXd entropySub(outset[loc].size()); entropySub.setZero();
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
	for (size_t qout=0; qout<outset[loc].size(); ++qout)
	{
		vector<size_t> s1vec, s3vec;
		map<size_t,vector<size_t> > s13map;
		map<pair<size_t,size_t>,size_t> s13qmap;
		for (size_t s1=0; s1<qloc[loc].size(); ++s1)
		for (size_t s3=0; s3<qloc[loc+1].size(); ++s3)
		for (size_t q13=0; q13<Apair[s1][s3].dim; ++q13)
		{
			if (Apair[s1][s3].in[q13] + qloc[loc][s1] == outset[loc][qout])
			{
				s1vec.push_back(s1);
				s3vec.push_back(s3);
				s13map[s1].push_back(s3);
				s13qmap[make_pair(s1,s3)] = q13;
			}
		}
		
		if (s1vec.size() != 0)
		{
			vector<MatrixType> Aclumpvec(qloc[loc].size());
			size_t istitch = 0;
			size_t jstitch = 0;
			vector<size_t> get_s3;
			vector<size_t> get_Ncols;
			bool COLS_ARE_KNOWN = false;
			
			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
			{
				for (size_t s3=0; s3<qloc[loc+1].size(); ++s3)
				{
					auto s3block = find(s13map[s1].begin(), s13map[s1].end(), s3);
					if (s3block != s13map[s1].end())
					{
						size_t q13 = s13qmap[make_pair(s1,s3)];
						addRight(Apair[s1][s3].block[q13], Aclumpvec[s1]);
						
						if (COLS_ARE_KNOWN == false)
						{
							get_s3.push_back(s3);
							get_Ncols.push_back(Apair[s1][s3].block[q13].cols());
						}
					}
				}
				if (get_s3.size() != 0) {COLS_ARE_KNOWN = true;}
			}
			
			vector<size_t> get_s1;
			vector<size_t> get_Nrows;
			MatrixType Aclump;
			for (size_t s1=0; s1<qloc[loc].size(); ++s1)
			{
				size_t Aclump_rows_old = Aclump.rows();
				addBottom(Aclumpvec[s1], Aclump);
				if (Aclump.rows() > Aclump_rows_old)
				{
					get_s1.push_back(s1);
					get_Nrows.push_back(Aclump.rows()-Aclump_rows_old);
				}
			}
			
			#ifdef DONT_USE_LAPACK_SVD
			BDCSVD<MatrixType> Jack; // Eigen SVD
			#else
			LapackSVD<Scalar> Jack; // Lapack SVD
			#endif
			
			#ifdef DONT_USE_LAPACK_SVD
			Jack.compute(Aclump,ComputeThinU|ComputeThinV);
			#else
			Jack.compute(Aclump);
			#endif
			
			// retained states:
			size_t Nret = Aclump.cols();
			Nret = (Jack.singularValues().array().abs() > this->eps_svd).count();
			Nret = min(max(Nret,1ul),static_cast<size_t>(Jack.singularValues().rows()));
			Nret = min(Nret,this->N_sv);
			
			truncWeightSub(qout) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
			size_t Nnz = (Jack.singularValues().array() > 1e-9).count();
			entropySub(qout) = -(Jack.singularValues().head(Nnz).array().square() * Jack.singularValues().head(Nnz).array().square().log()).sum();
			
			MatrixType Aleft, Aright, ACright, ACleft;
			Aleft = Jack.matrixU().leftCols(Nret);
			ACleft = Jack.matrixU().leftCols(Nret) * Jack.singularValues().head(Nret).asDiagonal();
			#ifdef DONT_USE_LAPACK_SVD
			Aright = Jack.matrixV().adjoint().topRows(Nret);
			ACright = Jack.singularValues().head(Nret).asDiagonal() * Jack.matrixV().adjoint().topRows(Nret);
			#else
			Aright = Jack.matrixVT().topRows(Nret);
			ACright = Jack.singularValues().head(Nret).asDiagonal() * Jack.matrixVT().topRows(Nret);
			#endif
			Sigma[loc][qout] = Jack.singularValues();
			
			// update AL[loc]
			istitch = 0;
			for (size_t i=0; i<get_s1.size(); ++i)
			{
				size_t s1 = get_s1[i];
				size_t Nrows = get_Nrows[i];
				qarray2<Nq> quple = {outset[loc][qout]-qloc[loc][s1], outset[loc][qout]};
				auto q = A[GAUGE::L][loc][s1].dict.find(quple);
				if (q != A[GAUGE::L][loc][s1].dict.end())
				{
					A[GAUGE::L][loc][s1].block[q->second] = Aleft.block(istitch,0, Nrows,Nret);
					A[GAUGE::C][loc][s1].block[q->second] = ACleft.block(istitch,0, Nrows,Nret);
				}
				istitch += Nrows;
			}
			
			// update AR[loc+1]
			jstitch = 0;
			for (size_t i=0; i<get_s3.size(); ++i)
			{
				size_t s3 = get_s3[i];
				size_t Ncols = get_Ncols[i];
				qarray2<Nq> quple = {outset[loc][qout], outset[loc][qout]+qloc[loc][s3]};
				auto q = A[GAUGE::R][loc+1][s3].dict.find(quple);
				if (q != A[GAUGE::R][loc+1][s3].dict.end())
				{
					A[GAUGE::R][loc+1][s3].block[q->second] = Aright.block(0,jstitch, Nret,Ncols);
					A[GAUGE::C][loc+1][s3].block[q->second] = ACright.block(0,jstitch, Nret,Ncols);
				}
				jstitch += Ncols;
			}
		}
	}
	
//	truncWeight(loc) = truncWeightSub.sum();
}

#endif
