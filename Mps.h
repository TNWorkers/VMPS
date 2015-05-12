#ifndef VANILLA_MPS
#define VANILLA_MPS

#include <iomanip>

#include "DmrgJanitor.h"
#include "DmrgConglutinations.h"
#include "Mpo.h"
#include "DmrgExternalQ.h"
#ifndef DONT_USE_LAPACK_SVD
	#include "LapackWrappers.h"
#endif

template<size_t D, typename Scalar>
class Mps : public DmrgJanitor<PivotMatrix<D> >
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

template<size_t D_, typename MpHamiltonian> friend class DmrgSolver;
friend class MpsCompressor<D,Scalar>;
template<size_t D_, size_t Daux_, typename S_> friend void HxV (const Mpo<D_,Daux_> &H, const Mps<D_,S_> &Vin, Mps<D_,S_> &Vout);

public:
	
	Mps(){};
	Mps (size_t L_input, size_t Dmax);
	
	string info() const;
	string test_ortho() const;
	string Asizes() const;
	size_t calc_Dmax() const;
	size_t calc_Mmax() const;
	
	template<typename OtherScalar> void addScale (const Mps<D,Scalar> &Vin, OtherScalar factor);
	Mps& operator+= (const Mps<D,Scalar> &Vin);
	Mps& operator-= (const Mps<D,Scalar> &Vin);
	template<typename OtherScalar> Mps& operator*= (const OtherScalar &factor);
	template<typename OtherScalar> Mps& operator/= (const OtherScalar &factor);
	template<typename OtherScalar> Mps<D,OtherScalar> cast() const;
	void collapse();
	
	Scalar dot (const Mps<D,Scalar> &V) const;
	void swap (Mps<D,Scalar> &V);
	
	void outerResize (size_t L_input);
	void innerResize (size_t Dmax, bool CONSERVATIVE=false);
	void setZero();
	void setRandom();
	
	void leftSweepStep  (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrix<D> *H = NULL);
	void rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrix<D> *H = NULL);
	void calc_noise (PivotMatrix<D> *H, DMRG::DIRECTION::OPTION DIR, 
	                 const std::array<std::array<MatrixType,D>,D> rho, 
	                 std::array<std::array<MatrixType,D>,D>       &rhoNoise);
	
	const std::array<MatrixType,D> &A_at (size_t loc) const {return A[loc];};
	
	void enrich_left  (size_t loc, PivotMatrix<D> *H, Matrix<Scalar,Dynamic,Dynamic> &Aclump);
	void enrich_right (size_t loc, PivotMatrix<D> *H, Matrix<Scalar,Dynamic,Dynamic> &Aclump);
	
private:
	
	template<typename OtherScalar> void add_site (size_t loc, OtherScalar factor, const Mps<D,Scalar> &Vin);
	
	vector<std::array<MatrixType,D> > A;
	
	ArrayXd truncWeight;
};

template<size_t D, typename Scalar>
Mps<D,Scalar>::
Mps (size_t L_input, size_t Dmax)
:DmrgJanitor<PivotMatrix<D> >()
{
	outerResize(L_input);
	innerResize(Dmax);
	this->pivot = -1;
}

template<size_t D, typename Scalar>
string Mps<D,Scalar>::
info() const
{
	stringstream ss;
	ss << "Mps: ";
	ss << "L=" << this->N_sites << ", ";
	ss << "D=" << D << ", ";
	ss << "Dmax=" << calc_Dmax() << ", ";
	ss << "pivot=" << this->pivot << ", ";
	ss << "trunc_weight=" << truncWeight.sum();
	
	return ss.str();
}

template<size_t D, typename Scalar>
string Mps<D,Scalar>::
Asizes() const
{
	stringstream ss;
	ss << "Asizes: ";
	for (size_t l=0; l<this->N_sites; ++l)
	{
		ss << "(" << A[l][0].rows() << "," << A[l][0].cols() << ") ";
	}
	return ss.str();
}

template<size_t D, typename Scalar>
size_t Mps<D,Scalar>::
calc_Dmax() const
{
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<D; ++s)
	{
		if (A[l][s].rows() > res) {res = A[l][s].rows();}
		if (A[l][s].cols() > res) {res = A[l][s].cols();}
	}
	return res;
}

template<size_t D, typename Scalar>
size_t Mps<D,Scalar>::
calc_Mmax() const
{
	size_t res = 0;
	for (size_t l=0; l<this->N_sites; ++l)
	{
		size_t M = 0;
		for (size_t s=0; s<D; ++s)
		{
			M += A[l][s].rows() * A[l][s].cols();
		}
		if (M>res) {res = M;}
	}
	return res;
}

// Not good to go below D with Dmax, will use max(Dmax,D)
template<size_t D, typename Scalar>
void Mps<D,Scalar>::
outerResize (size_t L_input)
{
	this->N_sites = L_input;
	
	truncWeight.resize(this->N_sites);
	truncWeight.setZero();
	
	A.resize(this->N_sites);
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
innerResize (size_t Dmax, bool CONSERVATIVE)
{
	for (size_t s=0; s<D; ++s)
	{
		A[0][s].resize(1,min(D,Dmax));
		A[this->N_sites-1][s].resize(min(D,Dmax),1);
	}
	
	for (size_t l=1; l<this->N_sites/2; ++l)
	{
//		size_t Nrows = min(max(Dmax,D), A[l-1][0].rows()*D);
//		size_t Ncols = min(max(Dmax,D), A[l-1][0].cols()*D);
		size_t Nrows = min(Dmax, A[l-1][0].rows()*D);
		size_t Ncols = min(Dmax, A[l-1][0].cols()*D);
		for (size_t s=0; s<D; ++s)
		{
			if (CONSERVATIVE == false)
			{
				A[l][s].resize(Nrows,Ncols);
				A[this->N_sites-l-1][s].resize(Ncols,Nrows);
			}
			else
			{
				size_t Noldrows = A[l][s].rows();
				size_t Noldcols = A[l][s].cols();
				A[l][s].conservativeResize(Nrows,Ncols);
				A[l][s].bottomRows(Nrows-Noldrows).setZero();
				A[l][s].rightCols (Ncols-Noldcols).setZero();
				
				Noldrows = A[this->N_sites-l-1][s].rows();
				Noldcols = A[this->N_sites-l-1][s].cols();
				A[this->N_sites-l-1][s].conservativeResize(Ncols,Nrows);
				A[this->N_sites-l-1][s].bottomRows(Ncols-Noldrows).setZero();
				A[this->N_sites-l-1][s].rightCols (Nrows-Noldcols).setZero();
			}
		}
	}
	
	// middle matrix for odd chain length:
	if (this->N_sites%2==1)
	{
		size_t centre = this->N_sites/2;
		int Nrows = A[centre-1][0].cols();
		int Ncols = A[centre+1][0].rows();
		for (size_t s=0; s<D; ++s)
		{
			if (CONSERVATIVE == false)
			{
				A[centre][s].resize(Nrows,Ncols);
			}
			else
			{
				size_t Noldrows = A[centre][s].rows();
				size_t Noldcols = A[centre][s].cols();
				A[centre][s].conservativeResize(Nrows,Ncols);
				A[centre][s].bottomRows(Nrows-Noldrows).setZero();
				A[centre][s].rightCols (Ncols-Noldcols).setZero();
			}
		}
	}
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
setZero()
{
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<D; ++s)
	{
		A[l][s].setZero();
	}
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
setRandom()
{
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<D; ++s)
	{
		A[l][s].setRandom();
	}
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
leftSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrix<D> *H)
{
	size_t Nrows = A[loc][0].rows();
	size_t Ncols = A[loc][0].cols();
	
	MatrixType deltaRho;
	std::array<std::array<MatrixType,D>,D> rhoArray;
	if (TOOL == DMRG::BROOM::RDM)
	{
		std::array<std::array<MatrixType,D>,D> rhoNoiseArray;
		
		#pragma omp parallel for collapse(2)
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			rhoArray[s1][s2] =  A[loc][s1].adjoint() * A[loc][s2];
		}
		
		calc_noise(H, DMRG::DIRECTION::LEFT, rhoArray, rhoNoiseArray);
		
		deltaRho.resize(D*Ncols,D*Ncols);
		deltaRho.setZero();
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			deltaRho.block(s1*Ncols,s2*Ncols, Ncols,Ncols) = rhoNoiseArray[s1][s2];
		}
	}
	
	MatrixType Aclump(Nrows,D*Ncols);
	for (size_t s=0; s<D; ++s)
	{
		Aclump.block(0,s*Ncols, Nrows,Ncols) = A[loc][s];
	}
	
	#ifdef DONT_USE_LAPACK_SVD
	JacobiSVD<MatrixType> Jack; // SVD
	#else
	LapackSVD<Scalar> Jack; // SVD
	#endif
	HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix;
	MatrixType rho; SelfAdjointEigenSolver<MatrixType> Eugen;
	size_t Nret = Nrows;
	
	auto perform_svd = [this,&Jack,&Aclump,&Nret,&TOOL,&loc]()
	{
		#ifdef DONT_USE_LAPACK_SVD
		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
		#else
		Jack.compute(Aclump);
		#endif
		if (TOOL == DMRG::BROOM::BRUTAL_SVD)
		{
			Nret = min(static_cast<size_t>(Jack.singularValues().rows()), this->N_sv);
		}
		else
		{
			Nret = (Jack.singularValues().array() > this->eps_svd).count();
		}
		Nret = max(Nret,1ul);
		truncWeight(loc) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
	};
	
	if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD)
	{
		perform_svd();
	}
//	if (TOOL == DMRG::BROOM::SVD)
//	{
//		#ifdef DONT_USE_LAPACK_SVD
//		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
//		#else
//		Jack.compute(Aclump);
//		#endif
//		Nret = (Jack.singularValues().array() > this->eps_svd).count();
//		Nret = max(Nret,1ul);
//		truncWeight(loc) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
//	}
//	else if (TOOL == DMRG::BROOM::BRUTAL_SVD)
//	{
//		#ifdef DONT_USE_LAPACK_SVD
//		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
//		#else
//		Jack.compute(Aclump);
//		#endif
//		Nret = min(static_cast<size_t>(Jack.singularValues().rows()), this->N_sv);
//		Nret = max(Nret,1ul);
//		truncWeight(loc) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
//	}
	else if (TOOL == DMRG::BROOM::QR)
	{
		Quirinus.compute(Aclump.adjoint());
		Qmatrix = MatrixType::Identity(Aclump.cols(), Aclump.rows());
		Qmatrix = (Quirinus.householderQ() * Qmatrix).adjoint();
		Rmatrix = MatrixType::Identity(Aclump.rows(), Aclump.cols());
		Rmatrix = (Rmatrix * Quirinus.matrixQR().template triangularView<Upper>()).adjoint();
	}
	else if (TOOL == DMRG::BROOM::RDM)
	{
		rho.resize(D*Ncols,D*Ncols);
		rho.setZero();
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			rho.block(s1*Ncols,s2*Ncols, Ncols,Ncols) = rhoArray[s1][s2];
		}
		rho += this->eps_noise * deltaRho;
		Eugen.compute(rho);
		
		Nret = (Eugen.eigenvalues().array() > this->eps_rdm).count();
		Nret = max(Nret,1ul);
		truncWeight(loc) = Eugen.eigenvalues().head(rho.rows()-Nret).sum();
	}
	else if (TOOL == DMRG::BROOM::RICH_SVD)
	{
		enrich_left(loc,H,Aclump);
		perform_svd();
	}
	
	for (size_t s=0; s<D; ++s)
	{
		if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
		{
			#ifdef DONT_USE_LAPACK_SVD
			A[loc][s] = Jack.matrixV().adjoint().block(0,s*Ncols, Nret,Ncols);
			#else
			A[loc][s] = Jack.matrixVT().block(0,s*Ncols, Nret,Ncols);
			#endif
		}
		else if (TOOL == DMRG::BROOM::QR)
		{
			A[loc][s] = Qmatrix.block(0,s*Ncols, Nrows,Ncols);
		}
		else if (TOOL == DMRG::BROOM::RDM)
		{
			A[loc][s] = Eugen.eigenvectors().rowwise().reverse().transpose().topRows(Nret).block(0,s*Ncols, Nret,Ncols);
		}
	}
	
	if (loc != 0)
	{
		for (int s=0; s<D; s++)
		{
			if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
			{
				A[loc-1][s] = A[loc-1][s] * 
				              Jack.matrixU().leftCols(Nret) * 
				              Jack.singularValues().head(Nret).asDiagonal();
			}
			else if (TOOL == DMRG::BROOM::QR)
			{
				A[loc-1][s] = A[loc-1][s] * Rmatrix;
			}
			else if (TOOL == DMRG::BROOM::RDM)
			{
				A[loc-1][s] = A[loc-1][s] * (Aclump * Eugen.eigenvectors().rowwise().reverse()).leftCols(Nret);
			}
		}
	}
	
	this->pivot = (loc==0)? 0 : loc-1;
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL, PivotMatrix<D> *H)
{
	size_t Nrows = A[loc][0].rows();
	size_t Ncols = A[loc][0].cols();
	
	MatrixType deltaRho;
	std::array<std::array<MatrixType,D>,D> rhoArray;
	if (TOOL == DMRG::BROOM::RDM)
	{
		std::array<std::array<MatrixType,D>,D> rhoNoiseArray;
		
		#pragma omp parallel for collapse(2)
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			rhoArray[s1][s2] = A[loc][s1] * A[loc][s2].adjoint();
		}
		
		calc_noise(H, DMRG::DIRECTION::RIGHT, rhoArray, rhoNoiseArray);
		
		deltaRho.resize(D*Nrows,D*Nrows);
		deltaRho.setZero();
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			deltaRho.block(s1*Nrows,s2*Nrows, Nrows,Nrows) = rhoNoiseArray[s1][s2];
		}
	}
	
	MatrixType Aclump(D*Nrows,Ncols);
	for (size_t s=0; s<D; ++s)
	{
		Aclump.block(s*Nrows,0, Nrows,Ncols) = A[loc][s];
	}
	
	#ifdef DONT_USE_LAPACK_SVD
	JacobiSVD<MatrixType> Jack; // SVD
	#else
	LapackSVD<Scalar> Jack; // SVD
	#endif
	HouseholderQR<MatrixType> Quirinus; MatrixType Qmatrix, Rmatrix;
	MatrixType rho; SelfAdjointEigenSolver<MatrixType> Eugen;
	size_t Nret = Nrows;
	
	auto perform_svd = [this,&Jack,&Aclump,&Nret,&TOOL,&loc]()
	{
		#ifdef DONT_USE_LAPACK_SVD
		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
		#else
		Jack.compute(Aclump);
		#endif
		if (TOOL == DMRG::BROOM::BRUTAL_SVD)
		{
			Nret = min(static_cast<size_t>(Jack.singularValues().rows()), this->N_sv);
		}
		else
		{
			Nret = (Jack.singularValues().array() > this->eps_svd).count();
		}
		Nret = max(Nret,1ul);
		truncWeight(loc) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
	};
	
	if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD)
	{
		perform_svd();
	}
	else if (TOOL == DMRG::BROOM::QR)
	{
		Quirinus.compute(Aclump);
		Qmatrix = MatrixType::Identity(Aclump.rows(), Aclump.cols());
		Qmatrix = Quirinus.householderQ() * Qmatrix;
		Rmatrix = MatrixType::Identity(Aclump.cols(), Aclump.rows());
		Rmatrix = Rmatrix * Quirinus.matrixQR().template triangularView<Upper>();
	}
	else if (TOOL == DMRG::BROOM::RDM)
	{
		rho.resize(D*Nrows,D*Nrows);
		rho.setZero();
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			rho.block(s1*Nrows,s2*Nrows, Nrows,Nrows) = rhoArray[s1][s2];
		}
		rho += this->eps_noise * deltaRho;
		Eugen.compute(rho);
		
		Nret = (Eugen.eigenvalues().array() > this->eps_rdm).count();
		Nret = max(Nret,1ul);
		truncWeight(loc) = Eugen.eigenvalues().head(rho.rows()-Nret).sum();
	}
	else if (TOOL == DMRG::BROOM::RICH_SVD)
	{
		enrich_right(loc,H,Aclump);
		perform_svd();
	}
	
	for (size_t s=0; s<D; ++s)
	{
		if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
		{
			A[loc][s] = Jack.matrixU().block(s*Nrows,0, Nrows,Nret);
		}
		else if (TOOL == DMRG::BROOM::QR)
		{
			A[loc][s] = Qmatrix.block(s*Nrows,0, Nrows,Ncols);
		}
		else if (TOOL == DMRG::BROOM::RDM)
		{
			A[loc][s] = (Eugen.eigenvectors().rowwise().reverse().leftCols(Nret)).block(s*Nrows,0, Nrows,Nret);
		}
	}
	
	if (loc != this->N_sites-1)
	{
		for (int s=0; s<D; s++)
		{
			if (TOOL == DMRG::BROOM::SVD or TOOL == DMRG::BROOM::BRUTAL_SVD or TOOL == DMRG::BROOM::RICH_SVD)
			{
				#ifdef DONT_USE_LAPACK_SVD
				A[loc+1][s] = Jack.singularValues().head(Nret).asDiagonal() * 
				              Jack.matrixV().adjoint().topRows(Nret) * 
				              A[loc+1][s];
				#else
				A[loc+1][s] = Jack.singularValues().head(Nret).asDiagonal() * 
				              Jack.matrixVT().topRows(Nret) * 
				              A[loc+1][s];
				#endif
			}
			else if (TOOL == DMRG::BROOM::QR)
			{
				A[loc+1][s] = Rmatrix * A[loc+1][s];
			}
			else if (TOOL == DMRG::BROOM::RDM)
			{
				A[loc+1][s] = (Eugen.eigenvectors().rowwise().reverse().adjoint() * Aclump).topRows(Nret) * A[loc+1][s];
			}
		}
	}
	
	this->pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
enrich_left (size_t loc, PivotMatrix<D> *H, Matrix<Scalar,Dynamic,Dynamic> &Aclump)
{
	size_t Nrows = A[loc][0].rows();
	size_t Ncols = A[loc][0].cols();
	
	std::array<MatrixType,D> P;
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
//	for (size_t a=0; a<H->R.size(); ++a)
	for (int k=0; k<H->W[s1][s2].outerSize(); ++k)
	for (SparseMatrixXd::InnerIterator iW(H->W[s1][s2],k); iW; ++iW)
	{
		size_t a = iW.col();
		MatrixType Mtmp;
		if (H->R[a].rows() != 0)
		{
			Mtmp = this->eps_rsvd * iW.value() * A[loc][s2] * H->R[a];
			if (P[s1].rows() == 0) {P[s1] = Mtmp;}
			else                   {P[s1] += Mtmp;}
		}
	}
	
//	size_t Daux = max(H->W[0][0].rows(), H->W[0][0].cols());
//	vector<std::array<MatrixType,D> > Paux(Daux);
//	
//	for (size_t s=0; s<D; ++s)
//	for (size_t w=0; w<Daux; ++w)
//	{
//		if (Paux[w][s].rows() == 0)
//		{
//			Paux[w][s].resize(A[loc][0].rows(), A[loc][0].cols());
//			Paux[w][s].setZero();
//		}
//	}
	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t a=0; a<H->R.size(); ++a)
//	for (int k=0; k<H->W[s1][s2].outerSize(); ++k)
//	for (SparseMatrixXd::InnerIterator iW(H->W[s1][s2],k); iW; ++iW)
//	{
//		MatrixType Mtmp;
//		if (H->R[a].rows() != 0)
//		{
//			Mtmp = (this->eps_rsvd * iW.value()) * A[loc][s2] * H->R[a];
//			if (Paux[iW.row()][s1].rows() == 0) {Paux[iW.row()][s1] = Mtmp;}
//			else                                {Paux[iW.row()][s1] += Mtmp;}
//		}
//	}
	
//	std::array<MatrixType,D> P;
//	for (size_t s=0; s<D; ++s)
//	for (size_t w=0; w<Daux; ++w)
//	{
//		addBottom(Paux[w][s],P[s]);
//	}
	
	Aclump.conservativeResize(Aclump.rows()+P[0].rows(), Aclump.cols());
	for (size_t s=0; s<D; ++s)
	{
		Aclump.block(Nrows,s*Ncols, P[s].rows(),P[s].cols()) = P[s];
	}
	
//	size_t Nzerorows=0;
//	for (size_t i=0; i<Aclump.rows(); ++i)
//	{
//		if (Aclump.row(i).norm() < 1e-12)
//		{
//			remove_row(i,Aclump);
//			++Nzerorows;
//		}
//	}
//	cout << "Nzerorows=" << Nzerorows << endl;
	
	if (loc != 0)
	{
		for (size_t s=0; s<D; ++s)
		{
			A[loc-1][s].conservativeResize(A[loc-1][s].rows(), A[loc-1][s].cols()+P[s].rows());
			A[loc-1][s].rightCols(P[s].rows()).setZero();
		}
	}
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
enrich_right (size_t loc, PivotMatrix<D> *H, Matrix<Scalar,Dynamic,Dynamic> &Aclump)
{
	size_t Nrows = A[loc][0].rows();
	size_t Ncols = A[loc][0].cols();
	
	std::array<MatrixType,D> P;
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
//	for (size_t a=0; a<H->L.size(); ++a)
	for (int k=0; k<H->W[s1][s2].outerSize(); ++k)
	for (SparseMatrixXd::InnerIterator iW(H->W[s1][s2],k); iW; ++iW)
	{
		size_t a = iW.row();
		MatrixType Mtmp;
		if (H->L[a].rows() != 0)
		{
			Mtmp = (this->eps_rsvd * iW.value()) * H->L[a] * A[loc][s2];
			if (P[s1].rows() == 0) {P[s1] = Mtmp;}
			else                   {P[s1] += Mtmp;}
		}
	}
	
//	size_t Daux = max(H->W[0][0].rows(), H->W[0][0].cols());
//	vector<std::array<MatrixType,D> > Paux(Daux);
//	
//	for (size_t s=0; s<D; ++s)
//	for (size_t w=0; w<Daux; ++w)
//	{
//		if (Paux[w][s].rows() == 0)
//		{
//			Paux[w][s].resize(A[loc][0].rows(), A[loc][0].cols());
//			Paux[w][s].setZero();
//		}
//	}
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t a=0; a<H->L.size(); ++a)
//	for (int k=0; k<H->W[s1][s2].outerSize(); ++k)
//	for (SparseMatrixXd::InnerIterator iW(H->W[s1][s2],k); iW; ++iW)
//	{
//		MatrixType Mtmp;
//		if (H->L[a].rows() != 0)
//		{
//			Mtmp = (this->eps_rsvd * iW.value()) * H->L[a] * A[loc][s2];
//			if (Paux[iW.col()][s1].rows() == 0) {Paux[iW.col()][s1] = Mtmp;}
//			else                                {Paux[iW.col()][s1] += Mtmp;}
//		}
//	}
//	
//	std::array<MatrixType,D> P;
//	for (size_t s=0; s<D; ++s)
//	for (size_t w=0; w<Daux; ++w)
//	{
//		addRight(Paux[w][s],P[s]);
//	}
	
	Aclump.conservativeResize(Aclump.rows(), Aclump.cols()+P[0].cols());
	for (size_t s=0; s<D; ++s)
	{
		Aclump.block(s*Nrows,Ncols, P[s].rows(),P[s].cols()) = P[s];
	}
	
//	size_t Nzerocols=0;
//	for (size_t i=0; i<Aclump.cols(); ++i)
//	{
//		if (Aclump.col(i).norm() < 1e-12)
//		{
//			remove_col(i,Aclump);
//			++Nzerocols;
//		}
//	}
//	cout << "Nzerocols=" << Nzerocols << endl;
	
	for (size_t s=0; s<D; ++s)
	{
		if (loc != this->N_sites-1)
		{
			A[loc+1][s].conservativeResize(A[loc+1][s].rows()+P[s].cols(), A[loc+1][s].cols());
			A[loc+1][s].bottomRows(P[s].cols()).setZero();
		}
	}
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
calc_noise (PivotMatrix<D> *H, DMRG::DIRECTION::OPTION DIR, 
            const std::array<std::array<MatrixType,D>,D> rho, 
            std::array<std::array<MatrixType,D>,D>       &rhoNoise)
{
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	{
		MatrixType Mtmp;
		
		if (DIR == DMRG::DIRECTION::RIGHT)
		{
			for (size_t a=0; a<H->L.size(); ++a)
			{
				if (H->L[a].rows() != 0)
				{
					if (Mtmp.rows() == 0)
					{
						Mtmp = H->L[a] * rho[s1][s2] * H->L[a].adjoint();
					}
					else
					{
						Mtmp += H->L[a] * rho[s1][s2] * H->L[a].adjoint();
					}
				}
			}
		}
		else if (DIR == DMRG::DIRECTION::LEFT)
		{
			for (size_t a=0; a<H->R.size(); ++a)
			{
				if (H->R[a].rows() != 0)
				{
					if (Mtmp.rows() == 0)
					{
						Mtmp = H->R[a].adjoint() * rho[s1][s2] * H->R[a];
					}
					else
					{
						Mtmp += H->R[a].adjoint() * rho[s1][s2] * H->R[a];
					}
				}
			}
		}
		
		if (rhoNoise[s1][s2].rows() == 0)
		{
			rhoNoise[s1][s2] = Mtmp;
		}
		else
		{
			rhoNoise[s1][s2] += Mtmp;
		}
	}
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
collapse()
{
	for (size_t l=0; l<this->N_sites; ++l)
	{
		MatrixXd BasisTrafo = randOrtho(D);
//		MatrixXd BasisTrafo = MatrixXd::Identity(D,D);
		vector<double> prob(D);
		vector<double> ranges(D+1);
		ranges[0] = 0.;
		
		for (size_t i=0; i<D; ++i)
		{
			prob[i] = 0.;
			
			MatrixType Arow = BasisTrafo(0,i) * A[l][0].adjoint();
			for (size_t s=1; s<D; ++s)
			{
				Arow += BasisTrafo(s,i) * A[l][s].adjoint();
			}
			
			MatrixType Acol = BasisTrafo(0,i) * A[l][0];
			for (size_t s=1; s<D; ++s)
			{
				Acol += BasisTrafo(s,i) * A[l][s];
			}
			
			prob[i] += (Acol*Arow)(0,0);
			ranges[i+1] = ranges[i] + prob[i];
		}
//		for (size_t i=0; i<D; ++i) {cout << prob[i] << endl;}
//		cout << ranges[D] << "\t" << fabs(ranges[D]-1.) << endl;
		assert(fabs(ranges[D]-1.) < 1e-10 and 
		       "Probabilities in collapse don't add up to 1!");
		
		double die = UniformDist(MtEngine);
		size_t select;
		for (size_t i=1; i<D+1; ++i)
		{
			if (die>=ranges[i-1] and die<ranges[i])
			{
				select = i-1;
			}
		}
		
		if (l < this->N_sites-1)
		{
			for (size_t s2=0; s2<D; ++s2)
			{
				MatrixXd Mtmp = BasisTrafo(0,select) * A[l][0] * A[l+1][s2];
				for (size_t s1=1; s1<D; ++s1)
				{
					Mtmp += BasisTrafo(s1,select) * A[l][s1] * A[l+1][s2];
				}
				A[l+1][s2] = 1./sqrt(prob[select]) * Mtmp;
			}
		}
		
		for (size_t s1=0; s1<D; ++s1)
		{
			A[l][s1].resize(1,1);
			A[l][s1](0,0) = BasisTrafo(s1,select);
		}
	}
}

template<size_t D, typename Scalar>
string Mps<D,Scalar>::
test_ortho() const
{
	Matrix<Scalar,Dynamic,Dynamic> Test;
	string sout = "";
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		Test = A[l][0].adjoint() * A[l][0];
		for (int s=1; s<D; ++s)
		{
			Test += A[l][s].adjoint() * A[l][s];
		}
		Test -= Matrix<Scalar,Dynamic,Dynamic>::Identity(Test.rows(),Test.cols());
		bool A_CHECK = Test.template lpNorm<Infinity>()<1e-10 ? true : false;
		
		Test = A[l][0] * A[l][0].adjoint();
		for (int s=1; s<D; ++s)
		{
			Test = Test += A[l][s] * A[l][s].adjoint();
		}
		Test -= Matrix<Scalar,Dynamic,Dynamic>::Identity(Test.rows(),Test.cols());
		bool B_CHECK = Test.template lpNorm<Infinity>()<1e-10 ? true : false;
		
		if (A_CHECK and B_CHECK) {sout += "X";}
		else if (A_CHECK)        {sout += "A";}
		else if (B_CHECK)        {sout += "B";}
		else                     {sout += "M";}
	}
	
	return sout;
}

template<size_t D, typename Scalar>
Mps<D,Scalar>& Mps<D,Scalar>::
operator+= (const Mps<D,Scalar> &Vin)
{
	addScale(Vin,+1.);
}

template<size_t D, typename Scalar>
Mps<D,Scalar>& Mps<D,Scalar>::
operator-= (const Mps<D,Scalar> &Vin)
{
	addScale(Vin,-1.);
}

template<size_t D, typename Scalar>
template<typename OtherScalar>
void Mps<D,Scalar>::
add_site (size_t loc, OtherScalar factor, const Mps<D,Scalar> &Vin)
{
	if (loc == 0)
	{
		for (size_t s=0; s<D; ++s)
		{
			addRight(factor*Vin.A[0][s], A[0][s]);
		}
	}
	else if (loc == this->N_sites-1)
	{
		for (size_t s=0; s<D; ++s)
		{
			addBottom(Vin.A[this->N_sites-1][s], A[this->N_sites-1][s]);
		}
	}
	else
	{
		for (size_t s=0; s<D; ++s)
		{
			addBottomRight(Vin.A[loc][s], A[loc][s]);
		}
	}
}

template<size_t D, typename Scalar>
template<typename OtherScalar>
void Mps<D,Scalar>::
addScale (const Mps<D,Scalar> &Vin, OtherScalar factor)
{
	if (&Vin.A == &A) // v+=α·v; results in v*=2·α;
	{
		operator*=(2.*factor);
	}
	else
	{
		add_site(0,factor,Vin);
		add_site(1,factor,Vin);
		rightSweepStep(0,DMRG::BROOM::SVD);
		for (size_t l=2; l<this->N_sites; ++l)
		{
			add_site(l,factor,Vin);
			rightSweepStep(l-1,DMRG::BROOM::SVD);
		}
	}
}

template<size_t D, typename Scalar>
template<typename OtherScalar>
Mps<D,Scalar>& Mps<D,Scalar>::
operator*= (const OtherScalar &factor)
{
	int loc = (this->pivot == -1)? 0 : this->pivot;
	for (size_t s=0; s<D; ++s)
	{
		A[loc][s] *= factor;
	}
}

template<size_t D, typename Scalar>
template<typename OtherScalar>
Mps<D,Scalar>& Mps<D,Scalar>::
operator/= (const OtherScalar &factor)
{
	for (size_t s=0; s<D; ++s)
	{
		A[0][s] /= factor;
	}
}

template<size_t D, typename Scalar, typename OtherScalar>
Mps<D,OtherScalar> operator* (const OtherScalar &alpha, const Mps<D,Scalar> &Vin)
{
	Mps<D,OtherScalar> Vout = Vin.template cast<OtherScalar>();
	Vout *= alpha;
	return Vout;
}

template<size_t D, typename Scalar>
Scalar Mps<D,Scalar>::
dot (const Mps<D,Scalar> &V) const
{
	assert(this->N_sites == V.length());
	
	MatrixType Mtmp = A[0][0].adjoint() * V.A[0][0];
	for (size_t s=1; s<D; ++s)
	{
		Mtmp += A[0][s].adjoint() * V.A[0][s];
	}
	MatrixType Mout = Mtmp;
	
	for (size_t l=1; l<this->N_sites; ++l)
	{
		Mtmp = A[l][0].adjoint() * Mout * V.A[l][0];
		for (size_t s=1; s<D; ++s)
		{
			Mtmp += A[l][s].adjoint() * Mout * V.A[l][s];
		}
		Mout = Mtmp;
	}
	
	assert(Mout.rows() == 1 and
	       Mout.cols() == 1 and
	       "Result of contraction in <φ|ψ> is not a scalar!");
	
	return Mout(0,0);
}

template<size_t D, typename Scalar>
template<typename OtherScalar>
Mps<D,OtherScalar> Mps<D,Scalar>::
cast() const
{
	Mps<D,OtherScalar> Vout(this->N_sites,1);
//	Vout.outerResize(*this);
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[l][s] = A[l][s].template cast<OtherScalar>();
	}
	
	Vout.eps_noise = this->eps_noise;
	Vout.eps_rdm = this->eps_rdm;
	Vout.eps_svd = this->eps_svd;
	Vout.N_sv = this->N_sv;
	Vout.pivot = this->pivot;
	Vout.truncWeight = truncWeight;
	
	return Vout;
}

template<size_t D, typename Scalar>
void Mps<D,Scalar>::
swap (Mps<D,Scalar> &V)
{
	assert(this->N_sites == V.length() and
	       "Need equal sizes for MPS swapping!");
	
	std::swap(this->N_sites, V.N_sites);
	std::swap(this->pivot, V.pivot);
	truncWeight.swap(V.truncWeight);
	
	std::swap(this->eps_noise, V.eps_noise);
	std::swap(this->eps_rdm, V.eps_rdm);
	std::swap(this->eps_svd, V.eps_svd);
	std::swap(this->N_sv, V.N_sv);
	
	for (size_t l=0; l<this->N_sites; ++l)
	for (size_t s=0; s<D; ++s)
	{
		A[l][s].swap(V.A[l][s]);
	}
}

template<size_t D, typename Scalar>
ostream &operator<< (ostream& os, const Mps<D,Scalar> &V)
{
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "Mps: L=" << V.length() << ", D=" << D;
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<V.length(); ++l)
	{
		for (size_t s=0; s<D; ++s)
		{
			os << "l=" << l << "\ts_index=" << s << endl;
			os << V.A_at(l)[s] << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != V.length()-1) {os << endl;}
	}
	return os;
}

#endif
