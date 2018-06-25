#ifndef VUMPSTYPEDEFS
#define VUMPSTYPEDEFS

/**Gauge of the UMPS tensor: \p L (left gauge), \p R (right gauge), or \p C (no gauge).*/
struct GAUGE
{
	enum OPTION {L=0, R=1, C=2};
};

std::ostream& operator<< (std::ostream& s, GAUGE::OPTION g)
{
	if      (g==GAUGE::OPTION::L) {s << "L";}
	else if (g==GAUGE::OPTION::R) {s << "R";}
	else if (g==GAUGE::OPTION::C) {s << "C";}
	return s;
}

struct VUMPS_ALG
{
	enum OPTION {PARA=0, SEQU=1};
};

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

#endif
