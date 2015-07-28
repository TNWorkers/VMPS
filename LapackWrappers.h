#ifndef LAPACKWRAPPERS
#define LAPACKWRAPPERS

// real SVD:
extern "C" void dgesdd_ (const char *JOBZ, const int *M, const int *N, 
                         double *A, const int *LDA, 
                         double *S, 
                         double *U, const int *LDU, 
                         double *VT, const int *LDVT, 
                         double *WORK, const int *LWORK, int *IWORK, int *INFO);

// complex SVD:
extern "C" void zgesdd_ (const char *JOBZ, const int *M, const int *N, 
                         complex<double> *A, const int *LDA, 
                         double *S, 
                         complex<double> *U, const int *LDU, 
                         complex<double> *VT, const int *LDVT, 
                         complex<double> *WORK, const int *LWORK, double *RWORK, int *IWORK, int *INFO);

// real symmetrix eigenvalues & eigenvectors:
extern "C" void dsyev_ (const char *JOBZ, const char *UPLO, const int *N, double *A, const int *LDA, double *W, double *WORK, const int *LWORK, int *INFO);

// real QR decomposition:
extern "C" void dgeqrf_ (const int *M, const int *N, double *A, const int *LDA, double *TAU, double *WORK, const int *LWORK, int *INFO);
// QR with column pivoting:
//extern "C" void dgeqp3_ (const int *M, const int *N, double *A, const int *LDA, int *JPVT, double *TAU, double *WORK, const int *LWORK, int *INFO);
// apply reflections to get Q-matrix:
extern "C" void dorgqr_ (const int *M, const int *N, const int *K, double *A, const int *LDA, double *TAU, double *WORK, const int *LWORK, int *INFO);

// complex QR decomposition:
extern "C" void zgeqrf_ (const int *M, const int *N, complex<double> *A, const int *LDA, complex<double> *TAU, complex<double> *WORK, const int *LWORK, int *INFO);
// apply reflections to get Q-matrix:
extern "C" void zungqr_ (const int *M, const int *N, const int *K, complex<double> *A, const int *LDA, complex<double> *TAU, complex<double> *WORK, const int *LWORK, int *INFO);

// Lapack is 10-100 times faster than Eigen
template<typename Scalar>
class LapackSVD
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	void compute (const MatrixType &A);
	
	inline MatrixType matrixU() const      {return U;}
	inline VectorXd singularValues() const {return S;}
	inline MatrixType matrixVT() const     {return VT;}
	
private:
	
	MatrixType U;
	VectorXd S;
	MatrixType VT;
};

template<>
void LapackSVD<double>::
compute (const MatrixXd &A)
{
	int Arows = A.rows();
	int Acols = A.cols();
	int minA  = min(Arows,Acols);
//	cout << "Arows=" << Arows << ", Acols=" << Acols << ", minA=" << minA << endl;
	
	U.resize(Arows,minA);
	S.resize(minA);
	VT.resize(minA,Acols);
//	double * Udata = new double[Arows*minA];
//	double * Sdata = new double[minA];
//	double * VTdata = new double[minA*Acols];
	int LWORK = -1;
	double * WORK = new double[1];
	int * IWORK = new int[8*minA];
	int INFO;
	
//	cout << "Arows=" << Arows << ", Acols=" << Acols << ", minA=" << minA << endl;
	
	MatrixXd Atmp = A;
//	dgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, Sdata, Udata, &Arows, VTdata, &minA, WORK, &LWORK, IWORK, &INFO);
	dgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, S.data(), U.data(), &Arows, VT.data(), &minA, WORK, &LWORK, IWORK, &INFO);
//	cout << "INFO=" << INFO << endl;
	LWORK = WORK[0];
	delete WORK;
	WORK = new double[LWORK];
//	dgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, Sdata, Udata, &Arows, VTdata, &minA, WORK, &LWORK, IWORK, &INFO);
	dgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, S.data(), U.data(), &Arows, VT.data(), &minA, WORK, &LWORK, IWORK, &INFO);
	delete WORK;
	delete IWORK;
//	U = Map<MatrixXd>(Udata,Arows,minA);
//	delete Udata;
//	S = Map<VectorXd>(Sdata,minA,1);
//	delete Sdata;
//	VT = Map<MatrixXd>(VTdata,minA,Acols);
//	delete VTdata;
}

template<>
void LapackSVD<complex<double> >::
compute (const MatrixXcd &A)
{
	int Arows = A.rows();
	int Acols = A.cols();
	int minA  = min(Arows,Acols);
	
	U.resize(Arows,minA);
	S.resize(minA);
	VT.resize(minA,Acols);
//	int LWORK = 2*(minA*minA+2*minA+max(Arows,Acols));
//	complex<double> * Udata = new complex<double>[Arows*minA];
//	double * Sdata = new double[minA];
//	complex<double> * VTdata = new complex<double>[minA*Acols];
	complex<double> * WORK = new complex<double>[1];
	int LWORK = -1;
	int * IWORK = new int[8*minA];
	double * RWORK = new double[5*minA*minA+7*minA];
//	VectorXcd WORK(LWORK);
//	VectorXi IWORK(8*minA);
	int INFO;
//	VectorXd RWORK(5*minA*minA+7*minA);
	
	MatrixXcd Atmp = A;
//	zgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, Sdata, Udata, &Arows, VTdata, &minA, WORK, &LWORK, RWORK, IWORK, &INFO);
	zgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, S.data(), U.data(), &Arows, VT.data(), &minA, WORK, &LWORK, RWORK, IWORK, &INFO);
	LWORK = static_cast<int>(WORK[0].real());
	delete WORK;
	WORK = new complex<double>[LWORK];
//	zgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, Sdata, Udata, &Arows, VTdata, &minA, WORK, &LWORK, RWORK, IWORK, &INFO);
	zgesdd_ ("S", &Arows, &Acols, Atmp.data(), &Arows, S.data(), U.data(), &Arows, VT.data(), &minA, WORK, &LWORK, RWORK, IWORK, &INFO);
	delete WORK;
	delete IWORK;
	delete RWORK;
//	U = Map<MatrixXcd>(Udata,Arows,minA);
//	delete Udata;
//	S = Map<VectorXd>(Sdata,minA,1);
//	delete Sdata;
//	VT = Map<MatrixXcd>(VTdata,minA,Acols);
//	delete VTdata;
}

// Eigen is 2 times faster than Lapack
class LapackSelfAdjointEigenSolver
{
public:
	
	void compute (const MatrixXd &A);
	
	inline VectorXd eigenvalues()  {return D;}
	inline MatrixXd eigenvectors() {return O;}
	
private:
	
	VectorXd D;
	MatrixXd O;
};

void LapackSelfAdjointEigenSolver::
compute (const MatrixXd &A)
{
	assert(A.rows() == A.cols());
	O = A;
	int N = A.rows();
	D.resize(N);
	int LWORK = 2*(3*N-1);
	VectorXd WORK(LWORK);
	int INFO;
	
	dsyev_ ("V", "U", &N, O.data(), &N, D.data(), WORK.data(), &LWORK, &INFO);
}

// Eigen is 1.25 times faster than Lapack
//class LapackQR
//{
//public:
//	
//	void compute (const MatrixXd &A);
//	
//	inline MatrixXd Qmatrix() {return Q;}
//	inline MatrixXd Rmatrix() {return R;}
//	
//private:
//	
//	MatrixXd Q;
//	MatrixXd R;
//};

//void LapackQR::
//compute (const MatrixXd &A)
//{
//	int Arows = A.rows();
//	int Acols = A.cols();
//	int minA  = min(Arows,Acols);
//	R = A;
//	int LWORK = 2*Acols;
//	VectorXd WORK(LWORK);
//	int INFO;
//	VectorXd TAU(minA);
//	
//	dgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK.data(), &LWORK, &INFO);
//	Q = R;
//	dorgqr_ (&Arows, &Acols, &minA, Q.data(), &Arows, TAU.data(), WORK.data(), &LWORK, &INFO);
//	R = MatrixXd::Identity(Acols,Arows) * R.triangularView<Upper>();
//}

template<typename Scalar>
class LapackQR
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	void compute (const MatrixType &A);
	
	inline MatrixType Qmatrix() {return Q;}
	inline MatrixType Rmatrix() {return R;}
	
private:
	
	MatrixType Q;
	MatrixType R;
};

template<>
void LapackQR<double>::
compute (const MatrixXd &A)
{
	int Arows = A.rows();
	int Acols = A.cols();
	int minA  = min(Arows,Acols);
	
//	R = A;
//	int LWORK = 2*Acols;
//	VectorXd WORK(LWORK);
//	int INFO;
//	VectorXd TAU(minA);
//	
//	dgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK.data(), &LWORK, &INFO);
//	Q = R;
//	dorgqr_ (&Arows, &Acols, &minA, Q.data(), &Arows, TAU.data(), WORK.data(), &LWORK, &INFO);
//	R = MatrixXd::Identity(Acols,Arows) * R.triangularView<Upper>();
	
	R = A;
	int LWORK = -1;
	double * WORK = new double[1];
	int INFO;
	VectorXd TAU(minA);
	
	dgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	LWORK = static_cast<int>(WORK[0]);
	delete WORK;
	WORK = new double[LWORK];
	dgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	
	Q = R;
	
	LWORK = -1;
	dorgqr_ (&Arows, &minA, &minA, Q.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	LWORK = static_cast<int>(WORK[0]);
	delete WORK;
	WORK = new double[LWORK];
	dorgqr_ (&Arows, &minA, &minA, Q.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	delete WORK;
	
	R = MatrixXd::Identity(Acols,Arows) * R.triangularView<Upper>();
	
	if (Acols > Arows) {Q.rightCols(Acols-Arows).setZero();}
}

template<>
void LapackQR<complex<double> >::
compute (const MatrixXcd &A)
{
	int Arows = A.rows();
	int Acols = A.cols();
	int minA  = min(Arows,Acols);
	
//	R = A;
//	int LWORK = 2*Acols;
//	VectorXcd WORK(LWORK);
//	int INFO;
//	VectorXcd TAU(minA);
//	
//	zgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK.data(), &LWORK, &INFO);
//	Q = R;
//	zungqr_ (&Arows, &minA, &minA, Q.data(), &Arows, TAU.data(), WORK.data(), &LWORK, &INFO);
//	R = MatrixXcd::Identity(Acols,Arows) * R.triangularView<Upper>();
	
	R = A;
	int LWORK = -1;
	complex<double> * WORK = new complex<double>[2];
	int INFO;
	VectorXcd TAU(minA);
	
	zgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	LWORK = static_cast<int>(WORK[0].real());
	delete WORK;
	WORK = new complex<double>[LWORK];
	zgeqrf_ (&Arows, &Acols, R.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	
	Q = R;
	
	LWORK = -1;
	zungqr_ (&Arows, &minA, &minA, Q.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	LWORK = static_cast<int>(WORK[0].real());
	delete WORK;
	WORK = new complex<double>[LWORK];
	zungqr_ (&Arows, &minA, &minA, Q.data(), &Arows, TAU.data(), WORK, &LWORK, &INFO);
	delete WORK;
	
	R = MatrixXcd::Identity(Acols,Arows) * R.triangularView<Upper>();
	
	if (Acols > Arows) {Q.rightCols(Acols-Arows).setZero();}
}

#endif
