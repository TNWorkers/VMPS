#define DONT_USE_BDCSVD
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#ifdef MKL
#define EIGEN_USE_MKL_VML
#endif
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

//*************************************************************************************************************//
//Include this file to use BLAS and LAPACK as a backend for linear algebra operations.
//See http://eigen.tuxfamily.org/dox/TopicUsingBlasLapack.html for what algorithms are ported to BLAS/LAPACK
//Remark: Linking against a BLAS and LAPACK implementations is mandatory. (For example -lopenblas and -llapack)
//*************************************************************************************************************//

