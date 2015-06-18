#ifndef SPINBASE
#define SPINBASE

#include <Eigen/Dense>
#include <complex>

enum SPINOP_LABEL {SX, SY, iSY, SZ, SP, SM};

std::ostream& operator<< (std::ostream& s, SPINOP_LABEL Sa)
{
	if      (Sa==SX)  {s << "Sx";}
	else if (Sa==SY)  {s << "Sy";}
	else if (Sa==iSY) {s << "iSy";}
	else if (Sa==SZ)  {s << "Sz";}
	else if (Sa==SP)  {s << "S+";}
	else if (Sa==SM)  {s << "S-";}
	return s;
}

template<int D>
struct SpinBase
{
	///@{
	/**\f$S^x\f$*/
	static const Eigen::Matrix<double,D,D,Eigen::RowMajor> Sx;
	/**\f$S^y\f$*/
	static const Eigen::Matrix<complex<double>,D,D,Eigen::RowMajor> Sy;
	/**\f$iS^y\f$*/
	static const Eigen::Matrix<double,D,D,Eigen::RowMajor> iSy;
	/**\f$S^z\f$*/
	static const Eigen::Matrix<double,D,D,Eigen::RowMajor> Sz;
	/**\f$S^+\f$*/
	static const Eigen::Matrix<double,D,D,Eigen::RowMajor> Sp;
	///@}
	
	static const Eigen::Matrix<double,D,D,Eigen::RowMajor> Scomp (SPINOP_LABEL Sa)
	{
		assert(Sa != SY);
		
		if      (Sa==SX)  {return Sx;}
		else if (Sa==iSY) {return iSy;}
		else if (Sa==SZ)  {return Sz;}
		else if (Sa==SP)  {return Sp;}
		else if (Sa==SM)  {return Sp.transpose();}
	}
};

//------------N=2------------

///@{
/**
\f$S^x = \left(
\begin{array}{cc}
0 & 0.5 \\
0.5 & 0 \\
\end{array}
\right)\f$
*/
static const double Sx2_data[] = 
{0.,  0.5, 
 0.5, 0.};

/**
\f$S^y = \left(
\begin{array}{cc}
0 & -0.5i \\
0.5i & 0 \\
\end{array}
\right)\f$
*/
static const complex<double> Sy2_data[] = 
{complex<double>(0.,0.),  complex<double>(0.,-0.5), 
 complex<double>(0.,0.5), complex<double>(0.,0.)};

/**
\f$S^x = \left(
\begin{array}{cc}
0 & 0.5 \\
-0.5 & 0 \\
\end{array}
\right)\f$
*/
static const double iSy2_data[] = 
{0.,   0.5, 
 -0.5, 0.};

/**
\f$S^z = \left(
\begin{array}{cc}
0.5 & 0 \\
0 & -0.5 \\
\end{array}
\right)\f$
*/
static const double Sz2_data[] = 
{0.5, 0., 
 0., -0.5};

/**
\f$S^+ = \left(
\begin{array}{cc}
0 & 1 \\
0 & 0 \\
\end{array}
\right)\f$
*/
static const double Sp2_data[] = 
{0., 1.,
 0., 0.};
///@}

template<> const Eigen::Matrix<double,2,2,Eigen::RowMajor> SpinBase<2>::Sx(Sx2_data);
template<> const Eigen::Matrix<complex<double>,2,2,Eigen::RowMajor> SpinBase<2>::Sy(Sy2_data);
template<> const Eigen::Matrix<double,2,2,Eigen::RowMajor> SpinBase<2>::iSy(iSy2_data);
template<> const Eigen::Matrix<double,2,2,Eigen::RowMajor> SpinBase<2>::Sz(Sz2_data);
template<> const Eigen::Matrix<double,2,2,Eigen::RowMajor> SpinBase<2>::Sp(Sp2_data);

//------------N=3------------

///@{
/**
\f$S^x = \frac{1}{\sqrt{2}} \left(
\begin{array}{ccc}
0 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 0 \\
\end{array}
\right)\f$
*/
static const double Sx3_data[] = 
{0.,        M_SQRT1_2, 0., 
 M_SQRT1_2, 0.,        M_SQRT1_2,
 0.,        M_SQRT1_2, 0.};

/**
\f$S^y = \frac{1}{\sqrt{2}} \left(
\begin{array}{ccc}
0 & -i & 0 \\
i & 0 & -i \\
0 & i & 0 \\
\end{array}
\right)\f$
*/
static const complex<double> Sy3_data[] = 
{complex<double>(0.,0.),         complex<double>(0.,-M_SQRT1_2), complex<double>(0.,0.), 
 complex<double>(0.,+M_SQRT1_2), complex<double>(0.,0.),         complex<double>(0.,-M_SQRT1_2),
 complex<double>(0.,0.),         complex<double>(0.,+M_SQRT1_2), complex<double>(0.,0.)};
 
/**
\f$iS^y = \frac{1}{\sqrt{2}} \left(
\begin{array}{ccc}
0 & 1 & 0 \\
-1 & 0 & 1 \\
0 & -1 & 0 \\
\end{array}
\right)\f$
*/
static const double iSy3_data[] = 
{0.,         M_SQRT1_2, 0., 
 -M_SQRT1_2, 0.,        M_SQRT1_2,
 0.,         M_SQRT1_2, 0.};

/**
\f$S^z = \left(
\begin{array}{ccc}
1 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & -1 \\
\end{array}
\right)\f$
*/
static const double Sz3_data[] = 
{1., 0.,  0., 
 0., 0.,  0.,
 0., 0., -1.};

/**
\f$S^+ = \sqrt{2}\left(
\begin{array}{ccc}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0 \\
\end{array}
\right)\f$
*/
static const double Sp3_data[] = 
{0., M_SQRT2, 0., 
 0., 0.,      M_SQRT2,
 0., 0.,      0.};
///@}

template<> const Eigen::Matrix<double,3,3,Eigen::RowMajor> SpinBase<3>::Sx(Sx3_data);
template<> const Eigen::Matrix<complex<double>,3,3,Eigen::RowMajor> SpinBase<3>::Sy(Sy3_data);
template<> const Eigen::Matrix<double,3,3,Eigen::RowMajor> SpinBase<3>::iSy(iSy3_data);
template<> const Eigen::Matrix<double,3,3,Eigen::RowMajor> SpinBase<3>::Sz(Sz3_data);
template<> const Eigen::Matrix<double,3,3,Eigen::RowMajor> SpinBase<3>::Sp(Sp3_data);

#endif
