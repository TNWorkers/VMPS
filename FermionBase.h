#ifndef FERMIONBASE
#define FERMIONBASE

#include "SpinBase.h"

struct FermionBase
{
	static const Eigen::Matrix<double,4,4,Eigen::RowMajor> Scomp (SPINOP_LABEL Sa)
	{
		assert(Sa != SY);
		
		if      (Sa==SX)  {return Sx;}
		else if (Sa==iSY) {return iSy;}
		else if (Sa==SZ)  {return Sz;}
		else if (Sa==SP)  {return Sp;}
		else if (Sa==SM)  {return Sm;}
	}
	
	///@{
	/**
	\f$c_{\uparrow} = \left(
	\begin{array}{cccc}
	0 & 1 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 1\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> cUP;
	
	/**
	\f$c_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & -1\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> cDN;
	
	/**
	\f$d = n_{\uparrow}n_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 1\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> d;
	
	/**
	\f$n = n_{\uparrow}+n_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 1 & 0 & 0\\
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & 2\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> n;
	
	/**
	\f$(1-2n_{\uparrow})*(1-2n_{\downarrow}) = \left(
	\begin{array}{cccc}
	1 & 0  & 0  & 0\\
	0 & -1 & 0  & 0\\
	0 & 0  & -1 & 0\\
	0 & 0  & 0  & 1\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> fsign;
	
	/**
	\f$s^+ = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> Sp;
	
	/**
	\f$s^+ = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 1 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> Sm;
	
	/**
	\f$s^x = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0.5 & 0\\
	0 & 0.5 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> Sx;
	
	/**
	\f$is^y = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0.5 & 0\\
	0 & -0.5 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> iSy;
	
	/**
	\f$s^z = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0.5 & 0 & 0\\
	0 & 0 & -0.5 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> Sz;
	///@}
};

static const double cUP_data[] =
{
	0., 1., 0., 0.,
	0., 0., 0., 0.,
	0., 0., 0., 1.,
	0., 0., 0., 0.
};
static const double cDN_data[] =
{
	0., 0., 1., 0.,
	0., 0., 0., -1.,
	0., 0., 0., 0.,
	0., 0., 0., 0.
};
static const double d_data[] =
{
	0., 0., 0., 0.,
	0., 0., 0., 0.,
	0., 0., 0., 0.,
	0., 0., 0., 1.
};
static const double n_data[] =
{
	0., 0., 0., 0.,
	0., 1., 0., 0.,
	0., 0., 1., 0.,
	0., 0., 0., 2.
};
static const double fsign_data[] =
{
	1.,  0.,  0., 0.,
	0., -1.,  0., 0.,
	0.,  0., -1., 0.,
	0.,  0.,  0., 1.
};
static const double SpHub_data[] =
{
	0., 0., 0., 0.,
	0., 0., 1., 0.,
	0., 0., 0., 0.,
	0., 0., 0., 0.
};
static const double SmHub_data[] =
{
	0., 0., 0., 0.,
	0., 0., 0., 0.,
	0., 1., 0., 0.,
	0., 0., 0., 0.
};
static const double SxHub_data[] =
{
	0., 0.,  0.,  0.,
	0., 0.,  0.5, 0.,
	0., 0.5, 0.,  0.,
	0., 0.,  0.,  0.
};
static const double iSyHub_data[] =
{
	0., 0.,   0.,  0.,
	0., 0.,   0.5,  0.,
	0., -0.5, 0., 0.,
	0., 0.,   0.,  0.
};
static const double SzHub_data[] =
{
	0., 0.,   0.,  0.,
	0., 0.5,  0.,  0.,
	0., 0.,  -0.5, 0.,
	0., 0.,   0.,  0.
};

const Eigen::Matrix<double,4,4,RowMajor> FermionBase::cUP(cUP_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::cDN(cDN_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::d(d_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::n(n_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::fsign(fsign_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::Sx(SxHub_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::iSy(iSyHub_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::Sz(SzHub_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::Sp(SpHub_data);
const Eigen::Matrix<double,4,4,RowMajor> FermionBase::Sm(SmHub_data);

#endif
