#ifndef FERMIONBASE
#define FERMIONBASE

#include <algorithm>
#include <iterator>
#include <boost/dynamic_bitset.hpp>

#include "SpinBase.h"
#include "DmrgTypedefs.h"

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
	
	FermionBase (size_t L_input);
	
	inline size_t dim() {return N_states;}
	
	vector<boost::dynamic_bitset<unsigned char> > basis;
	
	size_t N_orbitals;
	size_t N_states;
	
	int parity (const boost::dynamic_bitset<unsigned char> &state, int i);
	
	MatrixXd c    (int orbital, SPIN_INDEX sigma);
	MatrixXd cdag (int orbital, SPIN_INDEX sigma);
	MatrixXd docc (int orbital);
	// (1-2*n(i,UP))*(1-2*n(i,DN))
	MatrixXd fsign_ (int orbital);
	MatrixXd fsign_ (int orbital, SPIN_INDEX sigma);
	
	MatrixXd HubbardHamiltonian (double U);
};

//boost::dynamic_bitset<unsigned char> cast_to_bitset (const VectorXd &b)
//{
//	boost::dynamic_bitset<unsigned char> bout(b.rows());
//	for (int i=0; i<b.rows(); ++i)
//	{
//		if (b(i) == 1.) {bout.flip(i);}
//	}
//	return bout;
//}

FermionBase::
FermionBase (size_t L_input)
:N_orbitals(L_input)
{
	N_states = pow(4,N_orbitals);
//	basis.resize(16);
	basis.resize(N_states);
	
//	vector<MatrixXd> OrbitalBase(4);
	vector<int> nUP(4);
	vector<int> nDN(4);
//	OrbitalBase[0] = MatrixXd::Identity(4,4);
//	OrbitalBase[1] = cUP.transpose();
//	OrbitalBase[2] = cDN.cwiseAbs().transpose();
//	OrbitalBase[3] = cUP.transpose()*cDN.cwiseAbs().transpose();
	nUP[0] = 0; nDN[0] = 0;
	nUP[1] = 1; nDN[1] = 0;
	nUP[2] = 0; nDN[2] = 1;
	nUP[3] = 1; nDN[3] = 1;
//	VectorXd vac(16);
//	vac.setZero();
//	vac(0) = 1.;
	
	NestedLoopIterator Nelly(N_orbitals,4);
//	for (int i1=0; i1<4; ++i1)
//	for (int i2=0; i2<4; ++i2)
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
//		basis[i2+4*i1] = cast_to_bitset(kroneckerProduct(OrbitalBase[i1],OrbitalBase[i2]) * vac);
//		cout << (kroneckerProduct(OrbitalBase[i1],OrbitalBase[i2]) * vac).transpose() << endl;
//		basis[i2+4*i1].resize(4);
//		basis[i2+4*i1][0] = nUP[i1];
//		basis[i2+4*i1][1] = nDN[i1];
//		basis[i2+4*i1][2] = nUP[i2];
//		basis[i2+4*i1][3] = nDN[i2];
		basis[Nelly.index()].resize(2*N_orbitals);
		for (int i=0; i<N_orbitals; ++i)
		{
			basis[Nelly.index()][2*i]   = nUP[Nelly(i)];
			basis[Nelly.index()][2*i+1] = nDN[Nelly(i)];
		}
//		cout << i2+4*i1 << "\t:" << nUP[i1] << "\t" << nDN[i1] << "\t" << nUP[i2] << "\t" << nDN[i2] << endl;
	}
	
	for (int i=0; i<basis.size(); ++i)
	{
		cout << i << "\t" << basis[i] << endl;
	}
}

int FermionBase::
parity (const boost::dynamic_bitset<unsigned char> &b, int i)
{
	int out = 1;
	for (int j=0; j<i; ++j)
	{
		if (b[j]) {out *= -1;} // switch sign for every particle found between 0 & i
	}
	return out;
}

MatrixXd FermionBase::
c (int orbital, SPIN_INDEX sigma)
{
//	MatrixXd Mout(16,16);
	MatrixXd Mout(N_states,N_states);
	Mout.setZero();
	for (int j=0; j<basis.size(); ++j)
	{
		if (basis[j][2*orbital+static_cast<int>(sigma)]) // factor 2 because of ordering 1UP,1DN,2UP,2DN,...
		{
			boost::dynamic_bitset<unsigned char> b = basis[j];
			b[2*orbital+static_cast<int>(sigma)].flip();
			
			auto it = find(basis.begin(), basis.end(), b);
			int i = distance(basis.begin(), it);
			
			Mout(i,j) = 1. * parity(b, 2*orbital);
		}
	}
	return Mout;
}

inline MatrixXd FermionBase::
cdag (int orbital, SPIN_INDEX sigma)
{
	return c(orbital,sigma).transpose();
}

inline MatrixXd FermionBase::
docc (int orbital)
{
	return cdag(orbital,UP)*c(orbital,UP) * cdag(orbital,DN)*c(orbital,DN);
}

inline MatrixXd FermionBase::
fsign_ (int orbital)
{
	return (MatrixXd::Identity(N_states,N_states)-2.*cdag(orbital,UP)*c(orbital,UP))*
	       (MatrixXd::Identity(N_states,N_states)-2.*cdag(orbital,DN)*c(orbital,DN));
}

inline MatrixXd FermionBase::
fsign_ (int orbital, SPIN_INDEX sigma)
{
	return MatrixXd::Identity(N_states,N_states)-2.*cdag(orbital,sigma)*c(orbital,sigma);
}

MatrixXd FermionBase::
HubbardHamiltonian (double U)
{
	MatrixXd Mout(N_states,N_states);
	Mout.setZero();
	
	for (int i=0; i<N_orbitals-1; ++i) // for all bonds
	{
		MatrixXd T = cdag(i,UP)*c(i+1,UP) + cdag(i,DN)*c(i+1,DN);
		Mout += -1.*(T+T.transpose());
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		Mout += U*docc(i);
	}
	return Mout;
}

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
