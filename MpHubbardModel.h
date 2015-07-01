#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "MpoQ.h"
#include "SpinBase.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}
\f$.
\note If nnn-hopping is positive, the GS-energy is lowered.*/
class HubbardModel : public MpoQ<2,double>
{
public:
	
	/**
	\param L_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HubbardModel (size_t L_input, double U_input, double V_input=0., double tPrime_input=0., bool CALC_SQUARE=true);
	
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
	\f$d = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 1\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> nUP_nDN;
	
	/**
	\f$n_{\uparrow}+n_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 1 & 0 & 0\\
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & 2\\
	\end{array}
	\right)\f$
	*/
	static const Eigen::Matrix<double,4,4,RowMajor> nUP_plus_nDN;
	
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

	static const Eigen::Matrix<double,4,4,Eigen::RowMajor> Scomp (SPINOP_LABEL Sa)
	{
		assert(Sa != SY);
		
		if      (Sa==SX)  {return Sx;}
		else if (Sa==iSY) {return iSy;}
		else if (Sa==SZ)  {return Sz;}
		else if (Sa==SP)  {return Sp;}
		else if (Sa==SM)  {return Sp.transpose();}
	}

	static SuperMatrix<double> Generator (double U, double V=0., double tPrime=0.);
	
	MpoQ<2> Hsq();
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N_{\uparrow}\f$ and \f$N_{\downarrow}\f$. Used by default.*/
	static const std::array<qarray<2>,4> qloc;
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N=N_{\uparrow}+N_{\downarrow}\f$ and \f$2M=N_{\uparrow}-N_{\downarrow}\f$. Used in combination with KondoModel.*/
	static const std::array<qarray<2>,4> qlocNM;
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,2> Nlabel;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,double>                           StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<2,HubbardModel>              Solver;
	typedef MpsQCompressor<2,double,double>          CompressorXd;
	typedef MpsQCompressor<2,complex<double>,double> CompressorXcd;
	typedef MpoQ<2,double>                           OperatorXd;
	typedef MpoQ<2,complex<double> >                 OperatorXcd;
	
	static MpoQ<2> Auger (size_t L, size_t loc);
	static MpoQ<2> Aps (size_t L, size_t loc);
	static MpoQ<2> annihilator (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> creator     (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> d (size_t L, size_t loc); // double occupancy
	static MpoQ<2> n (size_t L, SPIN_INDEX sigma, size_t loc);
	static MpoQ<2> SzOp (size_t L, size_t loc);
	static MpoQ<2> triplon (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> antitriplon (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<2> quadruplon (size_t L, size_t loc);
	
private:
	
	double U, V, tPrime;
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
static const double nUP_nDN_data[] =
{
	0., 0., 0., 0.,
	0., 0., 0., 0.,
	0., 0., 0., 0.,
	0., 0., 0., 1.
};
static const double nUP_plus_nDN_data[] =
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

const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::cUP(cUP_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::cDN(cDN_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::nUP_nDN(nUP_nDN_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::nUP_plus_nDN(nUP_plus_nDN_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::fsign(fsign_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::Sx(SxHub_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::iSy(iSyHub_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::Sz(SzHub_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::Sp(SpHub_data);

const std::array<qarray<2>,4> HubbardModel::qloc {qarray<2>{0,0}, qarray<2>{1,0}, qarray<2>{0,1}, qarray<2>{1,1}};
const std::array<qarray<2>,4> HubbardModel::qlocNM {qarray<2>{0,0}, qarray<2>{1,1}, qarray<2>{1,-1}, qarray<2>{2,0}};
const std::array<string,2>    HubbardModel::Nlabel{"N↑","N↓"};

SuperMatrix<double> HubbardModel::
Generator (double U, double V, double tPrime)
{
	SuperMatrix<double> G;
	if(V==0)
	{
		if(tPrime==0)
		{
			size_t Daux = 6;
			G.setMatrix(Daux,4);
			G.setZero();

			G(0,0).setIdentity();
			G(1,0) = cUP.transpose();
			G(2,0) = cDN.transpose();
			G(3,0) = cUP;
			G(4,0) = cDN;

			G(Daux-1,0) = U*nUP_nDN;

			// note: fsign takes care of the fermionic sign
			G(Daux-1,1) = fsign * cUP;
			G(Daux-1,2) = fsign * cDN;
			G(Daux-1,3) = -fsign * cUP.transpose();
			G(Daux-1,4) = -fsign * cDN.transpose();

			G(Daux-1,Daux-1).setIdentity();
		}
		else
		{
			size_t Daux = 14;
			G.setMatrix(Daux,4);
			G.setZero();

			G(0,0).setIdentity();
			G(1,0) = tPrime*cDN;
			G(2,0) = tPrime*cUP;
			G(3,0) = tPrime*cDN.transpose();
			G(4,0) = tPrime*cUP.transpose();
			G(5,0) = cUP.transpose();
			G(6,0) = cDN.transpose();
			G(7,0) = cUP;
			G(8,0) = cDN;

			G(13,0) = U*nUP_nDN;

			G(13,5) = fsign * cUP;
			G(13,6) = fsign * cDN;
			G(13,7) = -fsign * cUP.transpose();
			G(13,8) = -fsign * cDN.transpose();
			G(13,9) = -fsign * cUP;
			G(13,10) = -fsign * cDN;
			G(13,11) = fsign * cUP.transpose();
			G(13,12) = fsign * cDN.transpose();

			G(9,4) = fsign;
			G(10,3) = fsign;
			G(11,2) = fsign;
			G(12,1) = fsign;
			G(13,13).setIdentity();
		}
	}

	else
	{
		if(tPrime==0)
		{
			size_t Daux = 7;
			G.setMatrix(Daux,4);
			G.setZero();
	
			G(0,0).setIdentity();
			G(1,0) = cUP.transpose();
			G(2,0) = cDN.transpose();
			G(3,0) = cUP;
			G(4,0) = cDN;
			G(5,0) = nUP_plus_nDN;
	
			G(Daux-1,0) = U*nUP_nDN;
	
			// note: fsign takes care of the fermionic sign
			G(Daux-1,1) = fsign * cUP;
			G(Daux-1,2) = fsign * cDN;
			G(Daux-1,3) = -fsign * cUP.transpose();
			G(Daux-1,4) = -fsign * cDN.transpose();
			G(Daux-1,5) = V*(nUP_plus_nDN);
			G(Daux-1,Daux-1).setIdentity();
		}
		else
		{
			size_t Daux = 15;
			G.setMatrix(Daux,4);
			G.setZero();

			G(0,0).setIdentity();
			G(1,0) = tPrime*cDN;
			G(2,0) = tPrime*cUP;
			G(3,0) = tPrime*cDN.transpose();
			G(4,0) = tPrime*cUP.transpose();
			G(5,0) = cUP.transpose();
			G(6,0) = cDN.transpose();
			G(7,0) = cUP;
			G(8,0) = cDN;
			G(9,0) = nUP_plus_nDN;
			
			G(14,0) = U*nUP_nDN;

			G(14,5) = fsign * cUP;
			G(14,6) = fsign * cDN;
			G(14,7) = -fsign * cUP.transpose();
			G(14,8) = -fsign * cDN.transpose();
			G(14,9) = V*(nUP_plus_nDN);
			G(14,10) = -fsign * cUP;
			G(14,11) = -fsign * cDN;
			G(14,12) = fsign * cUP.transpose();
			G(14,13) = fsign * cDN.transpose();

			G(10,4) = fsign;
			G(11,3) = fsign;
			G(12,2) = fsign;
			G(13,1) = fsign;
			G(14,14).setIdentity();
		}
	}
	return G;
}

HubbardModel::
HubbardModel (size_t L_input, double U_input, double V_input, double tPrime_input, bool CALC_SQUARE)
:MpoQ<2> (L_input, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
	U(U_input), V(V_input), tPrime(tPrime_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",tPrime=" << tPrime << ")";
	this->label += ss.str();

	if(tPrime == 0)
	{
		this->Daux = (V==0.)? 6 : 7;
	}
	else
	{
		this->Daux = (V==0.)? 14 : 15;		
	}
	
	SuperMatrix<double> G = Generator(U,V,tPrime);
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

MpoQ<2> HubbardModel::
Hsq()
{
	SuperMatrix<double> G = Generator(U,V);
	MpoQ<2> Mout(this->N_sites, tensor_product(G,G), vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), 
	             {0,0}, HubbardModel::Nlabel, "HubbardModel H^2");
	return Mout;
}

MpoQ<2> HubbardModel::
Auger (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {-1,-1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, cUP*cDN);
	return Mout;
}

MpoQ<2> HubbardModel::
Aps (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Aps(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {+1,+1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, cDN.transpose()*cUP.transpose());
	return Mout;
}

MpoQ<2> HubbardModel::
annihilator (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "c(" << loc << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-1,0} : qdiff = {0,-1};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = fsign;
	}
	M[loc].setMatrix(1,4);
	M[loc](0,0) = (sigma==UP)? cUP : cDN;
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
creator (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "c†(" << loc << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+1,0} : qdiff = {0,+1};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = fsign;
	}
	M[loc].setMatrix(1,4);
	M[loc](0,0) = (sigma==UP)? cUP.transpose() : cDN.transpose();
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
triplon (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "triplon(" << loc << ")" << "c(" << loc+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = fsign;
	}
	// c(loc,UP)*c(loc,DN)
	M[loc].setMatrix(1,4);
	M[loc](0,0) = cUP*cDN;
	// c(loc+1,UP|DN)
	M[loc+1].setMatrix(1,4);
	M[loc+1](0,0) = (sigma==UP)? cUP : cDN;
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
antitriplon (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "antitriplon(" << loc << ")" << "c(" << loc+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = fsign;
	}
	// c†(loc,DN)*c†(loc,UP)
	M[loc].setMatrix(1,4);
	M[loc](0,0) = cDN.transpose()*cUP.transpose();
	// c†(loc+1,UP|DN)
	M[loc+1].setMatrix(1,4);
	M[loc+1](0,0) = (sigma==UP)? cUP.transpose() : cDN.transpose();
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
quadruplon (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")" << "Auger(" << loc+1 << ")";
	
	vector<SuperMatrix<double> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	// c(loc,UP)*c(loc,DN)
	M[loc].setMatrix(1,4);
	M[loc](0,0) = cUP*cDN;
	// c(loc+1,UP)*c(loc+1,DN)
	M[loc+1].setMatrix(1,4);
	M[loc+1](0,0) = cUP*cDN;
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(L, M, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {-2,-2}, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
d (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "double_occ(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, HubbardModel::nUP_nDN);
	return Mout;
}

MpoQ<2> HubbardModel::
n (size_t L, SPIN_INDEX sigma, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "n(" << loc << ",σ=" << sigma << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, ss.str());
	(sigma==UP)? Mout.setLocal(loc, HubbardModel::cUP.transpose()*HubbardModel::cUP):
	             Mout.setLocal(loc, HubbardModel::cDN.transpose()*HubbardModel::cDN);
	return Mout;
}

MpoQ<2> HubbardModel::
SzOp (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<2> Mout(L, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, HubbardModel::Sz);
	return Mout;
}

}

#endif
