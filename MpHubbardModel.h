#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "MpoQ.h"

namespace VMPS
{

/**MPO representation of \f$H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
                                + U \sum_i n_{i\uparrow} n_{i\downarrow}\f$.*/
class HubbardModel : public MpoQ<4,2,double>
{
public:
	
	/**
	@param L_input : chain length
	@param U_input : \f$U\f$
	@param V_input : \f$V\f$
	@param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HubbardModel (size_t L_input, double U_input, double V_input=0., bool CALC_SQUARE=true);
	
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
	
	static SuperMatrix<4> Generator (double U, double V=0.);
	
	MpoQ<4,2> Hsq();
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$*/
	static const std::array<qarray<2>,4> qloc;
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$*/
	static const std::array<string,2> Nlabel;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<4,2,double>                     StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<4,2,complex<double> >           StateXcd;
	typedef DmrgSolverQ<4,2,HubbardModel>        Solver;
	typedef MpsQCompressor<4,2,double>           CompressorXd;
	typedef MpsQCompressor<4,2,complex<double> > CompressorXcd;
	typedef MpoQ<4,2>                          Operator;
	
	static MpoQ<4,2> Auger (size_t L, size_t loc);
	static MpoQ<4,2> Aps (size_t L, size_t loc);
	static MpoQ<4,2> annihilator (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<4,2> creator     (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<4,2> d (size_t L, size_t loc); // double occupancy
	static MpoQ<4,2> n (size_t L, SPIN_INDEX sigma, size_t loc);
	static MpoQ<4,2> triplon (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<4,2> antitriplon (size_t L, size_t loc, SPIN_INDEX sigma);
	static MpoQ<4,2> quadruplon (size_t L, size_t loc);
	
private:
	
	double U, V;
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

const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::cUP(cUP_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::cDN(cDN_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::nUP_nDN(nUP_nDN_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::nUP_plus_nDN(nUP_plus_nDN_data);
const Eigen::Matrix<double,4,4,RowMajor> HubbardModel::fsign(fsign_data);

const std::array<qarray<2>,4> HubbardModel::qloc {qarray<2>{0,0}, qarray<2>{1,0}, qarray<2>{0,1}, qarray<2>{1,1}};
const std::array<string,2>    HubbardModel::Nlabel{"N↑","N↓"};

SuperMatrix<4> HubbardModel::
Generator (double U, double V)
{
	SuperMatrix<4> G;
	size_t Daux = (V==0.)? 6 : 7;
	G.setMatrix(Daux);
	G.setZero();
	
	G(0,0).setIdentity();
	G(1,0) = cUP.transpose();
	G(2,0) = cDN.transpose();
	G(3,0) = cUP;
	G(4,0) = cDN;
	if (V!=0.) {G(5,0) = nUP_plus_nDN;}
	
	G(Daux-1,0) = U*nUP_nDN;
	
	// note: fsign takes care of the fermionic sign
	G(Daux-1,1) = fsign * cUP;
	G(Daux-1,2) = fsign * cDN;
	G(Daux-1,3) = -fsign * cUP.transpose();
	G(Daux-1,4) = -fsign * cDN.transpose();
	if (V!=0.) {G(Daux-1,5) = V*(nUP_plus_nDN);}
	G(Daux-1,Daux-1).setIdentity();
	
	// apply fermionic minus sign:
//	for (size_t a2=1; a2<=4; ++a2)
//	for (size_t s1=0; s1<4;  ++s1)
//	for (size_t s2=1; s2<=2; ++s2)
//	{
//		G(5,a2)(s1,s2) *= -1;
//	}
	
	// or like that:
//	for (size_t a2=1; a2<=4; ++a2)
//	for (size_t s2=0; s2<4;  ++s2)
//	for (size_t s1=1; s1<=2; ++s1)
//	{
//		G(Daux-1,a2)(s1,s2) *= -1;
//	}
	
	return G;
}

//SuperMatrix<4> HubbardModel::
//TevolGenerator (double U)
//{
//	std::array<std::array<std::array<std::array<Matrix<Scalar>,D>,D>,D>,D> Hloc;
//	std::array<std::array<std::array<std::array<Matrix<complex<double> >,D>,D>,D>,D> Hexp;
//	SuperMatrix<4> Generator G(U);
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t r1=0; r1<D; ++r1)
//	for (size_t r2=0; r2<D; ++r2)
//	{
//		Hloc[s1][s2][r1][r2] = TensorProduct(G(1,0),G(G.auxdim()-1,1));
//		for (size_t a=2; a<G.auxdim()-1; ++a)
//		{
//			Hloc[s1][s2][r1][r2] += TensorProduct(G(a,0),G(G.auxdim()-1,a));
//		}
//		Hloc[s1][s2][r1][r2] += TensorProduct(G(G.auxdim()-1,0), Matrix<Scalar,D,D>::Identity());
//		
//		SelfadjointEigenSolver<Matrix<Scalar,D,D> > Eugen(Hloc[s1][s2][r1][r2]);
//		Hloc[s1][s2][r1][r2] = Eugen.eigenvectors().adjoint() * (()*Eugen.eigenvalues().cwise().exp()) * Eugen.eigenvectors();
//	}
//}

HubbardModel::
HubbardModel (size_t L_input, double U_input, double V_input, bool CALC_SQUARE)
:MpoQ<4,2> (L_input, HubbardModel::qloc, {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ")";
	this->label += ss.str();
	
	this->Daux = (V==0.)? 6 : 7;
	this->N_sv = this->Daux;
	
	SuperMatrix<4> G = Generator(U,V);
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

MpoQ<4,2> HubbardModel::
Hsq()
{
	SuperMatrix<4> G = Generator(U,V);
	MpoQ<4,2> Mout(this->N_sites, tensor_product(G,G), HubbardModel::qloc, {0,0}, HubbardModel::Nlabel, "HubbardModel^2");
	return Mout;
}

MpoQ<4,2> HubbardModel::
Auger (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Auger(" << loc << ")";
	MpoQ<4,2> Mout(L, HubbardModel::qloc, {-1,-1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, cUP*cDN);
	return Mout;
}

MpoQ<4,2> HubbardModel::
Aps (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "Aps(" << loc << ")";
	MpoQ<4,2> Mout(L, HubbardModel::qloc, {+1,+1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, cDN.transpose()*cUP.transpose());
	return Mout;
}

MpoQ<4,2> HubbardModel::
annihilator (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "c(" << loc << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-1,0} : qdiff = {0,-1};
//	MpoQ<4,2> Mout(L, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
	
	vector<SuperMatrix<4> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0) = fsign;
	}
	M[loc].setMatrix(1);
	M[loc](0,0) = (sigma==UP)? cUP : cDN;
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0).setIdentity();
	}
	
//	Mout.construct(M, Mout.W, Mout.Gvec);
	return MpoQ<4,2>(L, M, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<4,2> HubbardModel::
creator (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "c†(" << loc << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+1,0} : qdiff = {0,+1};
//	MpoQ<4,2> Mout(L, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
	
	vector<SuperMatrix<4> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0) = fsign;
	}
	M[loc].setMatrix(1);
	M[loc](0,0) = (sigma==UP)? cUP.transpose() : cDN.transpose();
	for (size_t l=loc+1; l<L; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0).setIdentity();
	}
	
//	Mout.construct(M, Mout.W, Mout.Gvec);
//	return Mout;
	return MpoQ<4,2>(L, M, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<4,2> HubbardModel::
triplon (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "triplon(" << loc << ")" << "c(" << loc+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
//	MpoQ<4,2> Mout(L, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
	
	vector<SuperMatrix<4> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0) = fsign;
	}
	// c(loc,UP)*c(loc,DN)
	M[loc].setMatrix(1);
	M[loc](0,0) = cUP*cDN;
	// c(loc+1,UP|DN)
	M[loc+1].setMatrix(1);
	M[loc+1](0,0) = (sigma==UP)? cUP : cDN;
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0).setIdentity();
	}
	
//	Mout.construct(M, Mout.W, Mout.Gvec);
//	return Mout;
	return MpoQ<4,2>(L, M, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<4,2> HubbardModel::
antitriplon (size_t L, size_t loc, SPIN_INDEX sigma)
{
	assert(loc<L);
	stringstream ss;
	ss << "antitriplon(" << loc << ")" << "c(" << loc+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
//	MpoQ<4,2> Mout(L, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
	
	vector<SuperMatrix<4> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0) = fsign;
	}
	// c†(loc,DN)*c†(loc,UP)
	M[loc].setMatrix(1);
	M[loc](0,0) = cDN.transpose()*cUP.transpose();
	// c†(loc+1,UP|DN)
	M[loc+1].setMatrix(1);
	M[loc+1](0,0) = (sigma==UP)? cUP.transpose() : cDN.transpose();
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0).setIdentity();
	}
	
//	Mout.construct(M, Mout.W, Mout.Gvec);
//	return Mout;
	return MpoQ<4,2>(L, M, HubbardModel::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<4,2> HubbardModel::
quadruplon (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "quadruplon(" << loc << ")" << "Auger(" << loc+1 << ")";
//	MpoQ<4,2> Mout(L, HubbardModel::qloc, {-2,-2}, HubbardModel::Nlabel, ss.str());
	
	vector<SuperMatrix<4> > M(L);
	for (size_t l=0; l<loc; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0).setIdentity();
	}
	// c(loc,UP)*c(loc,DN)
	M[loc].setMatrix(1);
	M[loc](0,0) = cUP*cDN;
	// c(loc+1,UP)*c(loc+1,DN)
	M[loc+1].setMatrix(1);
	M[loc+1](0,0) = cUP*cDN;
	for (size_t l=loc+2; l<L; ++l)
	{
		M[l].setMatrix(1);
		M[l](0,0).setIdentity();
	}
	
//	Mout.construct(M, Mout.W, Mout.Gvec);
//	return Mout;
	return MpoQ<4,2>(L, M, HubbardModel::qloc, {-2,-2}, HubbardModel::Nlabel, ss.str());
}

MpoQ<4,2> HubbardModel::
d (size_t L, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "double_occ(" << loc << ")";
	MpoQ<4,2> Mout(L, HubbardModel::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(loc, HubbardModel::nUP_nDN);
	return Mout;
}

MpoQ<4,2> HubbardModel::
n (size_t L, SPIN_INDEX sigma, size_t loc)
{
	assert(loc<L);
	stringstream ss;
	ss << "n(" << loc << ",σ=" << sigma << ")";
	MpoQ<4,2> Mout(L, HubbardModel::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	(sigma==UP)? Mout.setLocal(loc, HubbardModel::cUP.transpose()*HubbardModel::cUP):
	             Mout.setLocal(loc, HubbardModel::cDN.transpose()*HubbardModel::cDN);
	return Mout;
}

}

#endif
