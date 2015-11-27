#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "MpoQ.h"
#include "FermionBase.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.*/
class HubbardModel : public MpoQ<2,double>
{
public:
	
	/**
	\param Lx_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$, next-nearest-neighbour (nnn) hopping. A minus sign in front of the hopping terms is assumed, so that \f$t^{\prime}>0\f$ is the usual choice.
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HubbardModel (size_t Lx_input, double U_input, double V_input=0., double tPrime_input=0., size_t Ly_input=1, bool CALC_SQUARE=true);
	
//	static SuperMatrix<double> Generator (double U, double V=0., double tPrime=0.);
	
	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. KondoModel.
	\param Olocal : the local interaction terms
	\param Otight : the tight-binding terms
	\param Onextn : the next-nearest-neighbour terms
	\param F : the FermionBase class where the operators are pulled from
	\param U : \f$U\f$
	\param V : \f$V\f$
	\param tPrime : \f$t'\f$
	*/
	static void set_operators (LocalTermsXd &Olocal, TightTermsXd &Otight, NextnTermsXd &Onextn, const FermionBase &F, double U, double V=0., double tPrime=0.);
	
	/**Calculates and returns \f$H^2\f$ on the fly.*/
	MpoQ<2> Hsq();
	
	/**single-site local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N_{\uparrow}\f$ and \f$N_{\downarrow}\f$. Used by default.*/
	static const std::array<qarray<2>,4> qssUD;
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N=N_{\uparrow}+N_{\downarrow}\f$ and \f$2M=N_{\uparrow}-N_{\downarrow}\f$. Used in combination with KondoModel.*/
	static const std::array<qarray<2>,4> qssNM;
	
	static const vector<qarray<2> > qloc (size_t N_legs);
	
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
	
	MpoQ<2> Auger (size_t locx, size_t locy=0);
	MpoQ<2> eta();
	MpoQ<2> Aps (size_t locx, size_t locy=0);
	MpoQ<2> c (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> d (size_t locx, size_t locy=0);
	MpoQ<2> n (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> Sz (size_t locx, size_t locy=0);
	MpoQ<2> SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2);
	MpoQ<2> triplon (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> quadruplon (size_t locx, size_t locy=0);
	
private:
	
	double U;
	double V = 0.;
	double tPrime = 0.;
	
	FermionBase F;
};

const std::array<qarray<2>,4> HubbardModel::qssUD {qarray<2>{0,0}, qarray<2>{1,0}, qarray<2>{0,1}, qarray<2>{1,1}};
const std::array<qarray<2>,4> HubbardModel::qssNM {qarray<2>{0,0}, qarray<2>{1,1}, qarray<2>{1,-1}, qarray<2>{2,0}};
const std::array<string,2>    HubbardModel::Nlabel{"N↑","N↓"};

const vector<qarray<2> > HubbardModel::
qloc (size_t N_legs)
{
	vector<qarray<2> > vout(pow(4,N_legs));
	
	NestedLoopIterator Nelly(N_legs,4);
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		vout[*Nelly] = HubbardModel::qssUD[Nelly(0)];
		
		for (int leg=1; leg<N_legs; ++leg)
		for (int q=0; q<2; ++q)
		{
			vout[*Nelly][q] += HubbardModel::qssUD[Nelly(leg)][q];
		}
	}
	
	return vout;
}

void HubbardModel::
set_operators (LocalTermsXd &Olocal, TightTermsXd &Otight, NextnTermsXd &Onextn, const FermionBase &F, double U, double V, double tPrime)
{
	Olocal.clear();
	Otight.clear();
	Onextn.clear();
	
	for (int leg=0; leg<F.orbitals(); ++leg)
	{
		Otight.push_back(make_tuple(-1., F.cdag(UP,leg), F.sign() * F.c(UP,leg)));
		Otight.push_back(make_tuple(-1., F.cdag(DN,leg), F.sign() * F.c(DN,leg)));
		Otight.push_back(make_tuple(+1., F.c(UP,leg),    F.sign() * F.cdag(UP,leg)));
		Otight.push_back(make_tuple(+1., F.c(DN,leg),    F.sign() * F.cdag(DN,leg)));
		
		if (V != 0.)
		{
			Otight.push_back(make_tuple(V, F.n(leg), F.n(leg)));
		}
	}
	
	if (tPrime != 0.)
	{
		Onextn.push_back(make_tuple(-tPrime, F.cdag(UP), F.sign() * F.c(UP),    F.sign()));
		Onextn.push_back(make_tuple(-tPrime, F.cdag(DN), F.sign() * F.c(DN),    F.sign()));
		Onextn.push_back(make_tuple(+tPrime, F.c(UP),    F.sign() * F.cdag(UP), F.sign()));
		Onextn.push_back(make_tuple(+tPrime, F.c(DN),    F.sign() * F.cdag(DN), F.sign()));
	}
	
	Olocal.push_back(make_tuple(1., F.HubbardHamiltonian(U,1.,V)));
}

HubbardModel::
HubbardModel (size_t Lx_input, double U_input, double V_input, double tPrime_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<2> (Lx_input, Ly_input, HubbardModel::qloc(Ly_input), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input), tPrime(tPrime_input)
{
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs);
	
	set_operators(Olocal,Otight,Onextn, F, U,V,tPrime);
	this->Daux = 2 + Otight.size() + 2*Onextn.size();
	
	SuperMatrix<double> G = ::Generator(Olocal,Otight,Onextn);
	
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
	SuperMatrix<double> G = ::Generator(Olocal,Otight,Onextn);
	MpoQ<2> Mout(N_sites, N_legs, tensor_product(G,G), HubbardModel::qloc(N_legs), {0,0}, HubbardModel::Nlabel, "HubbardModel H^2");
	return Mout;
}

MpoQ<2> HubbardModel::
Auger (size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "Auger(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {-1,-1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.c(UP,locy)*F.c(DN,locy));
	return Mout;
}

MpoQ<2> HubbardModel::
eta()
{
	stringstream ss;
	ss << "eta";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {-1,-1}, HubbardModel::Nlabel, ss.str());
	SparseMatrixXd etaloc = MatrixXd::Identity(F.dim(),F.dim()).sparseView();
	for (int ly=0; ly<N_legs; ++ly) {etaloc = etaloc * pow(-1.,ly) * F.c(UP,ly)*F.c(DN,ly);}
	Mout.setLocalSum(etaloc, true);
	return Mout;
}

MpoQ<2> HubbardModel::
Aps (size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "Aps(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {+1,+1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.cdag(DN,locy)*F.cdag(UP,locy));
	return Mout;
}

MpoQ<2> HubbardModel::
c (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "c(" << locx << "," << locy << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-1,0} : qdiff = {0,-1};
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = (sigma==UP)? F.sign_local(locy)*F.c(UP,locy) : F.sign_local(locy)*F.c(DN,locy);
	for (size_t l=locx+1; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(N_sites, N_legs, M, MpoQ<2>::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "c†(" << locx << "," << locy << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+1,0} : qdiff = {0,+1};
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = (sigma==UP)? F.sign_local(locy)*F.cdag(UP,locy) : F.sign_local(locy)*F.cdag(DN,locy);
	for (size_t l=locx+1; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(N_sites, N_legs, M, MpoQ<2>::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	// c(locx,UP)*c(locx,DN)
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
	// c(locx+1,UP|DN)
	M[locx+1].setMatrix(1,F.dim());
	M[locx+1](0,0) = (sigma==UP)? F.c(UP,locy) : F.c(DN,locy);
	for (size_t l=locx+2; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(N_sites, N_legs, M, MpoQ<2>::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	// c†(locx,DN)*c†(locx,UP)
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = F.cdag(DN,locy)*F.cdag(UP,locy);
	// c†(locx+1,UP|DN)
	M[locx+1].setMatrix(1,F.dim());
	M[locx+1](0,0) = (sigma==UP)? F.cdag(UP,locy) : F.cdag(DN,locy);
	for (size_t l=locx+2; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(N_sites, N_legs, M, MpoQ<2>::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
quadruplon (size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
	
	vector<SuperMatrix<double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0).setIdentity();
	}
	// c(loc,UP)*c(loc,DN)
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
	// c(loc+1,UP)*c(loc+1,DN)
	M[locx+1].setMatrix(1,F.dim());
	M[locx+1](0,0) = F.c(UP,locy)*F.c(DN,locy);
	for (size_t l=locx+2; l<N_sites; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0).setIdentity();
	}
	
	return MpoQ<2>(N_sites, N_legs, M, MpoQ<2>::qloc, {-2,-2}, HubbardModel::Nlabel, ss.str());
}

MpoQ<2> HubbardModel::
d (size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.d(locy));
	return Mout;
}

MpoQ<2> HubbardModel::
n (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "n(" << locx << "," << locy << ",σ=" << sigma << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	(sigma==UP)? Mout.setLocal(locx, F.n(UP,locy)):
	             Mout.setLocal(locx, F.n(DN,locy));
	return Mout;
}

MpoQ<2> HubbardModel::
Sz (size_t locx, size_t locy)
{
	assert(locx < N_sites and locy < N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.Sz(locy));
	return Mout;
}

MpoQ<2> HubbardModel::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" << "Sz(" << locx2 << "," << locy2 << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal({locx1,locx2}, {F.Sz(locy1),F.Sz(locy2)});
	return Mout;
}

}

#endif
