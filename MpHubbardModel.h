#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "MpoQ.h"
#include "FermionBase.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
    - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
    + V \sum_{<ij>} n_{i} n_{j}
    + J \sum_{<ij>} \mathbf{S}_{i} \mathbf{S}_{j}
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.
\warning \f$J>0\f$ is antiferromagnetic*/
class HubbardModel : public MpoQ<2,double>
{
public:

	/**Does nothing.*/
	HubbardModel ():MpoQ() {};

	/**
	\param Lx_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$, next-nearest-neighbour (nnn) hopping. A minus sign in front of the hopping terms is assumed, so that \f$t^{\prime}>0\f$ is the usual choice.
	\param J_input: \f$J\f$
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HubbardModel (size_t Lx_input, double U_input, double V_input=0., double tPrime_input=0., double J_input=0., size_t Ly_input=1, bool CALC_SQUARE=true);
	
	/**Constructor for Hubbard rings and cylinders.
	\param BC_input : boundary condition, e.g. BC<RING>(10) for a 10-site ring (folded into 5x2), BC<CYLINDER>(10,2) for a 10x2 cylinder
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param J_input : \f$J\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	template<BC_CHOICE CHOICE> HubbardModel (BC<CHOICE> BC_input, double U_input, double V_input=0., double J_input=0., bool CALC_SQUARE=true);
	
	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. KondoModel.
	\param F : the FermionBase class where the operators are pulled from
	\param U : \f$U\f$
	\param V : \f$V\f$
	\param tPrime : \f$t'\f$
	\param tIntra : hopping within the rungs of ladder (or between legs)
	\param J : \f$J\f$
	\param PERIODIC : if \p true, makes periodic boundary conditions in y-direction, i.e. a cylinder
	*/
	static HamiltonianTermsXd set_operators (const FermionBase &F, double U, double V=0., double tPrime=0., double tIntra=1., double J=0., bool PERIODIC=false);
	static HamiltonianTermsXd set_operators (const FermionBase &F, vector<double> U, MatrixXd tInter,
	                                         double V=0., double tPrime=0., double tIntra=1., double J=0., bool PERIODIC=false);
	
	/**single-site local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N_{\uparrow}\f$ and \f$N_{\downarrow}\f$. Used by default.*/
	static const std::array<qarray<2>,4> qssNupNdn;
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N=N_{\uparrow}+N_{\downarrow}\f$ and \f$2M=N_{\uparrow}-N_{\downarrow}\f$. Used in combination with KondoModel.*/
	static const std::array<qarray<2>,4> qssNM;
	
	static const vector<qarray<2> > qloc (size_t N_legs, bool U_IS_INFINITE=false);
	
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
	
protected:
	
	double U;
	double V = 0.;
	double tPrime = 0.;
	double J = 0.;
	
	FermionBase F;
};

const std::array<qarray<2>,4> HubbardModel::qssNupNdn {qarray<2>{0,0}, qarray<2>{1,0}, qarray<2>{0,1},  qarray<2>{1,1}};
const std::array<qarray<2>,4> HubbardModel::qssNM {qarray<2>{0,0}, qarray<2>{1,1}, qarray<2>{1,-1}, qarray<2>{2,0}};
const std::array<string,2>    HubbardModel::Nlabel{"N↑","N↓"};

const vector<qarray<2> > HubbardModel::
qloc (size_t N_legs, bool U_IS_INFINITE)
{
	size_t locdim = (U_IS_INFINITE)? 3 : 4;
	vector<qarray<2> > vout(pow(locdim,N_legs));
	
	NestedLoopIterator Nelly(N_legs,locdim);
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		vout[*Nelly] = HubbardModel::qssNupNdn[Nelly(0)];
		
		for (int leg=1; leg<N_legs; ++leg)
		for (int q=0; q<2; ++q)
		{
			vout[*Nelly][q] += HubbardModel::qssNupNdn[Nelly(leg)][q];
		}
	}
	
	return vout;
}

HamiltonianTermsXd HubbardModel::
set_operators (const FermionBase &F, double U, double V, double tPrime, double tIntra, double J, bool PERIODIC)
{
//	HamiltonianTermsXd Terms;
//	
//	for (int leg=0; leg<F.orbitals(); ++leg)
//	{
//		Terms.tight.push_back(make_tuple(-1., F.cdag(UP,leg), F.sign() * F.c(UP,leg)));
//		Terms.tight.push_back(make_tuple(-1., F.cdag(DN,leg), F.sign() * F.c(DN,leg)));
//		Terms.tight.push_back(make_tuple(+1., F.c(UP,leg),    F.sign() * F.cdag(UP,leg)));
//		Terms.tight.push_back(make_tuple(+1., F.c(DN,leg),    F.sign() * F.cdag(DN,leg)));
//		
//		if (V != 0.)
//		{
//			Terms.tight.push_back(make_tuple(V, F.n(leg), F.n(leg)));
//		}
//	}
//	
//	if (tPrime != 0.)
//	{
//		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(UP), F.sign() * F.c(UP),    F.sign()));
//		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(DN), F.sign() * F.c(DN),    F.sign()));
//		Terms.nextn.push_back(make_tuple(+tPrime, F.c(UP),    F.sign() * F.cdag(UP), F.sign()));
//		Terms.nextn.push_back(make_tuple(+tPrime, F.c(DN),    F.sign() * F.cdag(DN), F.sign()));
//	}
//	
//	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(U,tIntra,V,PERIODIC)));
//	
//	return Terms;
	
	vector<double> Uvec(F.orbitals());
	fill(Uvec.begin(), Uvec.end(), U);
	
	return set_operators(F, Uvec, MatrixXd::Identity(F.orbitals(),F.orbitals()), V, tPrime, tIntra, J, PERIODIC);
}

HamiltonianTermsXd HubbardModel::
set_operators (const FermionBase &F, vector<double> Uvec, MatrixXd tInter, double V, double tPrime, double tIntra, double J, bool PERIODIC)
{
	assert(Uvec.size() == F.orbitals());
	HamiltonianTermsXd Terms;
	
	for (int legI=0; legI<F.orbitals(); ++legI)
	for (int legJ=0; legJ<F.orbitals(); ++legJ)
	{
		if (tInter(legI,legJ) != 0.)
		{
			Terms.tight.push_back(make_tuple(-tInter(legI,legJ), F.cdag(UP,legI), F.sign() * F.c(UP,legJ)));
			Terms.tight.push_back(make_tuple(-tInter(legI,legJ), F.cdag(DN,legI), F.sign() * F.c(DN,legJ)));
			Terms.tight.push_back(make_tuple(+tInter(legI,legJ), F.c(UP,legI),    F.sign() * F.cdag(UP,legJ)));
			Terms.tight.push_back(make_tuple(+tInter(legI,legJ), F.c(DN,legI),    F.sign() * F.cdag(DN,legJ)));
		}
		if (V != 0. and legI == legJ)
		{
			Terms.tight.push_back(make_tuple(V, F.n(legI), F.n(legJ)));
		}
		if (J != 0. and legI == legJ)
		{
			Terms.tight.push_back(make_tuple(0.5*J, F.Sp(legI), F.Sm(legJ)));
			Terms.tight.push_back(make_tuple(0.5*J, F.Sm(legI), F.Sp(legJ)));
			Terms.tight.push_back(make_tuple(J,     F.Sz(legI), F.Sz(legJ)));
		}
	}
	
	if (tPrime != 0.)
	{
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(UP), F.sign() * F.c(UP),    F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(DN), F.sign() * F.c(DN),    F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(UP),    F.sign() * F.cdag(UP), F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(DN),    F.sign() * F.cdag(DN), F.sign()));
	}
	
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(Uvec,tIntra,V,J,PERIODIC)));
	
	return Terms;
}
	
HubbardModel::
HubbardModel (size_t Lx_input, double U_input, double V_input, double tPrime_input, double J_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<2> (Lx_input, Ly_input, HubbardModel::qloc(Ly_input,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input), tPrime(tPrime_input), J(J_input)
{
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ",J=" << J << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	HamiltonianTermsXd Terms = set_operators(F, U,V,tPrime,1.,J);
	SuperMatrix<double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
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

template<BC_CHOICE CHOICE>
HubbardModel::
HubbardModel (BC<CHOICE> BC_input, double U_input, double V_input, double J_input, bool CALC_SQUARE)
:MpoQ<2> (BC_input.Lx, BC_input.Ly, HubbardModel::qloc(BC_input.Ly,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input), J(J_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",J=" << J << BC_input.CHOICE << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	HamiltonianTermsXd Terms;
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (l==0)
		{
			if      (BC_input.CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,1.,J);}
			else if (BC_input.CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,1.,J,true);}
			
			this->Daux = Terms.auxdim();
			G[l].setRowVector(Daux,F.dim());
			G[l] = Generator(Terms).row(Daux-1);
			
			if (CALC_SQUARE == true)
			{
				Gsq[l].setRowVector(Daux*Daux,F.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else if (l==this->N_sites-1)
		{
			if      (BC_input.CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,1.,J);}
			else if (BC_input.CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,1.,J,true);}
			
			this->Daux = Terms.auxdim();
			G[l].setColVector(Daux,F.dim());
			G[l] = Generator(Terms).col(0);
			
			if (CALC_SQUARE == true)
			{
				Gsq[l].setColVector(Daux*Daux,F.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else
		{
			if      (BC_input.CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,0.,J);}
			else if (BC_input.CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,1.,J,true);}
			
			this->Daux = Terms.auxdim();
			G[l].setMatrix(Daux,F.dim());
			G[l] = Generator(Terms);
			
			if (CALC_SQUARE == true)
			{
				Gsq[l].setMatrix(Daux*Daux,F.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
	}
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(Gsq, this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

template<>
HubbardModel::
HubbardModel (BC<RING> BC_input, double U_input, double V_input, double J_input, bool CALC_SQUARE)
:MpoQ<2> (BC_input.Lx, 1, HubbardModel::qloc(1,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input), J(J_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << "," << BC_input.CHOICE << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	HamiltonianTermsXd Terms = set_operators(F, U,V,0.,1.,J);
	this->Daux = Terms.auxdim()+Terms.tight.size();
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (l==0)
		{
			G[l].setRowVector(Daux,F.dim());
			
			SuperMatrix<double> G_PBC;
			G_PBC.setRowVector(Terms.tight.size(),F.dim());
			for (int i=0; i<Terms.tight.size(); ++i)
			{
				G_PBC(0,i) = get<0>(Terms.tight[i]) * get<1>(Terms.tight[i]);
			}
			
			G[l] = directSum(Generator(Terms).row(Terms.auxdim()-1),G_PBC);
			
			if (CALC_SQUARE == true)
			{
				Gsq[l].setRowVector(Daux*Daux,F.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else if (l==this->N_sites-1)
		{
			G[l].setColVector(Daux,F.dim());
			
			SuperMatrix<double> G_PBC;
			G_PBC.setColVector(Terms.tight.size(),F.dim());
			for (int i=0; i<Terms.tight.size(); ++i)
			{
				G_PBC(i,0) = get<2>(Terms.tight[i]);
			}
			
			G[l] = directSum(Generator(Terms).col(0),G_PBC);
			
			if (CALC_SQUARE == true)
			{
				Gsq[l].setColVector(Daux*Daux,F.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else
		{
			G[l].setMatrix(Daux,F.dim());
			
			SuperMatrix<double> G_PBC;
			G_PBC.setMatrix(Terms.tight.size(),F.dim());
			for (int i=0; i<Terms.tight.size(); ++i)
			{
				G_PBC(i,i) = F.sign();
			}
			
			G[l] = directSum(Generator(Terms),G_PBC);
			
			if (CALC_SQUARE == true)
			{
				Gsq[l].setMatrix(Daux*Daux,F.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
	}
	
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(Gsq, this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

MpoQ<2> HubbardModel::
Auger (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
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
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Aps(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {+1,+1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.cdag(DN,locy)*F.cdag(UP,locy));
	return Mout;
}

MpoQ<2> HubbardModel::
c (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
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
	assert(locx<N_sites and locy<N_legs);
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
	assert(locx<N_sites and locy<N_legs);
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
	assert(locx<N_sites and locy<N_legs);
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
	assert(locx<N_sites and locy<N_legs);
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
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.d(locy));
	return Mout;
}

MpoQ<2> HubbardModel::
n (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "n(" << locx << "," << locy << ",σ=" << sigma << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.n(sigma,locy));
	return Mout;
}

MpoQ<2> HubbardModel::
Sz (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
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
