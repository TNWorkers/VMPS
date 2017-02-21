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
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.
\warning \f$J>0\f$ is antiferromagnetic*/
class HubbardModel : public MpoQ<2,double>
{
public:

	/**Does nothing.*/
	HubbardModel() : MpoQ(){};
	
	/**
	\param Lx_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$, next-nearest-neighbour (nnn) hopping. A minus sign in front of the hopping terms is assumed, so that \f$t^{\prime}>0\f$ is the usual choice.
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HubbardModel (size_t Lx_input, double U_input, double V_input=0., double tPrime_input=0., size_t Ly_input=1, bool CALC_SQUARE=true);
	
	/**Constructor for Hubbard rings and cylinders.
	\param BC_input : boundary condition, e.g. BC<RING>(10) for a 10-site ring; BC<HAIRSLIDE>(10) for the same, but folded into 5x2; BC<CYLINDER>(10,2) for a 10x2 cylinder
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	template<BC_CHOICE CHOICE> HubbardModel (BC<CHOICE> BC_input, double U_input, double tPerp=1., bool CALC_SQUARE=true);
	
	/**coupled plaquettes constructor*/
	template<BC_CHOICE CHOICE> HubbardModel (BC<CHOICE> BC_input, double U_input, double t1, double t2, double tPrime, bool CALC_SQUARE=true);
	
	/**Constructor with an external B-field.*/
	HubbardModel (size_t Lx_input, double U_input, vector<double> Bzvec, bool CALC_SQUARE=true);
	
	/**Constructor with onsite energies.*/
	HubbardModel (size_t Lx_input, vector<double> Uvec, vector<double> onsite, bool CALC_SQUARE=true);
	
	HubbardModel (size_t Lx_input, pair<double,double> t12, pair<double,double> eps12, double U_input, bool CALC_SQUARE=true);
	
	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. KondoModel.
	\param F : the FermionBase class where the operators are pulled from
	\param U : \f$U\f$
	\param V : \f$V\f$
	\param tPrime : \f$t'\f$
	\param tIntra : hopping within the rungs of ladder (or between legs)
	\param J : \f$J\f$
	\param Bz : \f$B_z\f$
	\param PERIODIC : if \p true, makes periodic boundary conditions in y-direction, i.e. a cylinder
	*/
	static HamiltonianTermsXd set_operators (const FermionBase &F, double U, double V=0., double tPrime=0., double tIntra=1., double J=0., double Bz=0., 
	                                         bool PERIODIC=false);
	
	static HamiltonianTermsXd set_operators (const FermionBase &F, vector<double> U, vector<double> onsite, MatrixXd tInter,
	                                         double V=0., double tPrime=0., double tIntra=1., double J=0., double Bz=0., bool PERIODIC=false);
	
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
	MpoQ<2,complex<double> > doublonPacket (complex<double> (*f)(int));
	MpoQ<2> eta();
	MpoQ<2> Aps (size_t locx, size_t locy=0);
	MpoQ<2> c (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2,complex<double> > electronPacket (complex<double> (*f)(int));
	MpoQ<2,complex<double> > holePacket (complex<double> (*f)(int));
	MpoQ<2> d (size_t locx, size_t locy=0);
	MpoQ<2> dtot();
	MpoQ<2> s (size_t locx, size_t locy=0);
	MpoQ<2> n (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> nn (SPIN_INDEX sigma1, size_t locx1, SPIN_INDEX sigma2, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<2> hh (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<2> Sz (size_t locx, size_t locy=0);
	MpoQ<2> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<2> SaSa (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t locy1=0, size_t locy2=0);
	MpoQ<2> triplon (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<2> quadruplon (size_t locx, size_t locy=0);
	
protected:
	
	double U;
	double V = 0.;
	double tPrime = 0.;
	
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
set_operators (const FermionBase &F, double U, double V, double tPrime, double tIntra, double J, double Bz, bool PERIODIC)
{
	vector<double> Uvec(F.orbitals());
	fill(Uvec.begin(), Uvec.end(), U);
	
	vector<double> onsite(F.orbitals());
	fill(onsite.begin(), onsite.end(), 0);
	
	return set_operators(F, Uvec,onsite, MatrixXd::Identity(F.orbitals(),F.orbitals()), V, tPrime, tIntra, J, Bz, PERIODIC);
}

HamiltonianTermsXd HubbardModel::
set_operators (const FermionBase &F, vector<double> Uvec, vector<double> onsite, MatrixXd tInter, double V, double tPrime, double tIntra, 
               double J, double Bz, bool PERIODIC)
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
	
////	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(Uvec,onsite,tIntra,V,J,Bz,PERIODIC)));
//	vector<double> Ushifted(Uvec.size());
//	vector<double> zeros(Uvec.size());
//	for (int l=0; l<Uvec.size(); ++l)
//	{
//		Ushifted[l] = Uvec[l]+onsite[l];
//		zeros[l] = 0;
//	}
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(Uvec,onsite,tIntra,V,J,Bz,PERIODIC)));
	
//	cout << F.HubbardHamiltonian(Uvec,onsite,tIntra,V,J,Bz,PERIODIC) << endl;
	
	return Terms;
}

HubbardModel::
HubbardModel (size_t Lx_input, double U_input, double V_input, double tPrime_input, size_t Ly_input, bool CALC_SQUARE)
:MpoQ<2> (Lx_input, Ly_input, HubbardModel::qloc(Ly_input,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input), tPrime(tPrime_input)
{
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	HamiltonianTermsXd Terms = set_operators(F, U,V,tPrime,1.);
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
HubbardModel (BC<CHOICE> BC_input, double U_input, double tPerp, bool CALC_SQUARE)
:MpoQ<2> (BC_input.Lx, BC_input.Ly, HubbardModel::qloc(BC_input.Ly,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(0.)
{
	stringstream ss;
	ss << "(U=" << U << ",t⟂=" << tPerp << "," << CHOICE << ")";
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
//			if      (CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,1.);} // t'=0, tIntra=1
//			else if (CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,1.,0.,0.,true);} // t'=0, tIntra=1, J=0, Bz=0
//			else if (CHOICE == CHAIN)     {Terms = set_operators(F, U,V,0.,0.,0.,0.,true);} // t'=0, tIntra=0, J=0, Bz=0
			if      (CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,tPerp);} // t'=0, tIntra=tPerp
			else if (CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,tPerp,0.,0.,true);} // t'=0, tIntra=tPerp, J=0, Bz=0
			else if (CHOICE == CHAIN)     {Terms = set_operators(F, U,V,0.,0.,0.,0.,true);} // t'=0, tIntra=0, J=0, Bz=0
			
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
//			if      (CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,1.);}
//			else if (CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,1.,0.,0.,true);}
//			else if (CHOICE == CHAIN)     {Terms = set_operators(F, U,V,0.,0.,0.,0.,true);}
			if      (CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,tPerp);}
			else if (CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,tPerp,0.,0.,true);}
			else if (CHOICE == CHAIN)     {Terms = set_operators(F, U,V,0.,0.,0.,0.,true);}
			
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
//			if      (CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,0.);}
//			else if (CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,1.,0.,0.,true);}
//			else if (CHOICE == CHAIN)     {Terms = set_operators(F, U,V,0.,0.,0.,0.,true);}
			if      (CHOICE == HAIRSLIDE) {Terms = set_operators(F, U,V,0.,0.);}
			else if (CHOICE == CYLINDER)  {Terms = set_operators(F, U,V,0.,tPerp,0.,0.,true);}
			else if (CHOICE == CHAIN)     {Terms = set_operators(F, U,V,0.,0.,0.,0.,true);}
			
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

template<BC_CHOICE CHOICE>
HubbardModel::
HubbardModel (BC<CHOICE> BC_input, double U_input, double t1, double t2, double tPrime, bool CALC_SQUARE)
:MpoQ<2> (BC_input.Lx, BC_input.Ly, HubbardModel::qloc(BC_input.Ly,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(0.)
{
	assert (CHOICE == HAIRSLIDE);
	assert (BC_input.Ly == 2);
	
	stringstream ss;
	ss << "(U=" << U << ",t1=" << t1 << ",t2=" << t2 << ",t'=" << tPrime << "," << CHOICE << ")";
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
			vector<double> Uvec(2); Uvec[0] = U; Uvec[1] = U;
			vector<double> onsite(2); onsite[0] = 0; onsite[1] = 0;
			MatrixXd tInter(2,2); tInter.setZero();
			if (l%2 == 0)
			{
				tInter(0,0) = (t1==0)? 1e-15:t1;
				tInter(1,1) = (t1==0)? 1e-15:t1;
				tInter(0,1) = (t2==0)? 1e-15:t2;
				tInter(1,0) = (t2==0)? 1e-15:t2;
			}
			else
			{
				tInter(0,0) = (tPrime==0)? 1e-15:tPrime;
				tInter(1,1) = (tPrime==0)? 1e-15:tPrime;
				tInter(0,1) = 1e-15;
				tInter(1,0) = 1e-15;
			}
			Terms = set_operators(F, Uvec,onsite,tInter,0.,0.,t1); // t'=0, tIntra=t1
			
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
			vector<double> Uvec(2); Uvec[0] = U; Uvec[1] = U;
			vector<double> onsite(2); onsite[0] = 0; onsite[1] = 0;
			MatrixXd tInter(2,2); tInter.setZero();
			if (l%2 == 0)
			{
				tInter(0,0) = (t1==0)? 1e-15:t1;
				tInter(1,1) = (t1==0)? 1e-15:t1;
				tInter(0,1) = (t2==0)? 1e-15:t2;
				tInter(1,0) = (t2==0)? 1e-15:t2;
			}
			else
			{
				tInter(0,0) = (tPrime==0)? 1e-15:tPrime;
				tInter(1,1) = (tPrime==0)? 1e-15:tPrime;
				tInter(0,1) = 1e-15;
				tInter(1,0) = 1e-15;
			}
			Terms = set_operators(F, Uvec,onsite,tInter,0.,0.,t1); // t'=0, tIntra=t1
			
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
			vector<double> Uvec(2); Uvec[0] = U; Uvec[1] = U;
			vector<double> onsite(2); onsite[0] = 0; onsite[1] = 0;
			MatrixXd tInter(2,2); tInter.setZero();
			if (l%2 == 0)
			{
				tInter(0,0) = (t1==0)? 1e-15:t1;
				tInter(1,1) = (t1==0)? 1e-15:t1;
				tInter(0,1) = (t2==0)? 1e-15:t2;
				tInter(1,0) = (t2==0)? 1e-15:t2;
			}
			else
			{
				tInter(0,0) = (tPrime==0)? 1e-15:tPrime;
				tInter(1,1) = (tPrime==0)? 1e-15:tPrime;
				tInter(0,1) = 1e-15;
				tInter(1,0) = 1e-15;
			}
			Terms = set_operators(F, Uvec,onsite,tInter,0.,0.,t1); // t'=0, tIntra=t1
			
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
HubbardModel (BC<RING> BC_input, double U_input, double V_input, bool CALC_SQUARE)
:MpoQ<2> (BC_input.Lx, 1, HubbardModel::qloc(1,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(V_input)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << "," << RING << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	HamiltonianTermsXd Terms = set_operators(F, U,V,0.,1.);
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

template<>
HubbardModel::
HubbardModel (BC<FLADDER> BC_input, double U_input, double tPerp, bool CALC_SQUARE)
:MpoQ<2> (BC_input.Lx, 1, HubbardModel::qloc(1,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(0.)
{
	stringstream ss;
	ss << "(U=" << U << ",t⟂=" << tPerp << "," << FLADDER << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		if (l==0)
		{
//			HamiltonianTermsXd Terms = set_operators(F, U,V,1.);
			vector<double> Uvec(1); Uvec[0] = U;
			vector<double> onsite(1); onsite[0] = 0;
			MatrixXd tInter(1,1);
			tInter.setConstant(tPerp);
			HamiltonianTermsXd Terms = set_operators(F, Uvec,onsite,tInter,V,1.);
			
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
//			HamiltonianTermsXd Terms = set_operators(F, U,V,1.);
			vector<double> Uvec(1); Uvec[0] = U;
			vector<double> onsite(1); onsite[0] = 0;
			MatrixXd tInter(1,1);
			(l%2==0)? tInter.setConstant(tPerp) : tInter.setConstant(1e-100);
			HamiltonianTermsXd Terms = set_operators(F, Uvec,onsite,tInter,V,1.);
			
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
			vector<double> Uvec(1); Uvec[0] = U;
			vector<double> onsite(1); onsite[0] = 0;
			MatrixXd tInter(1,1);
			(l%2==0)? tInter.setConstant(tPerp) : tInter.setConstant(1e-100);
//			HamiltonianTermsXd Terms = set_operators(F, Uvec,onsite,tInter,0.,1.);
			HamiltonianTermsXd Terms = set_operators(F, Uvec,onsite,tInter,V,1.);
			
			G[l].setMatrix(Daux,F.dim());
			
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

HubbardModel::
HubbardModel (size_t Lx_input, double U_input, vector<double> Bzvec, bool CALC_SQUARE)
:MpoQ<2> (Lx_input, 1, HubbardModel::qloc(1,!isfinite(U_input)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), V(0.), tPrime(0.)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ",Bz" << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		HamiltonianTermsXd Terms = set_operators(F, U,V,tPrime,1.,0.,Bzvec[l]);
		this->Daux = Terms.auxdim();
		
		if (l==0)
		{
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

HubbardModel::
HubbardModel (size_t Lx_input, pair<double,double> t12, pair<double,double> eps12, double U_input, bool CALC_SQUARE)
:MpoQ<2> (Lx_input, 1, HubbardModel::qloc(1,!isfinite(U)), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(U_input), tPrime(0)
{
	stringstream ss;
	ss << "(U=" << U << ",t_1=" << t12.first << ",t_2=" << t12.second << ")";
	this->label += ss.str();
	
	F = FermionBase(1,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		vector<double> Uvec(1); Uvec[0] = U;
		vector<double> onsite(1); onsite[0] = (l%2==0)? eps12.first : eps12.second;
		MatrixXd tInter(1,1); tInter(0,0) = (l%2==0)? t12.first : t12.second;
//		cout << "l=" << l << ", tInter=" << tInter << endl;
		HamiltonianTermsXd Terms = set_operators(F,Uvec,onsite,tInter);
		this->Daux = Terms.auxdim();
		
		if (l==0)
		{
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

HubbardModel::
HubbardModel (size_t Lx_input, vector<double> Uvec_input, vector<double> onsite_input, bool CALC_SQUARE)
:MpoQ<2> (Lx_input, 1, HubbardModel::qloc(1,!isfinite(Uvec_input[0])), {0,0}, HubbardModel::Nlabel, "HubbardModel"),
U(Uvec_input[0]), V(0.), tPrime(0.)
{
	stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ",Bz" << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		vector<double> Uvec(1); Uvec[0] = Uvec_input[l];
		vector<double> onsite(1); onsite[0] = onsite_input[l];
		HamiltonianTermsXd Terms = set_operators(F, Uvec,onsite,MatrixXd::Identity(F.orbitals(),F.orbitals()));
		this->Daux = Terms.auxdim();
		
		if (l==0)
		{
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

MpoQ<2,complex<double> > HubbardModel::
doublonPacket (complex<double> (*f)(int))
{
	stringstream ss;
	ss << "doublonPacket";
	MpoQ<2,complex<double> > Mout(N_sites, N_legs, MpoQ<2>::qloc, {-1,-1}, HubbardModel::Nlabel, ss.str());
	Mout.setLocalSum(F.c(UP)*F.c(DN), f);
	return Mout;
}

MpoQ<2> HubbardModel::
eta()
{
	assert(N_legs == 1);
	stringstream ss;
	ss << "eta";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {-1,-1}, HubbardModel::Nlabel, ss.str());
//	SparseMatrixXd etaloc = MatrixXd::Identity(F.dim(),F.dim()).sparseView();
//	for (int ly=0; ly<N_legs; ++ly) {etaloc = etaloc * pow(-1.,ly) * F.c(UP,ly)*F.c(DN,ly);}
	Mout.setLocalSum(F.c(UP)*F.c(DN), stagger);
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

MpoQ<2,complex<double> > HubbardModel::
electronPacket (complex<double> (*f)(int))
{
	assert(N_legs==1);
	stringstream ss;
	ss << "electronPacket";
	
	qarray<2> qdiff = {+1,0};
	
	vector<SuperMatrix<complex<double> > > M(N_sites);
	M[0].setRowVector(2,F.dim());
	M[0](0,0) = f(0) * F.cdag(UP);
	M[0](0,1).setIdentity();
	
	for (size_t l=1; l<N_sites-1; ++l)
	{
		M[l].setMatrix(2,F.dim());
		M[l](0,0) = complex<double>(1.,0.) * F.sign();
		M[l](1,0) = f(l) * F.cdag(UP);
		M[l](0,1).setZero();
		M[l](1,1).setIdentity();
	}
	
	M[N_sites-1].setColVector(2,F.dim());
	M[N_sites-1](0,0) = complex<double>(1.,0.) * F.sign();
	M[N_sites-1](1,0) = f(N_sites-1) * F.cdag(UP);
	
	return MpoQ<2,complex<double> >(N_sites, N_legs, M, MpoQ<2>::qloc, qdiff, HubbardModel::Nlabel, ss.str());
}

MpoQ<2,complex<double> > HubbardModel::
holePacket (complex<double> (*f)(int))
{
	assert(N_legs==1);
	stringstream ss;
	ss << "holePacket";
	
	qarray<2> qdiff = {-1,0};
	
	vector<SuperMatrix<complex<double> > > M(N_sites);
	M[0].setRowVector(2,F.dim());
	M[0](0,0) = f(0) * F.c(UP);
	M[0](0,1).setIdentity();
	
	for (size_t l=1; l<N_sites-1; ++l)
	{
		M[l].setMatrix(2,F.dim());
		M[l](0,0) = complex<double>(1.,0.) * F.sign();
		M[l](1,0) = f(l) * F.c(UP);
		M[l](0,1).setZero();
		M[l](1,1).setIdentity();
	}
	
	M[N_sites-1].setColVector(2,F.dim());
	M[N_sites-1](0,0) = complex<double>(1.,0.) * F.sign();
	M[N_sites-1](1,0) = f(N_sites-1) * F.c(UP);
	
	return MpoQ<2,complex<double> >(N_sites, N_legs, M, MpoQ<2>::qloc, qdiff, HubbardModel::Nlabel, ss.str());
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
dtot()
{
	stringstream ss;
	ss << "double_occ_total";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocalSum(F.d());
	return Mout;
}

MpoQ<2> HubbardModel::
s (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "single_occ(" << locx << "," << locy << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal(locx, F.n(UP,locy)+F.n(DN,locy)-2.*F.d(locy));
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
hh (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << "h(" << locx1 << "," << locy1 << ")h" << "(" << locx2 << "," << locy2 << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	SparseMatrixXd Id(F.dim(),F.dim()); Id.setIdentity();
	Mout.setLocal({locx1,locx2}, {F.d(locy1)-F.n(locy1)+Id,F.d(locy2)-F.n(locy2)+Id});
	return Mout;
}

MpoQ<2> HubbardModel::
nn (SPIN_INDEX sigma1, size_t locx1, SPIN_INDEX sigma2, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << "n(" << locx1 << "," << locy1 << ")n" << "(" << locx2 << "," << locy2 << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, {0,0}, HubbardModel::Nlabel, ss.str());
	Mout.setLocal({locx1,locx2}, {F.n(sigma1,locy1),F.n(sigma2,locy2)});
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

MpoQ<2> HubbardModel::
SaSa (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
	MpoQ<2> Mout(N_sites, N_legs, MpoQ<2>::qloc, F.Deltaq(SOP1)+F.Deltaq(SOP2), HubbardModel::Nlabel, ss.str());
	Mout.setLocal({locx1,locx2}, {F.Scomp(SOP1,locy1),F.Scomp(SOP2,locy2)});
	return Mout;
}

}

#endif
