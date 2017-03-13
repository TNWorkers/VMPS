#ifndef VANILLA_DZYALOSHINSKIMORIYAMODELXZ
#define VANILLA_DZYALOSHINSKIMORIYAMODELXZ

#include "MpHeisenbergModel.h"

namespace VMPS
{

class DzyaloshinskyMoriyaModelXZ : public MpoQ<0,double>
{
public:
	
	DzyaloshinskyMoriyaModelXZ (size_t Lx_input, double J_input, vector<double> Bz_input, double K_input, double DMy_input, 
	                          double Jprime_input=0, double DMyprime_input=0, 
	                          size_t D_input=2, bool CALC_SQUARE=true);
	
	static HamiltonianTermsXd set_operators (const SpinBase &S, double J, double Bz, double K, double DMy, double Jprime=0, double DMyprime=0);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<0,double>                                 StateXd;
	typedef DmrgSolverQ<0,DzyaloshinskyMoriyaModelXZ,double> Solver;
	///@}
	
	MpoQ<0> Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	
private:
	
	double J = -1.;
	double Jprime = 0;
	size_t D = 2;
	vector<double> Bz;
	vector<double> Bx;
	double DMy;
	double DMyprime;
	double K = 0;
	
	SpinBase B;
};

HamiltonianTermsXd DzyaloshinskyMoriyaModelXZ::
set_operators (const SpinBase &S, double J, double Bz, double K, double DMy, double Jprime, double DMyprime)
{
	assert(S.orbitals() == 1);
	
	HamiltonianTermsXd Terms;
	
	SparseMatrixXd Sp = S.Scomp(SP);
	SparseMatrixXd Sm = S.Scomp(SM);
	SparseMatrixXd Sx = S.Scomp(SX);
	SparseMatrixXd Sz = S.Scomp(SZ);
	SparseMatrixXd Id = MatrixXd::Identity(S.dim(),S.dim()).sparseView();
	
	// Heisenberg part
	if (J!=0.)
	{
		Terms.tight.push_back(make_tuple(-0.5*J, Sp, Sm));
		Terms.tight.push_back(make_tuple(-0.5*J, Sm, Sp));
		Terms.tight.push_back(make_tuple(-J, Sz, Sz));
	}
	
	if (Jprime!=0.)
	{
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, Sp, Sm, Id));
		Terms.nextn.push_back(make_tuple(-0.5*Jprime, Sm, Sp, Id));
		Terms.nextn.push_back(make_tuple(-Jprime, Sz, Sz, Id));
	}
	
	// Dzyaloshinsky-Moriya part
	Terms.tight.push_back(make_tuple(+DMy, Sx, Sz));
	Terms.tight.push_back(make_tuple(-DMy, Sz, Sx));
	
	if (DMyprime!=0.)
	{
		Terms.nextn.push_back(make_tuple(+DMyprime, Sx, Sz, Id));
		Terms.nextn.push_back(make_tuple(-DMyprime, Sz, Sx, Id));
	}
	
	// local part
	Terms.local.push_back(make_tuple(1., S.HeisenbergHamiltonian(J,J,Bz,0,K)));
	
	return Terms;
}

DzyaloshinskyMoriyaModelXZ::
DzyaloshinskyMoriyaModelXZ (size_t Lx_input, double J_input, vector<double> Bz_input, double K_input, double DMy_input,  double Jprime_input, double DMyprime_input, size_t D_input, bool CALC_SQUARE)
:MpoQ<0,double>(), J(J_input), Bz(Bz_input), K(K_input), DMy(DMy_input), Jprime(Jprime_input), DMyprime(DMyprime_input), D(D_input)
{
	B = SpinBase(1,D);
	
	this->N_sites = Lx_input;
	this->N_legs = 1;
	this->Qtot = {};
	this->qlabel = labeldummy;
	this->label = "DzyaloshinskyMoriyaModelXZ";
	stringstream ss;
	ss << "(J=" << J << ",";
	if (Jprime!=0.)
	{
		ss << "J'=" << Jprime << ",";
	}
	ss << "DMy=" << DMy << ",";
	if (DMyprime != 0.)
	{
		ss << "DMy'=" << DMyprime << ",";
	}
	ss << "K=" << K << ",";
	ss << "Bz[0]=" << Bz[0] << ",";
	ss << "D=" << D;
	ss << ")";
	this->label += ss.str();
	this->format = noFormat;
	this->qloc.resize(this->N_sites);
	
	// create the SuperMatrices
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	HamiltonianTermsXd Terms;
	vector<qarray<0> > qlocDdummy;
	for (int i=0; i<D; ++i)
	{
		qarray<0> qdummy = qarray<0>{};
		qlocDdummy.push_back(qdummy);
	}
	
	// first site
	Terms = set_operators(B, J,Bz[0],K,DMy,Jprime,DMyprime);
	this->Daux = Terms.auxdim();
	G[0].setRowVector(this->Daux,D);
	G[0] = Generator(Terms).row(this->Daux-1);
	if (CALC_SQUARE == true)
	{
		Gsq[0].setRowVector(this->Daux*this->Daux,D);
		Gsq[0] = tensor_product(G[0],G[0]);
	}
	this->qloc[0] = qlocDdummy;
	
	for (size_t l=1; l<this->N_sites-1; ++l)
	{
		Terms = set_operators(B, J,Bz[l],K,DMy,Jprime,DMyprime);
		G[l].setMatrix(this->Daux,D);
		G[l] = Generator(Terms);
		if (CALC_SQUARE == true)
		{
			Gsq[l].setMatrix(this->Daux*this->Daux,D);
			Gsq[l] = tensor_product(G[l],G[l]);
		}
		this->qloc[l] = qlocDdummy;
	}
	
	// last site
	size_t last = this->N_sites-1;
	Terms = set_operators(B, J,Bz[last],K,DMy,Jprime,DMyprime);
	G[last].setColVector(this->Daux,D);
	G[last] = Generator(Terms).col(0);
	
	if (CALC_SQUARE == true)
	{
		Gsq[last].setColVector(this->Daux*this->Daux,D);
		Gsq[last] = tensor_product(G[last],G[last]);
	}
	this->qloc[last] = qlocDdummy;
	
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

MpoQ<0> DzyaloshinskyMoriyaModelXZ::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << Sa << "(" << locx << "," << locy << ")";
	MpoQ<0> Mout(N_sites, N_legs, MpoQ<0,double>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(locx, B.Scomp(Sa,locy));
	return Mout;
}

}

#endif
