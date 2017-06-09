#ifndef VANILLA_DZYALOSHINSKIMORIYAMODEL
#define VANILLA_DZYALOSHINSKIMORIYAMODEL

#include "symmetry/U0.h"
#include "MpHeisenbergModel.h"

namespace VMPS
{

class DzyaloshinskyMoriyaModel : public MpoQ<Sym::U0,complex<double> >
{
typedef Sym::U0 Symmetry;

public:
	
	DzyaloshinskyMoriyaModel (size_t Lx_input, double J_input, vector<double> Bz_input, double K_input, const std::array<double,3> &DM_input, 
	                          double Jprime_input=0., const std::array<double,3> &DMprime_input={0.,0.,0.}, 
	                          size_t D_input=2, bool CALC_SQUARE=true);
	
	static HamiltonianTermsXcd set_operators (const SpinBase &S, double J, double B, double K, const std::array<double,3> &DM, 
	                                          double Jprime=0., const std::array<double,3> &DMprime={0.,0.,0.});
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,complex<double> >                                 StateXcd;
	typedef DmrgSolverQ<Symmetry,DzyaloshinskyMoriyaModel,complex<double> > Solver;
	///@}
	
	MpoQ<Symmetry>                  Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	MpoQ<Symmetry,complex<double> > Sy    (size_t locx, size_t locy=0) const;
	
private:
	
	double J = -1.;
	double Jprime = 0.;
	size_t D = 2;
	vector<double> Bz;
	vector<double> Bx;
	std::array<double,3> DM;
	std::array<double,3> DMprime;
	double K = 0.;
	
	SpinBase B;
};

HamiltonianTermsXcd DzyaloshinskyMoriyaModel::
set_operators (const SpinBase &S, double J, double Bz, double K, const std::array<double,3> &DM, double Jprime, const std::array<double,3> &DMprime)
{
	assert(S.orbitals() == 1);
	
	HamiltonianTermsXcd Terms;
	
	SparseMatrixXcd Sp = S.Scomp(SP).cast<complex<double> >();
	SparseMatrixXcd Sm = S.Scomp(SM).cast<complex<double> >();
	SparseMatrixXcd Sx = S.Scomp(SX).cast<complex<double> >();
	SparseMatrixXcd Sy = -1.i*S.Scomp(iSY);
	SparseMatrixXcd Sz = S.Scomp(SZ).cast<complex<double> >();
	SparseMatrixXcd Id = MatrixXcd::Identity(S.dim(),S.dim()).sparseView();
	
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
	double DMx = DM[0];
	double DMy = DM[1];
	double DMz = DM[2];
	
	if (DMz!=0. or DMx!=0.)
	{
		Terms.tight.push_back(make_tuple(1., Sy, DMz*Sx-DMx*Sz));
		Terms.tight.push_back(make_tuple(1., DMx*Sz-DMz*Sx, Sy));
	}
	if (DMy!=0.)
	{
		Terms.tight.push_back(make_tuple(+DMy, Sx, Sz));
		Terms.tight.push_back(make_tuple(-DMy, Sz, Sx));
	}
	
	double DMxprime = DMprime[0];
	double DMyprime = DMprime[1];
	double DMzprime = DMprime[2];
	if (DMzprime!=0. or DMxprime!=0.)
	{
		Terms.nextn.push_back(make_tuple(1., Sy, DMzprime*Sx-DMxprime*Sz, Id));
		Terms.nextn.push_back(make_tuple(1., DMxprime*Sz-DMzprime*Sx, Sy, Id));
	}
	if (DMyprime!=0.)
	{
		Terms.nextn.push_back(make_tuple(+DMyprime, Sx, Sz, Id));
		Terms.nextn.push_back(make_tuple(-DMyprime, Sz, Sx, Id));
	}
	
	// local part
	SparseMatrixXcd Hloc = S.HeisenbergHamiltonian(J,J,Bz,0,K).cast<complex<double> >();
	Terms.local.push_back(make_tuple(1., Hloc));
	
	return Terms;
}

DzyaloshinskyMoriyaModel::
DzyaloshinskyMoriyaModel (size_t Lx_input, double J_input, vector<double> Bz_input, double K_input, const std::array<double,3> &DM_input, 
                          double Jprime_input, const std::array<double,3> &DMprime_input, 
                          size_t D_input, bool CALC_SQUARE)
:MpoQ<Symmetry,complex<double> >(), J(J_input), Bz(Bz_input), K(K_input), DM(DM_input), Jprime(Jprime_input), DMprime(DMprime_input), D(D_input)
{
	B = SpinBase(1,D);
	
	this->N_sites = Lx_input;
	this->N_legs = 1;
	this->Qtot = {};
	this->qlabel = labeldummy;
	this->label = "DzyaloshinskyMoriyaModel";
	stringstream ss;
	ss << "(J=" << J << ",";
	if (Jprime!=0.)
	{
		ss << "J'=" << Jprime << ",";
	}
	ss << "DM=[" << DM[0] << "," << DM[1] << "," << DM[2] << "],";
	if (DMprime[0]!=0. or DMprime[1]!=0. or DMprime[2]!=0.)
	{
		ss << "DM'=[" << DMprime[0] << "," << DMprime[1] << "," << DMprime[2] << "],";
	}
	ss << "K=" << K << ",";
	ss << "Bz[0]=" << Bz[0] << ",";
	ss << "D=" << D;
	ss << ")";
	this->label += ss.str();
	this->format = noFormat;
	this->qloc.resize(this->N_sites);
	
	// create the SuperMatrices
	vector<SuperMatrix<complex<double> > > G(this->N_sites);
	vector<SuperMatrix<complex<double> > > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	HamiltonianTermsXcd Terms;
	vector<qarray<0> > qlocDdummy;
	for (int i=0; i<D; ++i)
	{
		qarray<0> qdummy = qarray<0>{};
		qlocDdummy.push_back(qdummy);
	}
	
	// first site
	Terms = set_operators(B, J,Bz[0],K,DM,Jprime,DMprime);
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
		Terms = set_operators(B, J,Bz[l],K,DM,Jprime,DMprime);
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
	Terms = set_operators(B, J,Bz[last],K,DM,Jprime,DMprime);
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

MpoQ<Sym::U1> DzyaloshinskyMoriyaModel::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << Sa << "(" << locx << "," << locy << ")";
	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry,complex<double> >::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(locx, B.Scomp(Sa,locy));
	return Mout;
}

MpoQ<Sym::U1,complex<double> > DzyaloshinskyMoriyaModel::
Sy (size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sy(" << locx << "," << locy << ")";
	MpoQ<Symmetry,complex<double> > Mout(N_sites, N_legs, MpoQ<Symmetry,complex<double> >::qloc, {}, labeldummy, ss.str());
	SparseMatrixXcd SyOp = -1.i*B.Scomp(iSY,locy);
	Mout.setLocal(locx,SyOp);
	return Mout;
}

}

#endif
