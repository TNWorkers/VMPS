#ifndef STRAWBERRY_KONDONECKLACEMODEL
#define STRAWBERRY_KONDONECKLACEMODEL

#include "MpHeisenbergModel.h"

namespace VMPS
{

class KondoNecklaceModel : public MpoQ<1,double>
{
public:
	
	KondoNecklaceModel (size_t Lx_input, double J_input, double txy_input=1., double tz_input=1., 
	                    double Bz_input=0., size_t D_input=2, bool CALC_SQUARE=true);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<1,double>                    StateXd;
	typedef MpsQ<1,complex<double> >          StateXcd;
	typedef DmrgSolverQ<1,KondoNecklaceModel> Solver;
	///@}
	
	// spin of the impurity
	MpoQ<1> Simp (SPINOP_LABEL Sa);
	
private:
	
	double J = -1.;
	double txy = 1.;
	double tz  = 1.;
	size_t D = 2;
	double Bz = 0.;
	
	SpinBase Bsub;
	SpinBase Bimp;
};

KondoNecklaceModel::
KondoNecklaceModel (size_t Lx_input, double J_input, double txy_input, double tz_input, double Bz_input, size_t D_input, bool CALC_SQUARE)
:MpoQ<1>(), J(J_input), txy(txy_input), tz(tz_input), Bz(Bz_input), D(D_input)
{
	Bimp = SpinBase(2,D);
	Bsub = SpinBase(1,D);
	
	this->N_sites = Lx_input;
	this->N_legs = 1;
	this->Qtot = {0};
	this->qlabel = HeisenbergModel::maglabel;
	this->label = "KondoNecklaceModel";
	stringstream ss;
	ss << "(J=" << J << "," << "txy=" << txy << ",tz=" << tz << ")";
	this->label += ss.str();
	this->format = halve;
	this->qloc.resize(this->N_sites);
	
	// create the SuperMatrices
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	MatrixXd T(2,2); T.setZero(); T(0,0) = 1.;
	Vector2d Bzvec; Bzvec << 0, Bz;
	Vector2d Bxvec; Bxvec << 0, 0;
	HamiltonianTermsXd TermsImp = HeisenbergModel::set_operators(Bimp, txy*T,tz*T,Bzvec,Bxvec,0.,J,J);
	HamiltonianTermsXd TermsSub = HeisenbergModel::set_operators(Bsub, txy,tz);
	this->Daux = TermsImp.auxdim();
	
	// first site
	G[0].setRowVector(this->Daux,4);
	G[0] = Generator(TermsImp).row(this->Daux-1);
	if (CALC_SQUARE == true)
	{
		Gsq[0].setRowVector(this->Daux*this->Daux,4);
		Gsq[0] = tensor_product(G[0],G[0]);
	}
	this->qloc[0] = HeisenbergModel::qloc(2,D);
	
	for (size_t l=1; l<this->N_sites-1; ++l)
	{
		G[l].setMatrix(this->Daux,2);
		G[l] = Generator(TermsSub);
		if (CALC_SQUARE == true)
		{
			Gsq[l].setMatrix(this->Daux*this->Daux,2);
			Gsq[l] = tensor_product(G[l],G[l]);
		}
		this->qloc[l] = HeisenbergModel::qloc(1,D);
	}
	
	// last site
	size_t last = this->N_sites-1;
	G[last].setColVector(this->Daux,2);
	G[last] = Generator(TermsSub).col(0);
	if (CALC_SQUARE == true)
	{
		Gsq[last].setColVector(this->Daux*this->Daux,2);
		Gsq[last] = tensor_product(G[last],G[last]);
	}
	this->qloc[last] = HeisenbergModel::qloc(1,D);
	
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

MpoQ<1> KondoNecklaceModel::
Simp (SPINOP_LABEL Sa)
{
	stringstream ss;
	ss << Sa;
	MpoQ<1> Mout(N_sites, 1, MpoQ<1>::qloc, {0}, HeisenbergModel::maglabel, ss.str(), halve);
	Mout.setLocal(0, Bimp.Scomp(Sa,1));
	return Mout;
}

}

#endif
