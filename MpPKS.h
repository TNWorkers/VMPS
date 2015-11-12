#include "MpHubbardModel.h"
#include "MpHeisenbergModel.h"
#include "SuperMatrix.h"

namespace VMPS
{
/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z
\f$.
\note \f$J<0\f$ : antiferromagnetic
\note The local magnetic fields act on the impurities only.
\note The parameters \f$t\f$ and \f$t^{\prime}\f$ are for pinning the PKS phase.*/
class PKS : public KondoModel
{
public:

	/**Does nothing.*/
	PKS () {};

	/**Constructs a Kondo Lattice Model with partial Kondo screening (PKS) geometry.
	\param L_input : chain length
	\param Bzval_input : \f$B^z_i\f$ (site dependent magnetic field acting on the impurities)
	\param J_input : \f$J\f$
	\param t_input : First hopping \f$t\f$. \f$t>0\f$ is the common sign.
	\param tPrime_input : Second hopping \f$t^{\prime}\f$ (\f$t^{\prime}>0\f$ is common sign).
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	PKS (size_t L_input, double J_input=-1., double t_input=-1., double tPrime_input=-1.,
	            vector<double> Bzval_input={}, double U_input=0., bool CALC_SQUARE=true, size_t D_input=2);

	typedef DmrgSolverQ<2,PKS> Solver;

	void set_operators(double J, double t, double tPrime, double Bzval, double U, size_t D);

private:
	double J=-1., Bz=0., t=-1., tPrime=0., U=0.;
	size_t D=2;
	
	vector<double> Bzval;
	vector<size_t> imploc;

};

PKS::
PKS (size_t L_input, double J_input, double t_input, double tPrime_input, vector<double> Bzval_input, double U_input, bool CALC_SQUARE, size_t D_input)
	:KondoModel(L_input,"KondoModel (PKS geometry) ",D_input ),
	J(J_input), t(t_input), tPrime(tPrime_input), U(U_input), D(D_input)
{	
	// initialize member variable imploc
	this->imploc.resize(L_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);
	
	// if Bzval_input empty, set it to zero
	if (Bzval_input.size() == 0)
	{
		Bzval.assign(this->N_sites,0.);
	}
	else
	{
		assert(this->N_sites == Bzval_input.size() and "Impurities and B-fields do not match!");
		Bzval = Bzval_input;
	}
	
	// label
	stringstream ss;
	ss << "(S=" << frac(D-1,2) << ",J=" << J << ",t=" << t << ",tPrime=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();

	//construct PKS topology with t-tPrime structure
	std::vector<double> tVec(this->N_sites);
	std::vector<double> tPrimeVec(this->N_sites);

	for (size_t l=0; l<this->N_sites; ++l)
	{
		if(l == 0) {tVec[l]=t; tPrimeVec[l]=tPrime;}
		if(l == 1) {tVec[l]=t; tPrimeVec[l]=tPrime;}
		if(((l+1) % 3 == 0) and (l > 1)) {tVec[l]=tPrime; tPrimeVec[l]=tPrime;}
		if(((l+1) % 3 == 1) and (l > 1)) {tVec[l]=tPrime; tPrimeVec[l]=t;}
		if(((l+1) % 3 == 2) and (l > 1)) {tVec[l]=t; tPrimeVec[l]=tPrime;}
	}

	// create the SuperMatrices
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
			G[l].setRowVector(14,8);
			this->set_operators(J,tVec[l],tPrimeVec[l],Bzval[l],U,D);
			G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn).row(13);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setRowVector(14*14,8);
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else if (l==this->N_sites-1)
		{
			G[l].setColVector(14,8);
			this->set_operators(J,tVec[l],tPrimeVec[l],Bzval[l],U,D);
			G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn).col(0);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setColVector(14*14,8);
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else
		{
			G[l].setMatrix(14,8);
			this->set_operators(J,tVec[l],tPrimeVec[l],Bzval[l],U,D);
			G[l] = ::Generator(this->Olocal,this->Otight,this->Onextn);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setMatrix(14*14,8);
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

	//clear old values of operators
	this->Olocal.resize(0);
	this->Otight.resize(0);
	this->Onextn.resize(0);
}

void PKS::
set_operators (double J, double t, double tPrime, double Bz, double U, size_t D)
{
	Eigen::MatrixXd Id4(4,4); Id4.setIdentity();
	Eigen::MatrixXd IdSpins(D,D); IdSpins.setIdentity();

	//clear old values of operators
	this->Olocal.resize(0);
	this->Otight.resize(0);
	this->Onextn.resize(0);
	
	//set local interaction Olocal
	this->Olocal.push_back(std::make_tuple(-0.5*J, kroneckerProduct(SpinBase::Scomp(SP,D), FermionBase::Sp.transpose())));
	this->Olocal.push_back(std::make_tuple(-0.5*J, kroneckerProduct(SpinBase::Scomp(SM,D), FermionBase::Sp)));
	this->Olocal.push_back(std::make_tuple(-J, kroneckerProduct(SpinBase::Scomp(SZ,D), FermionBase::Sz)));
	if (Bz != 0.)
	{
		this->Olocal.push_back(std::make_tuple(-Bz, kroneckerProduct(SpinBase::Scomp(SZ,D), Id4)));
	}

	if (Bz != 0.)
	{
		this->Olocal.push_back(std::make_tuple(U, kroneckerProduct(IdSpins, FermionBase::d)));
	}

	//set nearest neighbour term Otight
	this->Otight.push_back(std::make_tuple(-t,kroneckerProduct(IdSpins, FermionBase::cUP.transpose()), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP)));
	this->Otight.push_back(std::make_tuple(-t,kroneckerProduct(IdSpins, FermionBase::cDN.transpose()), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN)));
	this->Otight.push_back(std::make_tuple(t,kroneckerProduct(IdSpins, FermionBase::cUP), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP.transpose())));
	this->Otight.push_back(std::make_tuple(t,kroneckerProduct(IdSpins, FermionBase::cDN), kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN.transpose())));

	//set next nearest neighbour term Onextn
	this->Onextn.push_back(std::make_tuple(-tPrime,
										   kroneckerProduct(IdSpins, FermionBase::cUP.transpose()),
										   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP),
										   kroneckerProduct(IdSpins, FermionBase::fsign)));
	this->Onextn.push_back(std::make_tuple(-tPrime,
										   kroneckerProduct(IdSpins, FermionBase::cDN.transpose()),
										   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN),
										   kroneckerProduct(IdSpins, FermionBase::fsign)));
	this->Onextn.push_back(std::make_tuple(tPrime,
										   kroneckerProduct(IdSpins, FermionBase::cUP),
										   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cUP.transpose()),
										   kroneckerProduct(IdSpins, FermionBase::fsign)));
	this->Onextn.push_back(std::make_tuple(tPrime,
										   kroneckerProduct(IdSpins, FermionBase::cDN),
										   kroneckerProduct(IdSpins, FermionBase::fsign * FermionBase::cDN.transpose()),
										   kroneckerProduct(IdSpins, FermionBase::fsign)));
}

}

