#include "MpHubbardModel.h"
#include "MpHeisenbergModel.h"
#include "SuperMatrix.h"

namespace VMPS
{
/**MPO representation of 
\f$
H = -t \sum_{(i,j)\in \mathcal{T},\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{(i,j)\in \mathcal{T^{\prime}},\sigma} c^\dagger_{i\sigma}c_{j\sigma} - J \sum_{i} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i} B_i^z S_i^z + \frac{U}{2} \sum_{i,\sigma}n_{i\sigma}n_{i-\sigma}
\f$.
\note \f$J<0\f$ : antiferromagnetic
\note The local magnetic fields act on the impurities only.
\note \f$\mathcal{T}\f$ and \f$\mathcal{T^{\prime}}\f$ defines the PKS geometry.
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

	PKS (size_t Lx_input, size_t Ly_input=2, double J_input=-1., double t_input=-1., double tPrime_input=-1.,
		 vector<double> Bzval_input={}, double U_input=0., bool CALC_SQUARE=false, size_t D_input=2);

	typedef DmrgSolverQ<2,PKS> Solver;

private:
	double J=-1., Bz=0., t=-1., tPrime=0., U=0.;
	size_t D=2; size_t Ly=1;
	
	vector<double> Bzval;
	vector<size_t> imploc;

};

PKS::
PKS (size_t L_input, double J_input, double t_input, double tPrime_input, vector<double> Bzval_input, double U_input, bool CALC_SQUARE, size_t D_input)
	:KondoModel(L_input,"KondoModel (PKS geometry) ", 1, D_input ),
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
	std::vector<Eigen::MatrixXd> tVec(this->N_sites);
	std::vector<double> tPrimeVec(this->N_sites);
	Eigen::MatrixXd Zero(1,1); Zero.setZero();

	for (size_t l=0; l<this->N_sites; ++l)
	{
		tVec[l].resize(1,1);
		if((l % 3 == 0)) {tVec[l](0,0)=tPrime; tPrimeVec[l]=t;}
		if((l % 3 == 1)) {tVec[l](0,0)=t; tPrimeVec[l]=tPrime;}
		if((l % 3 == 2)) {tVec[l](0,0)=t; tPrimeVec[l]=t;}
	}

	F = FermionBase(1);
	S = SpinBase(1,D);

	for (size_t l=0; l<this->N_sites; ++l)
	{
		MpoQ<2>::qloc[l].resize(F.dim()*S.dim());
		for (size_t j=0; j<S.dim(); j++)
			for (size_t i=0; i<F.dim(); i++)
			{
				MpoQ<2>::qloc[l][i+F.dim()*j] = ::qvacuum<2>();
				MpoQ<2>::qloc[l][i+F.dim()*j] = F.qNums(i);
				MpoQ<2>::qloc[l][i+F.dim()*j][1] += S.qNums(j)[0];
			}
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
			KondoModel::set_operators(Olocal,Otight,Onextn,F,S,J,Bzval[l],tVec[l],Zero,0.,tPrimeVec[l],U);
			this->Daux = 2 + Otight.size() + 2*Onextn.size();
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
			KondoModel::set_operators(Olocal,Otight,Onextn,F,S,J,Bzval[l],tVec[l],Zero,0.,tPrimeVec[l],U);			
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
			KondoModel::set_operators(Olocal,Otight,Onextn,F,S,J,Bzval[l],tVec[l],Zero,0.,tPrimeVec[l],U);			
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

PKS::
PKS (size_t Lx_input, size_t Ly_input, double J_input, double t_input, double tPrime_input, vector<double> Bzval_input, double U_input, bool CALC_SQUARE, size_t D_input)
	:KondoModel(Lx_input,"KondoModel (PKS geometry) ", Ly_input, D_input ),
	J(J_input), t(t_input), tPrime(tPrime_input), U(U_input), D(D_input)
{	
	// initialize member variable imploc
	this->imploc.resize(N_sites);
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
	std::vector<Eigen::MatrixXd> tIntra(this->N_sites);
	std::vector<Eigen::MatrixXd> tInter(this->N_sites);

	for (size_t l=0; l<this->N_sites; ++l)
	{
		tInter[l].resize(N_legs,N_legs); tInter[l].setZero();
		tIntra[l].resize(N_legs,N_legs);

		if(((l) % 3 == 0))
		{
			tIntra[l].setZero();
			for (size_t i=0; i<N_legs; i++)
			{
				if(i % 2 == 0 ) {tInter[l](i,i)=t;}
				if(i % 2 == 1 ) {tInter[l](i,i)=tPrime;}
				if(i < N_legs-1) {tIntra[l](i,i+1)=tPrime;}
			}
			for (size_t i=1; i<N_legs; i+=2)
			{
				tInter[l](i,i-1) = t;
				if (i+1 < N_legs ) {tInter[l](i+1,i) = t;}
			}

		}
		if(((l) % 3 == 1))
		{
			tIntra[l].setZero();
			for (size_t i=0; i<N_legs; i++)
			{
				if(i % 2 == 0 ) {tInter[l](i,i)=t;}
				if(i % 2 == 1 ) {tInter[l](i,i)=t;}
				if(i < N_legs-1) {tIntra[l](i,i+1)=t;}
			}
			for (size_t i=1; i<N_legs; i+=2)
			{
				tInter[l](i,i-1) = t;
				if (i+1 < N_legs ) {tInter[l](i+1,i) = tPrime;}
			}
			
		}
		if(((l) % 3 == 2))
		{
			tIntra[l].setZero();		
			for (size_t i=0; i<N_legs; i++)
			{
				if(i % 2 == 0 ) {tInter[l](i,i)=tPrime;}
				if(i % 2 == 1 ) {tInter[l](i,i)=t;}
				if(i < N_legs-1) {tIntra[l](i,i+1)=t;}				
			}
			for (size_t i=1; i<N_legs; i+=2)
			{
				tInter[l](i,i-1) = t;
				if (i+1 < N_legs ) {tInter[l](i+1,i) = t;}
			}
			
		}
	}

	F = FermionBase(N_legs);
	S = SpinBase(N_legs,D);

	for (size_t l=0; l<this->N_sites; ++l)
	{
		MpoQ<2>::qloc[l].resize(F.dim()*S.dim());
		for (size_t j=0; j<S.dim(); j++)
			for (size_t i=0; i<F.dim(); i++)
			{
				MpoQ<2>::qloc[l][i+F.dim()*j] = ::qvacuum<2>();
				MpoQ<2>::qloc[l][i+F.dim()*j] = F.qNums(i);
				MpoQ<2>::qloc[l][i+F.dim()*j][1] += S.qNums(j)[0];
			}
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
			KondoModel::set_operators(Olocal,Otight,Onextn,F,S,J,Bzval[l],tInter[l],tIntra[l],0.,U);
			this->Daux = 2 + Otight.size() + 2*Onextn.size();
			G[l].setRowVector(Daux,F.dim()*S.dim());
			G[l] = ::Generator(Olocal,Otight,Onextn).row(Daux-1);

			if (CALC_SQUARE == true)
			{
				Gsq[l].setRowVector(Daux*Daux,F.dim()*S.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else if (l==this->N_sites-1)
		{
			KondoModel::set_operators(Olocal,Otight,Onextn,F,S,J,Bzval[l],tInter[l],tIntra[l],0.,U);
			G[l].setColVector(Daux,F.dim()*S.dim());			
			G[l] = ::Generator(Olocal,Otight,Onextn).col(0);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setColVector(Daux*Daux,F.dim()*S.dim());
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
		else
		{
			KondoModel::set_operators(Olocal,Otight,Onextn,F,S,J,Bzval[l],tInter[l],tIntra[l],0.,U);
			G[l].setMatrix(Daux,F.dim()*S.dim());			
			G[l] = ::Generator(Olocal,Otight,Onextn);
			if (CALC_SQUARE == true)
			{
				Gsq[l].setMatrix(Daux*Daux,F.dim()*S.dim());
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
	
}

