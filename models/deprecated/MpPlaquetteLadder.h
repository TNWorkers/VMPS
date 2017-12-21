#ifndef VANILLA_PLAQUETTELADDER
#define VANILLA_PLAQUETTELADDER

#include "MpHubbardModel.h"

namespace VMPS
{

class PlaquetteLadder : public MpoQ<0,double>
{
public:
	
	PlaquetteLadder (size_t Lx_input, size_t Ly_input, 
	                 double tPara_input, double tPerp_input, double tDiag_input, double tIntr_input, 
	                 double U_input, double mu_input, bool OPEN_BC=false, bool CALC_SQUARE=true);
	
	typedef MpsQ<0,double>                           StateXd;
	typedef MpsQ<0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<0,PlaquetteLadder,double>    Solver;
	typedef VumpsSolver<0,PlaquetteLadder,double>    uSolver;
	typedef MpsQCompressor<0,double,double>          CompressorXd;
	typedef MpsQCompressor<0,complex<double>,double> CompressorXcd;
	typedef MpoQ<0>                                  Operator;
	
	MpoQ<0> n (SPIN_INDEX sigma, size_t locx, size_t locy) const;
	MpoQ<0> d (size_t locx, size_t locy) const;
//	MpoQ<0> Sz (size_t loc) const;
	MpoQ<0> singletDensityPerp() const;
	vector<MpoQ<0> > singletCorPerpPara() const;
	vector<MpoQ<0> > singletDensityDiag() const;
	vector<MpoQ<0> > singletCorDiagPara() const;
	
private:
	
	FermionBase F;
	
	double tPara, tPerp, tDiag, tIntr;
	double U, mu;
};

PlaquetteLadder::
PlaquetteLadder (size_t Lx_input, size_t Ly_input, double tPara_input, double tPerp_input, double tDiag_input, double tIntr_input, double U_input, double mu_input, bool OPEN_BC, bool CALC_SQUARE)
:MpoQ<0> (Lx_input, Ly_input, 
          (Ly_input==1)? vector<qarray<0> >(begin(qloc4dummy),end(qloc4dummy)):vector<qarray<0> >(begin(qloc16dummy),end(qloc16dummy)), 
          {}, labeldummy, "PlaquetteLadder"),
tPara(tPara_input), tPerp(tPerp_input), tDiag(tDiag_input), tIntr(tIntr_input), U(U_input), mu(mu_input)
{
	assert(Ly_input==1 or Ly_input==2);
	stringstream ss;
	ss << "(U=" << U << ",mu=" << mu << ",t∥=" << tPara << ",t⟂=" << tPerp << ",t╳=" << tDiag << ",t¦=" << tIntr << ")";
	this->label += ss.str();
	
	F = FermionBase(N_legs,!isfinite(U));
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	vector<double> Uvec; Uvec.assign(N_legs,U);
	vector<double> muvec; muvec.assign(N_legs,-mu); // H=H_0-mu*N
	
	if (N_legs == 2)
	{
		MatrixXd tInter01(2,2); tInter01 << tPara, tDiag, tDiag, tPara;
		MatrixXd tInter10(2,2); tInter10 << tIntr, 1e-15, 1e-15, tIntr;
		
		HamiltonianTermsXd Terms01 = HubbardModel::set_operators(F,Uvec,muvec,tInter01,0,0,tPerp);
		HamiltonianTermsXd Terms10 = HubbardModel::set_operators(F,Uvec,muvec,tInter10,0,0,tPerp);
		this->Daux = Terms01.auxdim();
		
		SuperMatrix<double> G01 = ::Generator(Terms01);
		SuperMatrix<double> G10 = ::Generator(Terms10);
		
		for (size_t l=0; l<this->N_sites; ++l)
		{
			G[l].setMatrix(Daux,F.dim());
			G[l] = (l%2==0)? G01:G10;
			
			if (CALC_SQUARE == true)
			{
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
	}
	else if (N_legs == 1)
	{
		MatrixXd tPara_(1,1); tPara_ << tPara;
		MatrixXd tPerp_(1,1); tPerp_ << tPerp;
		MatrixXd tIntr_(1,1); tIntr_ << tIntr;
		MatrixXd tEmtp_(1,1); tEmtp_ << 1e-15;
		
		std::array<HamiltonianTermsXd,4> Terms;
		Terms[0] = HubbardModel::set_operators(F,Uvec,muvec,tPerp_,0,tPara);
		Terms[1] = HubbardModel::set_operators(F,Uvec,muvec,tIntr_,0,tPara);
		Terms[2] = HubbardModel::set_operators(F,Uvec,muvec,tPerp_,0,tIntr);
		Terms[3] = HubbardModel::set_operators(F,Uvec,muvec,tEmtp_,0,tIntr);
		this->Daux = Terms[0].auxdim();
		
		std::array<SuperMatrix<double>,4> Gcell;
		for (size_t l=0; l<4; ++l)
		{
			Gcell[l].set(this->Daux,this->Daux,4);
			Gcell[l] = ::Generator(Terms[l]);
		}
		
		for (size_t l=0; l<this->N_sites; ++l)
		{
			G[l].setMatrix(Daux,F.dim());
			G[l] = Gcell[l%4];
			
			if (CALC_SQUARE == true)
			{
				Gsq[l] = tensor_product(G[l],G[l]);
			}
		}
	}
	
	this->construct(G, this->W, this->Gvec, OPEN_BC);
	
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

MpoQ<0> PlaquetteLadder::
n (SPIN_INDEX sigma, size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "n(" << locx << "," << locy << ",σ=" << sigma << ")";
	MpoQ<0> Mout(N_sites, 2, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(locx,F.n(sigma,locy));
	return Mout;
}

MpoQ<0> PlaquetteLadder::
d (size_t locx, size_t locy) const
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "d(" << locx << "," << locy << ")";
	MpoQ<0> Mout(N_sites, 2, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(locx,F.d(locy));
	return Mout;
}

//MpoQ<0> PlaquetteLadder::
//Sz (size_t loc) const
//{
//	assert(loc<N_sites);
//	stringstream ss;
//	ss << "Sz(" << loc << ")";
//	MpoQ<0> Mout(N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
//	Mout.setLocal(loc, F.Sz());
//	return Mout;
//}

MpoQ<0> PlaquetteLadder::
singletDensityPerp() const
{
	assert(N_legs==2);
	
	stringstream ss;
	ss << "singletDensity⟂";
	
	SparseMatrixXd s    = (F.c(DN,0)*F.c(UP,1)-F.c(UP,0)*F.c(DN,1))/sqrt(2);
	SparseMatrixXd sdag = (F.cdag(UP,1)*F.cdag(DN,0)-F.cdag(DN,1)*F.cdag(UP,0))/sqrt(2);
//	SparseMatrixXd alt = 0.5*( F.n(UP,0)*F.n(DN,1)
//	                          +F.n(DN,0)*F.n(UP,1)
//	                          -F.Scomp(SP,0)*F.Scomp(SM,1)
//	                          -F.Scomp(SM,0)*F.Scomp(SP,1)); // alt=sdag*s
	
	MpoQ<0> Mout(N_sites, N_legs, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(0,sdag*s);
	return Mout;
}

vector<MpoQ<0> > PlaquetteLadder::
singletCorPerpPara() const
{
	assert(N_legs==2 and N_sites==2);
	
	stringstream ss;
	ss << "singletCorPerpPara";
	
	vector<SuperMatrix<double> > M0(N_sites);
	vector<SuperMatrix<double> > M1(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		M0[l].setMatrix(1,F.dim());
		M1[l].setMatrix(1,F.dim());
	}
	
	SparseMatrixXd singlet01 = (F.c(UP,0)*F.c(DN,1)-F.c(DN,0)*F.c(UP,1))/sqrt(2);
	
	M0[0](0,0) = singlet01 * F.cdag(DN,0) * F.sign();
	M0[1](0,0) = F.cdag(UP,0)/sqrt(2);
	
	M1[0](0,0) = -singlet01 * F.cdag(UP,0) * F.sign();
	M1[1](0,0) = F.cdag(DN,0)/sqrt(2);
	
	vector<MpoQ<0> > Mout(2);
	Mout[0] = MpoQ<0>(N_sites, 2, M0, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout[1] = MpoQ<0>(N_sites, 2, M1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	
	return Mout;
}

vector<MpoQ<0> > PlaquetteLadder::
singletDensityDiag() const
{
	assert(N_legs==2 and N_sites==2);
	
	stringstream ss;
	ss << "singletCorPerpPara";
	
	vector<SuperMatrix<double> > M0(N_sites);
	vector<SuperMatrix<double> > M1(N_sites);
	vector<SuperMatrix<double> > M2(N_sites);
	vector<SuperMatrix<double> > M3(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		M0[l].setMatrix(1,F.dim());
		M1[l].setMatrix(1,F.dim());
		M2[l].setMatrix(1,F.dim());
		M3[l].setMatrix(1,F.dim());
	}
	
	M0[0](0,0) = 0.5*F.n(UP,0);
	M0[1](0,0) =     F.n(DN,1);
	
	M1[0](0,0) = 0.5*F.n(DN,0);
	M1[1](0,0) =     F.n(UP,1);
	
	M2[0](0,0) = -0.5*F.Scomp(SM,0);
	M2[1](0,0) =      F.Scomp(SP,1);
	
	M3[0](0,0) = -0.5*F.Scomp(SP,0);
	M3[1](0,0) =      F.Scomp(SM,1);
	
	vector<MpoQ<0> > Mout(4);
	Mout[0] = MpoQ<0>(N_sites, 2, M0, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout[1] = MpoQ<0>(N_sites, 2, M1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout[2] = MpoQ<0>(N_sites, 2, M2, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout[3] = MpoQ<0>(N_sites, 2, M3, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	
	return Mout;
}

vector<MpoQ<0> > PlaquetteLadder::
singletCorDiagPara() const
{
	assert(N_legs==2 and N_sites==2);
	
	stringstream ss;
	ss << "singletCorPerpPara";
	
	vector<vector<SuperMatrix<double> > > M(4);
	for (int t=0; t<M.size(); ++t)
	{
		M[t].resize(3);
	}
	
	for (int t=0; t<M.size(); ++t)
	for (size_t l=0; l<3; ++l)
	{
		M[t][l].setMatrix(1,F.dim());
	}
	
	M[0][0](0,0) = F.c(DN,0) * F.sign(0,1);
	M[0][1](0,0) = 0.5 * (F.Id()-F.n(UP,1)) * F.sign(1,1);
	M[0][2](0,0) = F.cdag(DN,1);
	
	M[1][0](0,0) = F.c(UP,0)* F.sign(0,1);
	M[1][1](0,0) = 0.5 * (F.Id()-F.n(DN,1)) * F.sign(1,1);
	M[1][2](0,0) = F.cdag(UP,1);
	
	M[2][0](0,0) = F.c(UP,0) * F.sign(0,1);
	M[2][1](0,0) = 0.5 * F.Scomp(SP,1) * F.sign(1,1);
	M[2][2](0,0) = F.cdag(DN,1);
	
	M[3][0](0,0) = F.c(DN,0) * F.sign(0,1);
	M[3][1](0,0) = 0.5 * F.Scomp(SM,1) * F.sign(1,1);
	M[3][2](0,0) = F.cdag(UP,1);
	
	vector<vector<qarray<0> > > qloc_tmp(3);
	for (int l=0; l<3; ++l)
	{
		qloc_tmp[l] = MpoQ<0>::qloc[l%2];
	}
	
	vector<MpoQ<0> > Mout(4);
	for (int t=0; t<Mout.size(); ++t)
	{
		cout << "t=" << t << endl;
		Mout[t] = MpoQ<0>(3, 2, M[t], qloc_tmp, {}, labeldummy, ss.str());
	}
	
	return Mout;
}

}

#endif
