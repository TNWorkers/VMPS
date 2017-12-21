#ifndef STRAWBERRY_GRANDKONDOMODEL
#define STRAWBERRY_GRANDKONDOMODEL

#include "MpKondoModel.h"
#include "MpGrandHubbardModel.h"
#include "FermionBase.h"
#include "SpinBase.h"
#include "qarray.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z
\f$.
The set of impurities \f$I\f$ is completely free to choose.
\note \f$J<0\f$ : antiferromagnetic
\note The local magnetic fields act on the impurities only.
\note If nnn-hopping is positive, the GS-energy is lowered.*/
class GrandKondoModel : public MpoQ<0,double>
{
public:
	/**Does nothing.*/
	GrandKondoModel ():MpoQ() {};

	GrandKondoModel (size_t L_input, double J_input, double Bz_input=0., double tPrime_input=0.,
					 double U_input=0., double mu_input=0., double Bx_input=0., bool OPEN_BC=false, bool CALC_SQUARE=true);
	
	GrandKondoModel (size_t L_input, std::array<double,2> t_input, double J_input, double Bz_input=0.,
					 double tPrime_input=0., double U_input=0., double mu_input=0., double Bx_input=0., bool OPEN_BC=false, bool CALC_SQUARE=true);
		
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<0,double>                           StateXd;
	typedef MpsQ<0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<0,GrandKondoModel,double>    Solver;
	typedef VumpsSolver<0,GrandKondoModel>           uSolver;
	typedef MpsQCompressor<0,double,double>          CompressorXd;
	typedef MpsQCompressor<0,complex<double>,double> CompressorXcd;
	typedef MpoQ<0>                                  Operator;
	///@}
		
	///@{
	MpoQ<0> n (SPIN_INDEX sigma, size_t loc) const;
	MpoQ<0> Simp (size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<0> Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy=0);
	MpoQ<0> SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<0> SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<0> SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y=0, size_t loc2y=0);
	MpoQ<0> SimpSsubSimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
	                          size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);
	MpoQ<0> SimpSsubSimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, 
	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
	                          size_t loc1y=0, size_t loc2y=0, size_t loc3y=0, size_t loc4y=0);
	MpoQ<0> d (size_t locx, size_t locy=0);
	MpoQ<0> c (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<0> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<0> cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
protected:
	
	double J=-1., Bz=0., tPrime=0., U=0., mu=0., Bx=0.;
	size_t D=2;
	std::array<double,2> t;

	vector<double> Bzval;
	vector<size_t> imploc;
	FermionBase F; SpinBase S;	
};

GrandKondoModel::
GrandKondoModel (size_t L_input, double J_input, double Bz_input, double tPrime_input, double U_input,
				 double mu_input, double Bx_input, bool OPEN_BC, bool CALC_SQUARE)
:MpoQ<0> (L_input, 1, vector<qarray<0> >(begin(qloc8dummy),end(qloc8dummy)), {}, labeldummy, "GrandKondoModel"),
	J(J_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input), mu(mu_input), Bx(Bx_input)
{
	stringstream ss;
	ss << "(J=" << J << ",mu=" << mu << ")";
	this->label += ss.str();
	
	F = FermionBase(1,!isfinite(U));
	S = SpinBase(1,2);
	
	vector<double> Uvec(1); Uvec[0] = U;
	vector<double> muvec(1); muvec[0] = -mu; // H=H_0-mu*N
	MatrixXd tInter(1,1); tInter(0,0) = 1.;
	HamiltonianTermsXd Terms = KondoModel::set_operators(F,S,J,Bz,tInter,1.,Bx,tPrime,mu);
	this->Daux = Terms.auxdim();
	
	SuperMatrix<double> G = ::Generator(Terms);
	this->construct(G, this->W, this->Gvec, OPEN_BC);
	
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

GrandKondoModel::
GrandKondoModel (size_t L_input, std::array<double,2> t_input, double J_input, double Bz_input, double tPrime_input,
				 double U_input, double mu_input, double Bx_input, bool OPEN_BC, bool CALC_SQUARE)
:MpoQ<0> (L_input, 1, vector<qarray<0> >(begin(qloc8dummy),end(qloc8dummy)), {}, labeldummy, "GrandKondoModel"),
	J(J_input), t(t_input), Bz(Bz_input), tPrime(tPrime_input), U(U_input), mu(mu_input), Bx(Bx_input)
{
	stringstream ss;
	ss << "(J=" << J << ",mu=" << mu << ")";
	this->label += ss.str();
	
	F = FermionBase(1,!isfinite(U));
	S = SpinBase(1,2);

	vector<double> Jvec(1); Jvec[0] = J;
	vector<double> muvec(1); muvec[0] = -mu; // H=H_0-mu*N
	MatrixXd tInter01(1,1); tInter01(0,0) = t[0];
	MatrixXd tInter10(1,1); tInter10(0,0) = t[1];
	HamiltonianTermsXd Terms01 = KondoModel::set_operators(F,S,J,Bz,tInter01,1.,Bx,tPrime,U,mu);
	HamiltonianTermsXd Terms10 = KondoModel::set_operators(F,S,J,Bz,tInter10,1.,Bx,tPrime,U,mu);
	this->Daux = Terms01.auxdim();
	
	SuperMatrix<double> G01 = ::Generator(Terms01);
	SuperMatrix<double> G10 = ::Generator(Terms10);
	
	vector<SuperMatrix<double> > G(this->N_sites);
	vector<SuperMatrix<double> > Gsq;
	if (CALC_SQUARE == true)
	{
		Gsq.resize(this->N_sites);
	}
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		// if OPENBC
		this->Daux = Terms01.auxdim();
		G[l].setMatrix(Daux,F.dim());
		G[l] = (l%2==0)? G01:G10;
		
		if (CALC_SQUARE == true)
		{
			Gsq[l] = tensor_product(G[l],G[l]);
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

MpoQ<0> GrandKondoModel::
n (SPIN_INDEX sigma, size_t loc) const
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "n(" << loc << ",σ=" << sigma << ")";
	MpoQ<0> Mout(N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	MatrixXd IdImp(S.dim(),S.dim()); IdImp.setIdentity();
	Mout.setLocal(loc, kroneckerProduct(IdImp.sparseView(),F.n(sigma)));
	return Mout;
}

MpoQ<0> GrandKondoModel::
d (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	MatrixXd IdImp(S.dim(), S.dim()); IdImp.setIdentity();
	MpoQ<0> Mout(N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	Mout.setLocal(locx, kroneckerProduct(IdImp.sparseView(),F.d(locy)));
	return Mout;
}

MpoQ<0> GrandKondoModel::
Simp (size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<this->N_sites);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";
	MpoQ<0> Mout(this->N_sites, 1, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(S.Scomp(SOP,locy),IdSub.sparseView()));
	return Mout;
}

MpoQ<0> GrandKondoModel::
Ssub (size_t locx, SPINOP_LABEL SOP, size_t locy)
{
	assert(locx<this->N_sites);
	stringstream ss;
	ss << SOP << "(" << locx << "," << locy << ")";
	MpoQ<0> Mout(this->N_sites, this->N_legs, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	MatrixXd IdImp(S.dim(), S.dim()); IdImp.setIdentity();
	Mout.setLocal(locx, kroneckerProduct(IdImp.sparseView(), F.Scomp(SOP,locy)));
	return Mout;
}

MpoQ<0> GrandKondoModel::
SimpSimp (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<0> Mout(this->N_sites, this->N_legs, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub.sparseView()), 
				kroneckerProduct(S.Scomp(SOP2,loc2y),IdSub.sparseView())});
	return Mout;
}


MpoQ<0> GrandKondoModel::
SsubSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<0> Mout(this->N_sites, this->N_legs, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	MatrixXd IdImp1(S.dim(), S.dim()); IdImp1.setIdentity();
	MatrixXd IdImp2(S.dim(), S.dim()); IdImp2.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(IdImp1.sparseView(),F.Scomp(SOP1,loc1y)), 
				kroneckerProduct(IdImp2.sparseView(),F.Scomp(SOP2,loc2y))}
		);
	return Mout;
}

MpoQ<0> GrandKondoModel::
SimpSsub (size_t loc1x, SPINOP_LABEL SOP1, size_t loc2x, SPINOP_LABEL SOP2, size_t loc1y, size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")";
	MpoQ<0> Mout(this->N_sites, this->N_legs, MpoQ<0>::qloc, {}, labeldummy, ss.str());
	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
	MatrixXd IdImp(S.dim(), S.dim()); IdImp.setIdentity();
	Mout.setLocal({loc1x, loc2x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub.sparseView()), 
				kroneckerProduct(IdImp.sparseView(),F.Scomp(SOP2,loc2y))}
		);
	return Mout;
}

} //end namespace VMPS
#endif
