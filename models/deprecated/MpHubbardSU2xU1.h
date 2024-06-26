#ifndef HUBBARDMODELSU2XU1_H_
#define HUBBARDMODELSU2XU1_H_

#include "MpoQ.h"
#include "MultipedeQ.h"
#include "fermions/BaseSU2xU1.h"
#include "symmetry/SU2xU1.h"

namespace VMPS::models
{

/** \class HubbardSU2xU1
  * \ingroup Models
  *
  * \brief Hubbard Model
  *
  * MPO representation of 
  * 
  \f$
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  + U \sum_i n_{i\uparrow} n_{i\downarrow}
  + V \sum_{<ij>} n_{i} n_{j}
  \f$.
  *
  \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
  \note If the nnn-hopping is positive, the ground state energy is lowered.
  \warning \f$J>0\f$ is antiferromagnetic
  \todo Implement more observables.
  */
class HubbardSU2xU1 : public MpoQ<Sym::SU2xU1<double> ,double>
{
public:
	typedef Sym::SU2xU1<double> Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	template<Index Rank> using TensorType = Eigen::Tensor<double,Rank,Eigen::ColMajor,Index>;
	typedef SiteOperator<Symmetry,double> Operator;
public:

	/**Does nothing.*/
	HubbardSU2xU1() : MpoQ(){};
	
	/**
	\param Lx_input : chain length
	\param U_input : \f$U\f$
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$, next-nearest-neighbour (nnn) hopping. A minus sign in front of the hopping terms is assumed, so that \f$t^{\prime}>0\f$ is the usual choice.
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param t_input : \f$t\f$, nearest-neighbour hopping. Convenient sign is positive.
	*/
	HubbardSU2xU1 (std::size_t Lx_input, double U_input, double V_input=0., double tPrime_input=0.,
				   std::size_t Ly_input=1, bool CALC_SQUARE=false, double t_input=1.);
			
	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. KondoModel.
	\param F : the FermionBase class where the operators are pulled from
	\param U : \f$U\f$
	\param V : \f$V\f$
	\param tPrime : \f$t'\f$
	\param tIntra : hopping within the rungs of ladder (or between legs)
	\param J : \f$J\f$
	\param PERIODIC : if \p true, makes periodic boundary conditions in y-direction, i.e. a cylinder
	\param tInter : \f$tInter\f$, nn hopping (always set to 1.)
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const fermions::BaseSU2xU1<> &F, double U, 
													   double V=0., double tPrime=0., double tIntra=1., double J=0., bool PERIODIC=false, double tInter=1.);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const fermions::BaseSU2xU1<> &F, vector<double> U, Eigen::MatrixXd tInter,
	                                         double V=0., double tPrime=0., double tIntra=1., double J=0., bool PERIODIC=false);

	static const std::vector<qType> getqloc (const fermions::BaseSU2xU1<> &F);
	static const std::vector<qType> getqOp();
	static const std::vector<Index> getqlocDeg(const fermions::BaseSU2xU1<> &F);
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,Symmetry::Nq> Slabel;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<Symmetry,double>                            StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<Symmetry,complex<double> >                  StateXcd;
	typedef DmrgSolverQ<Symmetry,HubbardSU2xU1,double,false> Solver;
	typedef MpsQCompressor<Symmetry,double,double>           CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double>  CompressorXcd;
	typedef MpoQ<Symmetry,double>                            OperatorXd;
	typedef MpoQ<Symmetry,complex<double> >                  OperatorXcd;
	
	MpoQ<Symmetry> Auger (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> eta(std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> Aps (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> c (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> cdag (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> cdagc (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	MpoQ<Symmetry> d (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> n (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> S (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> Sdag (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> SSdag (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	MpoQ<Symmetry> EtaEtadag (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);	
	MpoQ<Symmetry> triplon (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> antitriplon (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> quadruplon (std::size_t locx, std::size_t locy=0);

protected:
	double U;	
	double V = 0.;
	double tPrime = 0.;
	double t=1.;
	fermions::BaseSU2xU1<> F;
};

const std::array<string,Sym::SU2xU1<double>::Nq> HubbardSU2xU1::Slabel{"S","N"};

const std::vector<Sym::SU2xU1<double>::qType> HubbardSU2xU1::
getqloc (const fermions::BaseSU2xU1<> &F)
{
	return F.qloc();
}

const std::vector<Index> HubbardSU2xU1::
getqlocDeg (const fermions::BaseSU2xU1<> &F)
{
	return F.qlocDeg();
}

const std::vector<Sym::SU2xU1<double>::qType> HubbardSU2xU1::
getqOp ()
{	
	std::vector<qType> vout(3);
	vout[0] = {1,0}; //d and Identity operators
	vout[1] = {2,+1}; //cdag
	vout[2] = {2,-1}; //c
	return vout;
}

HamiltonianTermsXd<Sym::SU2xU1<double> > HubbardSU2xU1::
set_operators (const fermions::BaseSU2xU1<> &F, double U, double V, double tPrime, double tIntra, double J, bool PERIODIC, double tInter)
{
	std::vector<double> Uvec(F.orbitals());
	std::fill(Uvec.begin(), Uvec.end(), U);
	Eigen::MatrixXd tInterMat(F.orbitals(),F.orbitals()); tInterMat.setIdentity(); tInterMat *= tInter;
	// return set_operators(F, Uvec, Eigen::MatrixXd::Identity(F.orbitals(),F.orbitals()), V, tPrime, tIntra, J, PERIODIC);
	return set_operators(F, Uvec, tInterMat, V, tPrime, tIntra, J, PERIODIC);

}

HamiltonianTermsXd<Sym::SU2xU1<double> > HubbardSU2xU1::
set_operators (const fermions::BaseSU2xU1<> &F, vector<double> Uvec, Eigen::MatrixXd tInter, double V, double tPrime, double tIntra, double J, bool PERIODIC)
{
	// assert(F.orbitals() == 1 and "More than one orbital in the Hubbard-Model is not working until now.");
	assert(Uvec.size() == F.orbitals());
	HamiltonianTermsXd<Sym::SU2xU1<double> > Terms;
	for (int legI=0; legI<F.orbitals(); ++legI)
	for (int legJ=0; legJ<F.orbitals(); ++legJ)
	{
		if (tInter(legI,legJ) != 0.)
		{
			Terms.tight.push_back(std::make_tuple(tInter(legI,legJ)*std::sqrt(2.), F.cdag(legI), Operator::prod(F.sign(),F.c(legJ),{2,-1})));
			Terms.tight.push_back(std::make_tuple(tInter(legI,legJ)*std::sqrt(2.), F.c(legI), Operator::prod(F.sign(),F.cdag(legJ),{2,1})));
		}
		if (V != 0. and legI == legJ)
		{
			Terms.tight.push_back(std::make_tuple(V, F.n(legI), F.n(legJ)));
		}
		if (J != 0. and legI == legJ)
		{
			Terms.tight.push_back(std::make_tuple(-std::sqrt(3)*J, F.Sdag(legI), F.S(legJ)));
		}
	}
	
	if (tPrime != 0.)
	{
		Terms.nextn.push_back(make_tuple(tPrime*std::sqrt(2.), F.cdag(), Operator::prod(F.sign(),F.c(),{2,-1}), F.sign()));
		Terms.nextn.push_back(make_tuple(tPrime*std::sqrt(2.), F.c(), Operator::prod(F.sign(),F.cdag(),{2,1}), F.sign()));
	}
	Terms.local.push_back(std::make_tuple(1.,F.HubbardHamiltonian(Uvec[0],tIntra,V,J,PERIODIC)));
	Terms.Id = F.Id();
	return Terms;
}

HubbardSU2xU1::
HubbardSU2xU1 (std::size_t Lx_input, double U_input, double V_input, double tPrime_input, std::size_t Ly_input, bool CALC_SQUARE, double t_input)
	:MpoQ<Symmetry> (Lx_input, Ly_input),
	U(U_input), V(V_input), tPrime(tPrime_input), t(t_input)
{
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");

	//assign stuff
	this->label = "HubbardSU2⊗U1";
	std::stringstream ss;
	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ")";
	this->label += ss.str();
	
	F = fermions::BaseSU2xU1<>(N_legs,!std::isfinite(U));
	for (std::size_t l=0; l<N_sites; l++)
	{
		qloc__[l] = F.basis();
		qloc[l] = qloc__[l].qloc();
		qOp[l] = HubbardSU2xU1::getqOp();		
	}
	Qtot = {1,0};
	qlabel = HubbardSU2xU1::Slabel;

	HamiltonianTermsXd<Sym::SU2xU1<double> > Terms = set_operators(F, U, V, tPrime, t, 0.,false,t);
	auto G = ::Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G);

	// this->Wloc = F.c();
	// this->Wlocdag = F.cdag();
	if (CALC_SQUARE == true)
	{
		// this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}


// MpoQ<Symmetry> HubbardSU2xU1::
// Auger (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "Auger(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {1}, HubbardSU2xU1::Slabel, ss.str());
// 	Mout.setLocal(locx, F.c(UP,locy)*F.c(DN,locy));
// 	return Mout;
// }

// MpoQ<Symmetry> HubbardSU2xU1::
// eta()
// {
// 	stringstream ss;
// 	ss << "eta";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {-1,-1}, HubbardSU2xU1::Nlabel, ss.str());
// 	SparseMatrixXd etaloc = MatrixXd::Identity(F.dim(),F.dim()).sparseView();
// 	for (int ly=0; ly<N_legs; ++ly) {etaloc = etaloc * pow(-1.,ly) * F.c(UP,ly)*F.c(DN,ly);}
// 	Mout.setLocalSum(etaloc, true);
// 	return Mout;
// }

// MpoQ<Symmetry> HubbardSU2xU1::
// Aps (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "Aps(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {+1,+1}, HubbardSU2xU1::Nlabel, ss.str());
// 	Mout.setLocal(locx, F.cdag(DN,locy)*F.cdag(UP,locy));
// 	return Mout;
// }

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
c (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "c(" << locx << "," << locy << ")";

	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	Mout.label = ss.str();
	Mout.setQtarget({2,-1});
	Mout.qlabel = HubbardSU2xU1::Slabel;

	Mout.setLocal(locx, F.c(locy), {2,-1}, F.sign());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
cdag (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "c†(" << locx << "," << locy << ")";

	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	Mout.label = ss.str();
	Mout.setQtarget({2,+1});
	Mout.qlabel = HubbardSU2xU1::Slabel;

	Mout.setLocal(locx, F.cdag(locy), {2,+1}, F.sign());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
cdagc (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "c†(" << loc1x << "," << loc1y << ")" << "c(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	auto cdag = F.cdag(loc1y);
	auto c = F.c(loc2y);
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2xU1::Slabel;
	if(loc1x == loc2x)
	{
		auto product = std::sqrt(2.)*Operator::prod(cdag,c,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
		return Mout;
	}
	else if(loc1x<loc2x)
	{

		Mout.setLocal({loc1x, loc2x}, {std::sqrt(2.)*cdag, Operator::prod(F.sign(),c,{2,-1})}, {{2,1},{2,-1}}, F.sign());
		return Mout;
	}
	else if(loc1x>loc2x)
	{

		Mout.setLocal({loc1x, loc2x}, {std::sqrt(2.)*Operator::prod(F.sign(),cdag,{2,+1}), c}, {{2,1},{2,-1}}, F.sign());
		return Mout;
	}
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
d (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2xU1::Slabel;
	Mout.setLocal(locx, F.d(locy), Symmetry::qvacuum());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
n (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "n(" << locx << "," << locy << ")";

	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2xU1::Slabel;
	Mout.setLocal(locx, F.n(locy), Symmetry::qvacuum());
	return Mout;
}

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// S (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "S(" << locx << "," << locy << ")";

// 	std::vector<std::vector<qType> > qOptmp(N_sites);
// 	for (std::size_t l=0; l<N_sites; l++)
// 	{
// 		qOptmp[l].resize(1);
// 		qOptmp[l][0] = (l == locx) ? 3 : 1;
// 	}

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2xU1::Slabel, ss.str());
// 	Mout.setLocal(locx, F.S(locy));
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// Sdag (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "S†(" << locx << "," << locy << ")";

// 	std::vector<std::vector<qType> > qOptmp(N_sites);
// 	for (std::size_t l=0; l<N_sites; l++)
// 	{
// 		qOptmp[l].resize(1);
// 		qOptmp[l][0] = (l == locx) ? 3 : 1;
// 	}

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2xU1::Slabel, ss.str());
// 	Mout.setLocal(locx, F.Sdag(locy));
// 	return Mout;
// }

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
SSdag (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "S†(" << loc1x << "," << loc1y << ")" << "S(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	auto Sdag = F.Sdag(loc1y);
	auto S = F.S(loc2y);
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2xU1::Slabel;
	if(loc1x == loc2x)
	{
		auto product = std::sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {std::sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
		return Mout;
	}
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
EtaEtadag (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "η†(" << loc1x << "," << loc1y << ")" << "η(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

	auto Etadag = F.Etadag(loc1y);
	auto Eta = F.Eta(loc2y);
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2xU1::Slabel;
	if(loc1x == loc2x)
	{
		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {Etadag, Eta}, {{1,2},{1,-2}});
		return Mout;
	}
}

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// SSdag (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
// {
// 	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
// 	std::stringstream ss;
// 	ss << "S†S(" << locx1 << "," << locy1 << ")" << "Sz(" << locx2 << "," << locy2 << ")";
// 	std::vector<std::vector<qType> > qOptmp(N_sites);
// 	for (std::size_t l=0; l<N_sites; l++)
// 	{
// 		qOptmp[l].resize(1);
// 		qOptmp[l][0] = (l == locx1 or l == locx2) ? 3 : 1;
// 	}

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {1}, HubbardSU2xU1::Slabel, ss.str());
// 	Mout.setLocal({locx1,locx2}, {F.S(locy1),F.Sdag(locy2)});
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
// 	qarray<2> qdiff;
// 	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0) = F.sign();
// 	}
// 	// c(locx,UP)*c(locx,DN)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	// c(locx+1,UP|DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = (sigma==UP)? F.c(UP,locy) : F.c(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2xU1::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
// 	qarray<2> qdiff;
// 	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0) = F.sign();
// 	}
// 	// c†(locx,DN)*c†(locx,UP)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.cdag(DN,locy)*F.cdag(UP,locy);
// 	// c†(locx+1,UP|DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = (sigma==UP)? F.cdag(UP,locy) : F.cdag(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2xU1::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// quadruplon (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
// 	// c(loc,UP)*c(loc,DN)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	// c(loc+1,UP)*c(loc+1,DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,4);
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, {-2,-2}, HubbardSU2xU1::Nlabel, ss.str());
// }

} //end namespace VMPS::models

#endif
