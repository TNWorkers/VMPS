#ifndef STRAWBERRY_GRANDHUBBARDMODELSU2
#define STRAWBERRY_GRANDHUBBARDMODELSU2

#include "fermions/BaseSU2.h"
#include "MpoQ.h"
#include "MultipedeQ.h"
#include "SiteOperator.h"

namespace VMPS::models
{

/** \class HubbardSU2
  * \ingroup Models
  *
  * \brief Grandcanonical Hubbard Model
  *
  * MPO representation of the grand-canonical HUbbard model: 
  \f$
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  + U \sum_i n_{i\uparrow} n_{i\downarrow}
  + V \sum_{<ij>} n_{i} n_{j}
  \f$.
  *
  \note Take use of the Spin SU(2) symmetry.
  \note If the nnn-hopping is positive, the ground state energy is lowered.
  \warning \f$J>0\f$ is antiferromagnetic
  \todo Implement more observables.
  */
class HubbardSU2 : public MpoQ<Sym::SU2<double> ,double>
{
public:
	typedef Sym::SU2<double> Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	template<Index Rank> using TensorType = Eigen::Tensor<double,Rank,Eigen::ColMajor,Index>;
	typedef SiteOperator<Symmetry,double> Operator;
	
public:
	/**Does nothing.*/
	HubbardSU2() : MpoQ(){};
	
	/**
	\param Lx_input : chain length
	\param U_input : \f$U\f$
	\param mu_input : \f$\mu\f$ (chemical potential)
	\param V_input : \f$V\f$
	\param tPrime_input : \f$t^{\prime}\f$, next-nearest-neighbour (nnn) hopping. A minus sign in front of the hopping terms is assumed, so that \f$t^{\prime}>0\f$ is the usual choice.
	\param Ly_input : amount of legs in ladder
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param t_input : nn Hopping. Always set to +1
	*/
	HubbardSU2 (std::size_t Lx_input, double U_input, double mu_input=std::numeric_limits<double>::infinity(), double V_input=0.,
				double tPrime_input=0., std::size_t Ly_input=1, bool CALC_SQUARE=false, double t_input=1.);
			
	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. KondoModel.
	\param F : the FermionBase class where the operators are pulled from
	\param U : \f$U\f$
	\param mu : \f$\mu\f$ (chemical potential)
	\param V : \f$V\f$
	\param tPrime : \f$t'\f$
	\param tIntra : hopping within the rungs of ladder (or between legs)
	\param J : \f$J\f$
	\param PERIODIC : if \p true, makes periodic boundary conditions in y-direction, i.e. a cylinder
	\param tInter : nn Hopping. Always set to +1
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const fermions::BaseSU2<> &F, double U, double mu, double V=0.,
													   double tPrime=0., double tIntra=1., double J=0., bool PERIODIC=false, double tInter=1.);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const fermions::BaseSU2<> &F, std::vector<double> U, double mu, Eigen::MatrixXd tInter,
													   double V=0., double tPrime=0., double tIntra=1., double J=0., bool PERIODIC=false);
		
	static const std::vector<qType> getqloc (const fermions::BaseSU2<> &F);
	static const std::vector<Sym::SU2<double>::qType> getqOp();
	static const std::vector<Index> getqlocDeg(const fermions::BaseSU2<> &F);
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,1> Slabel;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<Symmetry,double>                            StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<Symmetry,complex<double> >                  StateXcd;
	typedef DmrgSolverQ<Symmetry,HubbardSU2,double,false>    Solver;
	typedef MpsQCompressor<Symmetry,double,double>           CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double>  CompressorXcd;
	typedef MpoQ<Symmetry,double>                            OperatorXd;
	typedef MpoQ<Symmetry,complex<double> >                  OperatorXcd;
	
	MpoQ<Symmetry> Auger (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> eta();
	MpoQ<Symmetry> Aps (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> c (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> cdag (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> d (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> n (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> S (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> Sdag (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> SSdag (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	MpoQ<Symmetry> triplon (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> antitriplon (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> quadruplon (std::size_t locx, std::size_t locy=0);

protected:
	double U;
	double mu = std::numeric_limits<double>::infinity();
	double V = 0.;
	double tPrime = 0.;
	double t = 1.;
	
	fermions::BaseSU2<> F;
};

const std::array<string,1> HubbardSU2::Slabel{"S"};

const std::vector<Sym::SU2<double>::qType> HubbardSU2::
getqloc (const fermions::BaseSU2<> &F)
{
	return F.qloc();
}

const std::vector<Index> HubbardSU2::
getqlocDeg (const fermions::BaseSU2<> &F)
{
	return F.qlocDeg();
}

const std::vector<Sym::SU2<double>::qType> HubbardSU2::
getqOp ()
{	
	std::vector<qType> vout(2);
	vout[0] = {1}; //d and Identity operators
	vout[1] = {2}; //cdag and c operators
	return vout;
}

HamiltonianTermsXd<Sym::SU2<double> > HubbardSU2::
set_operators (const fermions::BaseSU2<> &F, double U, double mu, double V, double tPrime, double tIntra, double J, bool PERIODIC, double tInter)
{
	std::vector<double> Uvec(F.orbitals());
	std::fill(Uvec.begin(), Uvec.end(), U);
	
	return set_operators(F, Uvec, mu, Eigen::MatrixXd::Identity(F.orbitals(),F.orbitals()), V, tPrime, tIntra, J, PERIODIC);
}

HamiltonianTermsXd<Sym::SU2<double> > HubbardSU2::
set_operators (const fermions::BaseSU2<> &F, std::vector<double> Uvec, double mu, Eigen::MatrixXd tInter, double V,
			   double tPrime, double tIntra, double J, bool PERIODIC)
{
	assert(F.orbitals() == 1 and "More than one orbital in the Hubbard-Model is not working until now.");
	assert(Uvec.size() == F.orbitals());
	HamiltonianTermsXd<Sym::SU2<double> > Terms;
	for (int legI=0; legI<F.orbitals(); ++legI)
	for (int legJ=0; legJ<F.orbitals(); ++legJ)
	{
		if (tInter(legI,legJ) != 0.)
		{
			Terms.tight.push_back(std::make_tuple(tInter(legI,legJ)*std::sqrt(2.), F.cdag(legI), Operator::prod(F.sign(),F.c(legJ),{2})));
			Terms.tight.push_back(std::make_tuple(tInter(legI,legJ)*std::sqrt(2.), F.c(legI), Operator::prod(F.sign(),F.cdag(legJ),{2})));
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
		Terms.nextn.push_back(make_tuple(tPrime*std::sqrt(2.), F.cdag(), Operator::prod(F.sign(),F.c(),{2}), F.sign()));
		Terms.nextn.push_back(make_tuple(tPrime*std::sqrt(2.), F.c(), Operator::prod(F.cdag(),F.sign(),{2}), F.sign()));
	}
	Terms.local.push_back(std::make_tuple(1.,F.HubbardHamiltonian(Uvec[0],mu,tIntra,V,J,PERIODIC) ));
	Terms.Id = F.Id();
	return Terms;
}

HubbardSU2::
HubbardSU2 (std::size_t Lx_input, double U_input, double mu_input, double V_input, double tPrime_input,
			std::size_t Ly_input, bool CALC_SQUARE, double t_input)
	:MpoQ<Symmetry> (Lx_input, Ly_input),
	U(U_input), mu(mu_input), V(V_input), tPrime(tPrime_input), t(t_input)
{
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	if (mu == std::numeric_limits<double>::infinity()) { mu = U/2.; } //set mu to particle hole symmetric point as default.

	//assign stuff
	label = "HubbardSU2";
	std::stringstream ss;
	ss << "(U=" << U << ",μ=" << mu << ",V=" << V << ",t'=" << tPrime << ")";
	this->label += ss.str();
	
	F=fermions::BaseSU2<>(N_legs,!std::isfinite(U));
	for (std::size_t l=0; l<N_sites; l++)
	{
		qloc[l] = HubbardSU2::getqloc(F);
		qlocDeg[l] = HubbardSU2::getqlocDeg(F);
		qOp[l] = HubbardSU2::getqOp();		
	}
	Qtot = {1};
	qlabel = HubbardSU2::Slabel;
	
	HamiltonianTermsXd<Sym::SU2<double> > Terms = set_operators(F, U, mu, V, tPrime, t, 0., false, t);
	auto G = ::Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G);

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


// MpoQ<Symmetry> HubbardSU2::
// Auger (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "Auger(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {1}, HubbardSU2::Slabel, ss.str());
// 	Mout.setLocal(locx, F.c(UP,locy)*F.c(DN,locy));
// 	return Mout;
// }

// MpoQ<Symmetry> HubbardSU2::
// eta()
// {
// 	stringstream ss;
// 	ss << "eta";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {-1,-1}, HubbardSU2::Nlabel, ss.str());
// 	SparseMatrixXd etaloc = MatrixXd::Identity(F.dim(),F.dim()).sparseView();
// 	for (int ly=0; ly<N_legs; ++ly) {etaloc = etaloc * pow(-1.,ly) * F.c(UP,ly)*F.c(DN,ly);}
// 	Mout.setLocalSum(etaloc, true);
// 	return Mout;
// }

// MpoQ<Symmetry> HubbardSU2::
// Aps (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "Aps(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {+1,+1}, HubbardSU2::Nlabel, ss.str());
// 	Mout.setLocal(locx, F.cdag(DN,locy)*F.cdag(UP,locy));
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2::
// c (std::size_t locx, std::size_t locy)
// {
// 	std::array<Index,2> reshape_dims = {static_cast<Index>(F.dim()),static_cast<Index>(F.dim())};
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "c(" << locx << "," << locy << ")";
	
// 	std::vector<MultipedeQ<4,SymSU2<double>,double,-2> > M(N_sites);
// 	std::vector<std::vector<qType> > qOptmp(N_sites);

// 	for (std::size_t l=0; l<locx; ++l)
// 	{
// 		qOptmp[l].resize(1); qOptmp[l][0] = 1;
// 		// M[l][0].reshape(reshape_dims) = F.sign();
// 	}
// 	qOptmp[locx].resize(1); qOptmp[locx][0] = 2;
// 	M[locx].resize(1);
// 	M[locx][0].resize(1,1,F.dim(),F.dim());
// 	M[locx][0].reshape(reshape_dims) = F.c(locy);
// 	for (std::size_t l=locx+1; l<N_sites; ++l)
// 	{
// 		qOptmp[l].resize(1); qOptmp[l][0] = 1;
// 		M[l].resize(1);
// 		M[l][0].resize(1,1,F.dim(),F.dim());
// 		M[l][0].reshape(reshape_dims) = F.Id();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qOptmp, {2}, HubbardSU2::Slabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2::
// cdag (std::size_t locx, std::size_t locy)
// {
// 	std::array<Index,2> reshape_dims = {static_cast<Index>(F.dim()),static_cast<Index>(F.dim())};
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "c†(" << locx << "," << locy << ")";
	
// 	std::vector<std::vector<TensorType<4> > > M(N_sites);
// 	std::vector<std::vector<qType> > qOptmp(N_sites);

// 	for (std::size_t l=0; l<locx; ++l)
// 	{
// 		qOptmp[l].resize(1); qOptmp[l][0] = 1;
// 		M[l].resize(1);
// 		M[l][0].resize(1,1,F.dim(),F.dim());
// 		// M[l][0].reshape(reshape_dims) = F.sign();
// 	}
// 	qOptmp[locx].resize(1); qOptmp[locx][0] = 2;
// 	M[locx].resize(1);
// 	M[locx][0].resize(1,1,F.dim(),F.dim());
// 	M[locx][0].reshape(reshape_dims) = F.cdag(locy);
// 	for (std::size_t l=locx+1; l<N_sites; ++l)
// 	{
// 		qOptmp[l].resize(1); qOptmp[l][0] = 1;
// 		M[l].resize(1);
// 		M[l][0].resize(1,1,F.dim(),F.dim());
// 		M[l][0].reshape(reshape_dims) = F.Id();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qOptmp, {2}, HubbardSU2::Slabel, ss.str());

// }

// MpoQ<SymSU2<double> > HubbardSU2::
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
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2::
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
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2::
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
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, {-2,-2}, HubbardSU2::Nlabel, ss.str());
// }

MpoQ<Sym::SU2<double> > HubbardSU2::
d (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry,double> Mout(N_sites, N_legs);
	for (std::size_t l=0; l<N_sites; l++)
	{
		Mout.setLocBasis(HubbardSU2::getqloc(F),l);
		Mout.setLocBasisDeg(HubbardSU2::getqlocDeg(F),l);
	}
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2::Slabel;
	Mout.setLocal(locx, F.d(locy), Symmetry::qvacuum());
	return Mout;
}

MpoQ<Sym::SU2<double> > HubbardSU2::
n (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "occ(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry,double> Mout(N_sites, N_legs);
	for (std::size_t l=0; l<N_sites; l++)
	{
		Mout.setLocBasis(HubbardSU2::getqloc(F),l);
		Mout.setLocBasisDeg(HubbardSU2::getqlocDeg(F),l);
	}
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HubbardSU2::Slabel;
	Mout.setLocal(locx, F.n(locy), Symmetry::qvacuum());
	return Mout;
}

// MpoQ<SymSU2<double> > HubbardSU2::
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

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2::Slabel, ss.str());
// 	Mout.setLocal(locx, F.S(locy));
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2::
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

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2::Slabel, ss.str());
// 	Mout.setLocal(locx, F.Sdag(locy));
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2::
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

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {1}, HubbardSU2::Slabel, ss.str());
// 	Mout.setLocal({locx1,locx2}, {F.S(locy1),F.Sdag(locy2)});
// 	return Mout;
// }

} //end namespace VMPS::models

#endif
