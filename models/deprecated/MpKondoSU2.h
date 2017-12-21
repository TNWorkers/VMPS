#ifndef KONDOMODELSU2_H_
#define KONDOMODELSU2_H_

#include "spins/BaseSU2.h"
#include "fermions/BaseSU2.h"
#include "MultipedeQ.h"
#include "MpoQ.h"
#include "MpHubbardSU2.h"

namespace VMPS::models
{
/** \class KondoSU2
  * \ingroup Models
  *
  * \brief Kondo Model
  *
  * MPO representation of
  \f$
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
  \f$.
  *
  \note Take use only of the Spin SU(2) symmetry.
  \note If the nnn-hopping is positive, the ground state energy is lowered.
  \warning \f$J<0\f$ is antiferromagnetic
  */
class KondoSU2 : public MpoQ<Sym::SU2<double>,double>
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
	KondoSU2 ():MpoQ() {};
	
	/**Constructs a Kondo Lattice Model on a N-ladder.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param mu_input : \f$\mu\f$ (chemical potential)
	\param Ly_input : chain width
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoSU2 (std::size_t Lx_input, double J_input=-1., double mu_input=0., std::size_t Ly_input=1, double tPrime_input=0.,
	            double U_input=0., bool CALC_SQUARE=false, std::size_t D_input=2);

	/**Constructs a Kondo Impurity Model on a N-ladder (aka a diluted Kondo Model) using initializer lists for the set of impurities.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param mu_input : \f$\mu\f$ (chemical potential)
	\param imploc_input : list with locations of the impurities
	\param Ly_input : chain width
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoSU2 (std::size_t Lx_input, double J_input, double mu_input, std::initializer_list<std::size_t> imploc_input,
				 std::size_t Ly_input=1, double tPrime_input=0., double U_input=0., bool CALC_SQUARE=true, std::size_t D_input=2);

	/**Constructs a Kondo Impurity Model on a N-ladder (aka a diluted Kondo Model) using vectors for the set of impurities.
	\param Lx_input : chain length
	\param J_input : \f$J\f$
	\param mu_input : \f$\mu\f$ (chemical potential)
	\param imploc_input : list with locations of the impurities
	\param Ly_input : chain width
	\param tPrime_input : \f$t^{\prime}\f$ next nearest neighbour (nnn) hopping. \f$t^{\prime}>0\f$ is common sign.
	\param U_input : \f$U\f$ (local Hubbard interaction)
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	\param D_input : \f$2S+1\f$ (impurity spin)*/
	KondoSU2 (std::size_t Lx_input, double J_input, double mu_input, std::vector<std::size_t> imploc_input,
				 std::size_t Ly_input=1, double tPrime_input=0., double U_input=0., bool CALC_SQUARE=true, std::size_t D_input=2);

	/**Determines the operators of the Hamiltonian. Made static to be called from other classes, e.g. TransverseKondoModel.
	\param F : the FermionBase class where the local Fermion operators are pulled from
	\param S : the SpinBase class where the local Spin operators are pulled from
	\param J : \f$J\f$
	\param tInter: hopping matrix for hopping from site \f$i\f$ to \f$i+1\f$ from orbital \f$m\f$ to \f$m^{\prime}\f$
	\param tIntra: hopping inside the super site.
	\param tPrime : \f$t'\f$
	\param U : \f$U\f$
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const fermions::BaseSU2<> &F, const spins::BaseSU2<> &S, 
													   double J, double mu, Eigen::MatrixXd tInter, double tIntra, double tPrime=0., double U=0.);

	static const std::vector<std::vector<qType> > getqloc (const fermions::BaseSU2<> &F, const spins::BaseSU2<> &S,
														   const std::vector<std::size_t>& imps, const std::size_t& length);
	static const std::vector<qType> getqOp();
	static const std::vector<std::vector<Index> > getqlocDeg(const fermions::BaseSU2<> &F, const spins::BaseSU2<> &S,
															 const std::vector<std::size_t>& imps, const std::size_t& length);

	/**Makes half-integers in the output for the magnetization quantum number.*/
	static std::string N_halveM (qType qnum);
	
	/**Labels the conserved quantum numbers as "S".*/
	static const std::array<std::string,1> Slabel;
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                                StateXd;
	typedef MpsQ<Symmetry,std::complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,KondoSU2,double,false>          Solver;
	typedef MpsQCompressor<Symmetry,double,double>               CompressorXd;
	typedef MpsQCompressor<Symmetry,std::complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry,double>                                OperatorXd;
	typedef MpoQ<Symmetry,std::complex<double> >                 OperatorXcd;
	///@}
		
	/**Validates whether a given \p qnum is a valid \p S for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;

	///@{
	MpoQ<Symmetry> n (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> d (std::size_t locx, std::size_t locy=0);
	MpoQ<Symmetry> ninj (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	MpoQ<Symmetry> SimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	MpoQ<Symmetry> SsubSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	MpoQ<Symmetry> SimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	MpoQ<Symmetry> SimpSsubSimpSimp (std::size_t loc1x, std::size_t loc2x,
									 std::size_t loc3x, std::size_t loc4x,
									 std::size_t loc1y=0, std::size_t loc2y=0, std::size_t loc3y=0, std::size_t loc4y=0);
	MpoQ<Symmetry> SimpSsubSimpSsub (std::size_t loc1x, std::size_t loc2x,
									 std::size_t loc3x, std::size_t loc4x,
									 std::size_t loc1y=0, std::size_t loc2y=0, std::size_t loc3y=0, std::size_t loc4y=0);
	///@}
	
protected:
	
	double J=-1., mu=0., t=1., tPrime=0., U=0.;
	std::size_t D=2;
	
	std::vector<std::size_t> imploc;
	fermions::BaseSU2<> F; spins::BaseSU2<> Spins;	
};

const std::array<std::string,Sym::SU2<double>::Nq> KondoSU2::Slabel{"S"};

HamiltonianTermsXd<Sym::SU2<double> > KondoSU2::
set_operators (const fermions::BaseSU2<> &F, const spins::BaseSU2<> &Spins, double J, double mu, Eigen::MatrixXd tInter, double tIntra, double tPrime, double U)
{
	assert(F.orbitals() == 1 and "More than one orbital in the Kondo-Model is not working until now.");
	HamiltonianTermsXd<Symmetry> Terms;
	
	Operator KondoHamiltonian({1},Spins.basis().combine(F.basis()));
		
	//set Hubbard part of Kondo Hamiltonian
	KondoHamiltonian = Operator::outerprod(Spins.Id(),F.HubbardHamiltonian(U,mu,tIntra),{1});
	
	//set Heisenberg part of Hamiltonian
	KondoHamiltonian += Operator::outerprod(Spins.HeisenbergHamiltonian(0.,0.),F.Id(),{1});
	
	//set interaction part of Hamiltonian.
	for (int i=0; i<F.orbitals(); ++i)
	{
		KondoHamiltonian += -J*std::sqrt(3.)*Operator::outerprod(Spins.Sdag(i),F.S(i),{1});
	}
	
	std::cout << Operator::outerprod(Spins.Id(), F.cdag(0), {2}).data().print(false,true) << std::endl;
	//set local interaction
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian));
	//set nearest neighbour term
	for (int legI=0; legI<F.orbitals(); ++legI)
	for (int legJ=0; legJ<F.orbitals(); ++legJ)
	{
		if (tInter(legI,legJ) != 0)
		{
			Terms.tight.push_back(std::make_tuple(tInter(legI,legJ)*std::sqrt(2.),
												  Operator::outerprod(Spins.Id(), F.cdag(legI), {2}),
												  Operator::prod( Operator::outerprod(Spins.Id(), F.sign(),{1}),
																  Operator::outerprod(Spins.Id(),F.c(legJ), {2}), {2})));
			Terms.tight.push_back(std::make_tuple(tInter(legI,legJ)*std::sqrt(2.),
												  Operator::outerprod(Spins.Id(), F.c(legI), {2}),
												  Operator::prod( Operator::outerprod(Spins.Id(), F.sign(),{1}),
																  Operator::outerprod(Spins.Id(),F.cdag(legJ), {2}), {2})));
		}
	}

	if (tPrime != 0.)
	{
		//set next nearest neighbour term
		Terms.nextn.push_back(make_tuple(tPrime*std::sqrt(2.),
										 Operator::outerprod(Spins.Id(), F.cdag(0), {2}),
										 Operator::prod( Operator::outerprod(Spins.Id(), F.sign(),{1}),
														 Operator::outerprod(Spins.Id(),F.c(0), {2}), {2}),
										 Operator::outerprod(Spins.Id(), F.sign(),{1})));
		Terms.nextn.push_back(make_tuple(tPrime*std::sqrt(2.),
										 Operator::outerprod(Spins.Id(), F.c(0), {2}),
										 Operator::prod( Operator::outerprod(Spins.Id(), F.sign(),{1}),
														 Operator::outerprod(Spins.Id(),F.cdag(0), {2}), {2}),
										 Operator::outerprod(Spins.Id(), F.sign(),{1})));
	}
	Terms.Id = Operator::outerprod(Spins.Id(),F.Id(),{1});
	return Terms;
}

const std::vector<std::vector<Sym::SU2<double>::qType> > KondoSU2::
getqloc (const fermions::BaseSU2<> &F, const spins::BaseSU2<> &S, const std::vector<std::size_t>& imps, const std::size_t& length)
{
	std::vector<std::vector<qType> > out(length);
	for(std::size_t l=0; l<length; l++)
	{
		if( auto it=std::find(imps.begin(),imps.end(),l) == imps.end() )
		{
			out[l] = HubbardSU2::getqloc(F);
		}
		else
		{
			auto TensorBasis = S.basis().combine(F.basis());
			out[l] = TensorBasis.qloc();
		}
	}
	return out;
}

const std::vector<std::vector<Eigen::Index> > KondoSU2::
getqlocDeg (const fermions::BaseSU2<> &F, const spins::BaseSU2<> &S, const std::vector<std::size_t>& imps, const std::size_t& length)
{
	std::vector<std::vector<Index> > out(length);
	for(std::size_t l=0; l<length; l++)
	{
		if( auto it=std::find(imps.begin(),imps.end(),l) == imps.end() )
		{
			out[l] = HubbardSU2::getqlocDeg(F);
		}
		else
		{
			auto TensorBasis = S.basis().combine(F.basis());
			out[l] = TensorBasis.qlocDeg();
		}
	}
	return out;
}

const std::vector<Sym::SU2<double>::qType> KondoSU2::
getqOp ()
{	
	std::vector<qType> vout(3);
	vout[0] = {1}; //Kondo Interaction and Identity operators
	vout[1] = {2}; //cdag
	vout[2] = {2}; //c
	return vout;
}

KondoSU2::
KondoSU2 (std::size_t Lx_input, double J_input, double mu_input, std::size_t Ly_input, double tPrime_input,
		  double U_input, bool CALC_SQUARE, std::size_t D_input)
	:MpoQ<Symmetry> (Lx_input,Ly_input),
	J(J_input), mu(mu_input), tPrime(tPrime_input), U(U_input), D(D_input)
{
	// assign stuff
	this->Qtot = {1};
	this->qlabel = Slabel;
	this->label = "KondoSU2:";
	this->format = N_halveM;
	
	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");
	
	// initialize member variable imploc
	this->imploc.resize(Lx_input);
	std::iota(this->imploc.begin(), this->imploc.end(), 0);
	
	std::stringstream ss;
	ss << "(J=" << J << ",μ=" << mu << ",t'=" << tPrime << ",U=" << U << ")";
	this->label += ss.str();

	F = fermions::BaseSU2<double>(N_legs);
	Spins = spins::BaseSU2<double>(N_legs,D);
	for(std::size_t l=0; l<this->N_sites; l++)
	{
		this->qloc__[l] = Spins.basis().combine(F.basis());
		qloc[l] = qloc__[l].qloc();
		this->setOpBasis(KondoSU2::getqOp(),l);
	}

	Eigen::MatrixXd tInter(N_legs,N_legs); tInter.setIdentity(); tInter*=0.25;
	std::cout << tInter << std::endl;
	HamiltonianTermsXd<Symmetry> Terms = set_operators(F,Spins,J,mu,tInter,1.,tPrime,U);
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

KondoSU2::
KondoSU2 (std::size_t Lx_input, double J_input, double mu_input, std::vector<std::size_t> imploc_input, std::size_t Ly_input,
			 double tPrime_input, double U_input, bool CALC_SQUARE, std::size_t D_input)
	:MpoQ<Symmetry,double>(Lx_input,Ly_input), J(J_input), mu(mu_input), imploc(imploc_input), D(D_input), tPrime(tPrime_input), U(U_input)
{	
	// assign stuff
	this->Qtot = {1};
	this->qlabel = Slabel;
	this->label = "KondoSU2 (impurity):";
	// this->format = N_halveM;
	
	F = fermions::BaseSU2<double>(N_legs);
	Spins = spins::BaseSU2<double>(N_legs,D);
	
	Eigen::MatrixXd tInter(N_legs,N_legs); tInter.setIdentity();// tInter*=-1.;
		
	// // make a pretty label
	std::stringstream ss;
	ss << "(S=" << frac(D-1,2) << ",J=" << J << ",μ=" << mu << ",imps={";
	for (auto i=0; i<imploc.size(); ++i)
	{
		assert(imploc[i] < this->N_sites and "Invalid impurity location!");
		ss << imploc[i];
		if (i!=imploc.size()-1) {ss << ",";}
	}
	ss << "}";
	ss << ")";
	this->label += ss.str();

	this->setLocBasis(KondoSU2::getqloc(F,Spins,imploc,N_sites));
	// this->setLocBasisDeg(KondoSU2::getqlocDeg(F,Spins,imploc,N_sites));

	for (std::size_t l=0; l<N_sites; l++)
	{
		this->setOpBasis(KondoSU2::getqOp(),l);
	}

	// // create the SuperMatrices
	std::vector<MultipedeQ<4,Symmetry,double,-2> > G(this->N_sites);
	
	for (std::size_t l=0; l<this->N_sites; ++l)
	{
		auto it = std::find(imploc.begin(),imploc.end(),l);
		// got an impurity
		if (it!=imploc.end())
		{			
			std::size_t i = it-imploc.begin();
			if (l==0)
			{
				HamiltonianTermsXd<Symmetry> Terms = set_operators(F,Spins, J, mu, tInter, t, tPrime, U);
				this->Daux = Terms.auxdim();
				auto Gtmp = ::Generator(Terms);
				for (std::size_t nu=0; nu<Gtmp.size(); nu++)
				{
					std::array<Index, 4> offsets = {Gtmp.block[nu].dimension(0)-1, 0, 0, 0};
					std::array<Index, 4> extents = {1, Gtmp.block[nu].dimension(1), Gtmp.block[nu].dimension(2), Gtmp.block[nu].dimension(3)};
					TensorType<4> A = Gtmp.block[nu].slice(offsets,extents);
					G[l].push_back(Gtmp.index[nu],A);
				}
			}
			else if (l==this->N_sites-1)
			{
				HamiltonianTermsXd<Symmetry> Terms = set_operators(F,Spins, J, mu, tInter, t, tPrime, U);
				this->Daux = Terms.auxdim();
				auto Gtmp = ::Generator(Terms);
				for (std::size_t nu=0; nu<Gtmp.size(); nu++)
				{
					std::array<Index, 4> offsets = {0, 0, 0, 0};
					std::array<Index, 4> extents = {Gtmp.block[nu].dimension(0), 1, Gtmp.block[nu].dimension(2), Gtmp.block[nu].dimension(3)};
					TensorType<4> A = Gtmp.block[nu].slice(offsets,extents);
					G[l].push_back(Gtmp.index[nu],A);
				}
			}
			else
			{
				HamiltonianTermsXd<Symmetry> Terms = set_operators(F,Spins, J, mu, tInter, t, tPrime, U);
				G[l] = ::Generator(Terms);
			}
		}
		// no impurity
		else
		{			
			if (l==0)
			{
				HamiltonianTermsXd<Symmetry> Terms = HubbardSU2::set_operators(F,U,0.,mu,tPrime,t);
				this->Daux = Terms.auxdim();
				auto Gtmp = ::Generator(Terms);
				for (std::size_t nu=0; nu<Gtmp.size(); nu++)
				{
					std::array<Index, 4> offsets = {Gtmp.block[nu].dimension(0)-1, 0, 0, 0};
					std::array<Index, 4> extents = {1, Gtmp.block[nu].dimension(1), Gtmp.block[nu].dimension(2), Gtmp.block[nu].dimension(3)};
					TensorType<4> A = Gtmp.block[nu].slice(offsets,extents);
					G[l].push_back(Gtmp.index[nu],A);
				}
			}
			else if (l==this->N_sites-1)
			{
				HamiltonianTermsXd<Symmetry> Terms = HubbardSU2::set_operators(F,U,0.,mu,tPrime,t);
				this->Daux = Terms.auxdim();
				auto Gtmp = ::Generator(Terms);
				for (std::size_t nu=0; nu<Gtmp.size(); nu++)
				{
					std::array<Index, 4> offsets = {0, 0, 0, 0};
					std::array<Index, 4> extents = {Gtmp.block[nu].dimension(0), 1, Gtmp.block[nu].dimension(2), Gtmp.block[nu].dimension(3)};
					TensorType<4> A = Gtmp.block[nu].slice(offsets,extents);
					G[l].push_back(Gtmp.index[nu],A);
				}
			}
			else
			{
				HamiltonianTermsXd<Symmetry> Terms = HubbardSU2::set_operators(F,U,0.,mu,tPrime,t);
				this->Daux = Terms.auxdim();
				G[l] = ::Generator(Terms);
			}
		}
	}
	
	this->construct(G);
	
	if (CALC_SQUARE == true)
	{
		// this->construct(Gsq, this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

KondoSU2::
KondoSU2 (std::size_t Lx_input, double J_input, double mu_input, std::initializer_list<std::size_t> imploc_input, std::size_t Ly_input,
			 double tPrime_input, double U_input, bool CALC_SQUARE, std::size_t D_input)
	:KondoSU2(Lx_input, J_input, mu_input, std::vector<std::size_t>(begin(imploc_input),end(imploc_input)),
			  Ly_input, tPrime_input, U_input, CALC_SQUARE, D_input)
{}

std::string KondoSU2::
N_halveM (qType qnum)
{
	std::stringstream ss;
	ss << "not implemented";
	return ss.str();
}

MpoQ<Sym::SU2<double> > KondoSU2::
d (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Sym::SU2<double> > Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

	if (auto it=std::find(imploc.begin(),imploc.end(),locx) == imploc.end())
	{
		Mout.label = ss.str();
		Mout.setQtarget(Symmetry::qvacuum());
		Mout.qlabel = HubbardSU2::Slabel;
		Mout.setLocal(locx, F.d(locy), Symmetry::qvacuum());
		return Mout;
	}
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = KondoSU2::Slabel;
	auto d = Operator::outerprod(Spins.Id(),F.d(locy),Symmetry::qvacuum());
	Mout.setLocal({locx}, {d}, {Symmetry::qvacuum()});
	return Mout;
}

MpoQ<Sym::SU2<double> > KondoSU2::
n (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	std::stringstream ss;
	ss << "occ(" << locx << "," << locy << ")";
	
	MpoQ<Sym::SU2<double> > Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }
	
	if (auto it=std::find(imploc.begin(),imploc.end(),locx) == imploc.end())
	{
		std::cout << "not here please" << std::endl;
		Mout.label = ss.str();
		Mout.setQtarget(Symmetry::qvacuum());
		Mout.qlabel = HubbardSU2::Slabel;
		Mout.setLocal(locx, F.n(locy), Symmetry::qvacuum());
		return Mout;
	}
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = KondoSU2::Slabel;
	auto n = Operator::outerprod(Spins.Id(),F.n(locy),Symmetry::qvacuum());
	Mout.setLocal({locx}, {n}, {Symmetry::qvacuum()});
	return Mout;
}

MpoQ<Sym::SU2<double> > KondoSU2::
ninj (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "n(" << loc1x << "," << loc1y << ")"  << "n(" << loc2x << "," << loc2y << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

	auto n = Operator::outerprod(Spins.Id(),F.n(loc2y),Symmetry::qvacuum());
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = KondoSU2::Slabel;
	if(loc1x == loc2x)
	{
		auto product = Operator::prod(n,n,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {n, n}, {Symmetry::qvacuum(),Symmetry::qvacuum()});
		return Mout;
	}

	return Mout;
}

MpoQ<Sym::SU2<double> > KondoSU2::
SimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "S(" << loc1x << "," << loc1y << ")" << "S(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

	auto Sdag = Operator::outerprod(Spins.Sdag(loc1y),F.Id(),{3,0});
	auto S = Operator::outerprod(Spins.S(loc2y),F.Id(),{3,0});
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = KondoSU2::Slabel;
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


MpoQ<Sym::SU2<double> > KondoSU2::
SsubSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "s(" << loc1x << "," << loc1y << ")" << "s(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

	auto Sdag = Operator::outerprod(Spins.Id(),F.Sdag(loc1y),{3,0});
	auto S = Operator::outerprod(Spins.Id(),F.S(loc2y),{3,0});
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = KondoSU2::Slabel;
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

MpoQ<Sym::SU2<double> > KondoSU2::
SimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "S(" << loc1x << "," << loc1y << ")" << "s(" << loc2x << "," << loc2y << ")";

	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

	auto Sdag = Operator::outerprod(Spins.Sdag(loc1y),F.Id(),{3,0});
	auto S = Operator::outerprod(Spins.Id(),F.S(loc2y),{3,0});
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = KondoSU2::Slabel;
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

// MpoQ<Sym::SU2<double> > KondoSU2::
// SimpSsubSimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc3x, std::size_t loc4x,
//                   std::size_t loc1y, std::size_t loc2y, std::size_t loc3y, std::size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/F.dim(), MpoQ<2>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(S.Scomp(SOP4,loc4y),IdSub)});
// 	return Mout;
// }

// MpoQ<Sym::SU2<double> > KondoSU2::
// SimpSsubSimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc3x, std::size_t loc4x,
// 				  std::size_t loc1y, std::size_t loc2y, std::size_t loc3y, std::size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/F.dim(), MpoQ<2>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP4,loc4y))}
// 		);
// 	return Mout;
// }

bool KondoSU2::
validate (qType qnum) const
{
	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
	return (qnum[0]-1+N_legs*Sx2*imploc.size())%2 == qnum[1]%2;
}

} //end namespace VMPS::models

#endif
