#ifndef KONDOMODELSU2XU1_H_
#define KONDOMODELSU2XU1_H_

#include "ParamHandler.h" // from HELPERS

#include "bases/SpinBaseSU2xU1.h"
#include "bases/FermionBaseSU2xU1.h"
#include "Mpo.h"

namespace VMPS
{
/** 
 * \class KondoSU2xU1
 * \ingroup Kondo
 *
 * \brief Kondo Model
 *
 * MPO representation of
 * \f[
 * H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
 * - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
 * \f].
 *
 * where further parameters from HubbardSU2xU1 and HeisenbergSU2 are possible.
 * \note The default variable settings can be seen in \p KondoSU2xU1::defaults.
 * \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
 * \note If the nnn-hopping is positive, the ground state energy is lowered.
 * \warning \f$J<0\f$ is antiferromagnetic
 */
class KondoSU2xU1 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,double>
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	///@{
	KondoSU2xU1 ():Mpo() {};
	KondoSU2xU1 (const size_t &L, const vector<Param> &params);
	///@}

	///@{
	/**Push params for DMRG algorithms via these functions to an instance of DmrgSolver.*/
	DMRG::CONTROL::DYN get_DynParam(const vector<Param> &params={}) const;
	DMRG::CONTROL::GLOB get_GlobParam(const vector<Param> &params={}) const;
	///@}

	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local spin-operators are received
	 * \param F : Base class from which the local fermion-operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	 */
	static HamiltonianTermsXd<Symmetry> set_operators (const SpinBase<Symmetry> &B, const FermionBase<Symmetry> &F,
	                                                    const ParamHandler &P, size_t loc=0);
				
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;

	///@{
	Mpo<Symmetry> c (std::size_t locx, std::size_t locy=0);
	Mpo<Symmetry> cdag (std::size_t locx, std::size_t locy=0);
	///@}
	
	///@{
	Mpo<Symmetry> n (std::size_t locx, std::size_t locy=0);
	Mpo<Symmetry> d (std::size_t locx, std::size_t locy=0);
	Mpo<Symmetry> cdagc (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	///@}
	
	///@{
	Mpo<Symmetry> ninj (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	//*\warning not implemented
	Mpo<Symmetry> cdagcdagcc (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	///@}
	
	///@{
	Mpo<Symmetry> SimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	Mpo<Symmetry> SsubSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	Mpo<Symmetry> SimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	///@}

	///@{ \warning not implemented
	Mpo<Symmetry> SimpSsubSimpSimp (std::size_t loc1x, std::size_t loc2x,
									 std::size_t loc3x, std::size_t loc4x,
									 std::size_t loc1y=0, std::size_t loc2y=0, std::size_t loc3y=0, std::size_t loc4y=0);
	Mpo<Symmetry> SimpSsubSimpSsub (std::size_t loc1x, std::size_t loc2x,
									 std::size_t loc3x, std::size_t loc4x,
									 std::size_t loc1y=0, std::size_t loc2y=0, std::size_t loc3y=0, std::size_t loc4y=0);
	///@}

	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;

protected:

	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> KondoSU2xU1::defaults =
{
	{"t",1.}, {"tPerp",0.},{"tPrime",0.},
	{"J",-1.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"mu",0.}, {"t0",0.},
	{"D",2ul},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::map<string,std::any> KondoSU2xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"eps_svd",1.e-7}, {"lim_for_alpha",20ul},
	{"Dincr_abs", 10ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",60ul}, {"max_halfsweeps",6ul},
	{"Dinit",100ul}, {"Qinit",50ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-8}, {"tol_state",1.e-8},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KondoSU2xU1::
KondoSU2xU1 (const size_t &L, const vector<Param> &params)
	:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", true)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)));
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);
		
		Terms[l] = set_operators(B[l],F[l],P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

DMRG::CONTROL::GLOB KondoSU2xU1::
get_GlobParam(const vector<Param> &params) const
{
	ParamHandler P(params,sweep_defaults);
	DMRG::CONTROL::GLOB out;
	out.min_halfsweeps = P.get<size_t>("min_halfsweeps");
	out.max_halfsweeps = P.get<size_t>("max_halfsweeps");
	out.Dinit          = P.get<size_t>("Dinit");
	out.Qinit          = P.get<size_t>("Qinit");
	out.Dlimit         = P.get<size_t>("Dlimit");
	out.tol_eigval     = P.get<double>("tol_eigval");
	out.tol_state      = P.get<double>("tol_state");
	out.savePeriod     = P.get<size_t>("savePeriod");
	out.CONVTEST       = P.get<DMRG::CONVTEST::OPTION>("CONVTEST");
	out.CALC_S_ON_EXIT = P.get<bool>("CALC_S_ON_EXIT");
	return out;
}

DMRG::CONTROL::DYN KondoSU2xU1::
get_DynParam(const vector<Param> &params) const
{
	ParamHandler P(params,sweep_defaults);
	DMRG::CONTROL::DYN out;
	double tmp1          = P.get<double>("max_alpha");
	size_t lim_for_alpha = P.get<size_t>("lim_for_alpha");
	out.max_alpha_rsvd   = [tmp1] (size_t i) { return (i<=lim_for_alpha)? tmp1:0; };
	tmp1                 = P.get<double>("min_alpha");
	out.min_alpha_rsvd   = [tmp1] (size_t i) { return tmp1; };
	tmp1                 = P.get<double>("eps_svd");
	out.eps_svd          = [tmp1] (size_t i) { return tmp1; };
	size_t tmp2          = P.get<size_t>("Dincr_abs");
	out.Dincr_abs        = [tmp2] (size_t i) { return tmp2; };
	tmp2                 = P.get<size_t>("Dincr_per");
	out.Dincr_per        = [tmp2] (size_t i) { return tmp2; };
	tmp1                 = P.get<double>("Dincr_rel");
	out.Dincr_rel        = [tmp1] (size_t i) { return tmp1; };
	tmp2                 = P.get<size_t>("min_Nsv");
	out.min_Nsv          = [tmp2] (size_t i) { return tmp2; };
	int tmp3             = P.get<int>("max_Nrich");
	out.max_Nrich	     = [tmp3] (size_t i) { return tmp3; };
	return out;
}

bool KondoSU2xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[1],2); //electrons have spin 1/2
	frac Smax = S_elec;
	for (size_t l=0; l<N_sites; ++l) { Smax+=static_cast<int>(B[l].orbitals())*frac(B[l].get_D()-1,2); } //add local spins to Smax
	
	frac S_tot(qnum[0]-1,2);
	if (Smax.denominator()==S_tot.denominator() and S_tot<=Smax and qnum[0]<=2*static_cast<int>(this->N_phys) and qnum[0]>0) {return true;}
	else {return false;}
}

HamiltonianTermsXd<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
set_operators (const SpinBase<Symmetry> &B, const FermionBase<Symmetry> &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;

	frac S = frac(B.get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};

	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F.orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F.orbitals(),loc);
	save_label(Vlabel);
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			auto Otmp = OperatorType::prod(OperatorType::outerprod(B.Id(),F.sign(),{1,0}),OperatorType::outerprod(B.Id(),F.c(j),{2,-1}),{2,-1});
			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
											 OperatorType::outerprod(B.Id(),F.cdag(i),{2,+1}).plain<double>(),
											 Otmp.plain<double>()));
			Otmp = OperatorType::prod(OperatorType::outerprod(B.Id(),F.sign(),{1,0}),OperatorType::outerprod(B.Id(),F.cdag(j),{2,+1}),{2,+1});
			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
											 OperatorType::outerprod(B.Id(),F.c(i),{2,-1}).plain<double>(),
											 Otmp.plain<double>()));
			// Use this version:
			// auto cF    = OperatorType::prod(F.c(),   F.sign(),{2,-1});
			// auto cdagF = OperatorType::prod(F.cdag(),F.sign(),{2,+1});
			// /**\todo: think about crazy fermionic signs here:*/
			// Terms.nextn.push_back(make_tuple(+tPrime.x*sqrt(2.), cdagF.plain<double>(), F.c().plain<double>(),    F.sign().plain<double>()));
			// Terms.nextn.push_back(make_tuple(+tPrime.x*sqrt(2.), cF.plain<double>()   , F.cdag().plain<double>(), F.sign().plain<double>()));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j),
											 OperatorType::outerprod(B.Id(),F.n(i),{1,0}).plain<double>(),
											 OperatorType::outerprod(B.Id(),F.n(j),{1,0}).plain<double>()));
		}
	}

	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);

	if (tPrime.x != 0.)
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");

		auto Otmp = OperatorType::prod(F.sign(),F.c(),{2,-1});
		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.),
										 OperatorType::outerprod(B.Id(),F.cdag(),{2,+1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),Otmp,{2,-1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),F.sign(),{1,0}).plain<double>()));
		Otmp = OperatorType::prod(F.sign(),F.cdag(),{2,+1});
		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.),
										 OperatorType::outerprod(B.Id(),F.c(),{2,-1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),Otmp,{2,+1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),F.sign(),{1,0}).plain<double>()));
	}

	// local terms
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// V⟂
	param0d Vperp = P.fill_array0d<double>("Vperp","Vperp",loc);
	save_label(Vperp.label);
	
	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F.orbitals(),loc);
	save_label(Ulabel);
	
	// mu
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	save_label(mulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F.orbitals(),loc);
	save_label(t0label);

	OperatorType KondoHamiltonian({1,0},B.get_basis().combine(F.get_basis()));

	//set Hubbard part of Kondo Hamiltonian
	KondoHamiltonian = OperatorType::outerprod(B.Id(),F.HubbardHamiltonian(Uorb,t0orb-muorb,tPerp.x,Vperp.x,0., P.get<bool>("CYLINDER")),{1,0});

	//set Heisenberg part of Hamiltonian
	KondoHamiltonian += OperatorType::outerprod(B.HeisenbergHamiltonian(0.,P.get<bool>("CYLINDER")),F.Id(),{1,0});

	// Kondo-J
	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F.orbitals(),loc);
	save_label(Jlabel);

	//set interaction part of Hamiltonian.
	for (int i=0; i<F.orbitals(); ++i)
	{
		KondoHamiltonian += -Jorb(i)*std::sqrt(3.)*OperatorType::outerprod(B.Sdag(i),F.S(i),{1,0});
	}

	Terms.name = "Kondo SU(2)⊗U(1)";
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
	
	return Terms;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
c (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<B[locx].dim()*F[locx].dim());
	std::stringstream ss;
	ss << "c(" << locx << "," << locy << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	Mout.setLocal(locx, OperatorType::outerprod(B[locx].Id(),F[locx].c(locy),{2,-1}).plain<double>(),
				        OperatorType::outerprod(B[locx].Id(),F[locx].sign(),{1,0}).plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
cdag (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<B[locx].dim()*F[locx].dim());
	std::stringstream ss;
	ss << "c†(" << locx << "," << locy << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	Mout.setLocal(locx, OperatorType::outerprod(B[locx].Id(),F[locx].cdag(locy),{2,+1}).plain<double>(),
				  OperatorType::outerprod(B[locx].Id(),F[locx].sign(),{1,0}).plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
cdagc (size_t loc1x, size_t loc2x, size_t loc1y, size_t loc2y)
{
	//Not well implemented.. the same problems with sign as in the Hubbard model.
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	stringstream ss;
	ss << "c†(" << loc1x << "," << loc1y << ")" << "c(" << loc2x << "," << loc2y << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto cdag = OperatorType::outerprod(B[loc1x].Id(),F[loc1x].cdag(loc1y),{2,+1});
	auto c    = OperatorType::outerprod(B[loc2x].Id(),F[loc2x].c(loc2y),{2,-1});
	auto sign = OperatorType::outerprod(B[loc2x].Id(),F[loc2x].sign(),{1,0});
	if(loc1x == loc2x)
	{
		auto product = std::sqrt(2.)*OperatorType::prod(cdag,c,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product.plain<double>());
		return Mout;
	}
	else if(loc1x<loc2x)
	{

		Mout.setLocal({loc1x, loc2x}, {-sqrt(2.)*cdag.plain<double>(), OperatorType::prod(sign,c,{2,-1}).plain<double>()}, sign.plain<double>());
		return Mout;
	}
	else if(loc1x>loc2x)
	{

		Mout.setLocal({loc1x, loc2x}, {-sqrt(2.)*OperatorType::prod(sign,cdag,{2,+1}).plain<double>(), c.plain<double>()}, sign.plain<double>());
		return Mout;
	}
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
n (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<B[locx].dim()*F[locx].dim());
	std::stringstream ss;
	ss << "occ(" << locx << "," << locy << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto n = OperatorType::outerprod(B[locx].Id(),F[locx].n(locy),Symmetry::qvacuum());
	Mout.setLocal(locx, n.plain<double>());

	return Mout;	
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
d (std::size_t locx, std::size_t locy)
{
	assert(locx<N_sites and locy<B[locx].dim()*F[locx].dim());
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto d = OperatorType::outerprod(B[locx].Id(),F[locx].d(locy),Symmetry::qvacuum());
	Mout.setLocal(locx, d.plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
ninj (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "n(" << loc1x << "," << loc1y << ")"  << "n(" << loc2x << "," << loc2y << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto n1 = OperatorType::outerprod(B[loc1x].Id(),F[loc1x].n(loc1y),Symmetry::qvacuum());
	auto n2 = OperatorType::outerprod(B[loc2x].Id(),F[loc2x].n(loc2y),Symmetry::qvacuum());

	if(loc1x == loc2x)
	{
		auto product = OperatorType::prod(n1,n2,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {n1.plain<double>(), n2.plain<double>()});
		return Mout;
	}

	return Mout;
}

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cdagcdagcc (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "η†(" << loc1x << "," << loc1y << ")" << "η(" << loc2x << "," << loc2y << ")";

// 	Mpo<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Etadag = Operator::outerprod(Spins.Id(),F.Etadag(loc1y),{1,2});
// 	auto Eta = Operator::outerprod(Spins.Id(),F.Eta(loc2y),{1,-2});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({loc1x, loc2x}, {Etadag, Eta}, {{1,2},{1,-2}});
// 		return Mout;
// 	}
// }

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
SimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "S(" << loc1x << "," << loc1y << ")" << "S(" << loc2x << "," << loc2y << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto Sdag = OperatorType::outerprod(B[loc1x].Sdag(loc1y),F[loc1x].Id(),{3,0});
	auto S = OperatorType::outerprod(B[loc2x].S(loc2y),F[loc1x].Id(),{3,0});

	if(loc1x == loc2x)
	{
		auto product = sqrt(3.)*OperatorType::prod(Sdag,S,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {sqrt(3.)*Sdag.plain<double>(), S.plain<double>()});
		return Mout;
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
SsubSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "s(" << loc1x << "," << loc1y << ")" << "s(" << loc2x << "," << loc2y << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto Sdag = OperatorType::outerprod(B[loc1x].Id(),F[loc1x].Sdag(loc1y),{3,0});
	auto S = OperatorType::outerprod(B[loc2x].Id(),F[loc1x].S(loc2y),{3,0});

	if(loc1x == loc2x)
	{
		auto product = sqrt(3.)*OperatorType::prod(Sdag,S,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {sqrt(3.)*Sdag.plain<double>(), S.plain<double>()});
		return Mout;
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
SimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
{
	assert(loc1x<this->N_sites and loc2x<this->N_sites);
	std::stringstream ss;
	ss << "S(" << loc1x << "," << loc1y << ")" << "s(" << loc2x << "," << loc2y << ")";

	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }

	auto Sdag = OperatorType::outerprod(B[loc1x].Sdag(loc1y),F[loc1x].Id(),{3,0});
	auto S = OperatorType::outerprod(B[loc2x].Id(),F[loc1x].S(loc2y),{3,0});

	if(loc1x == loc2x)
	{
		auto product = sqrt(3.)*OperatorType::prod(Sdag,S,Symmetry::qvacuum());
		Mout.setLocal(loc1x,product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({loc1x, loc2x}, {sqrt(3.)*Sdag.plain<double>(), S.plain<double>()});
		return Mout;
	}
	return Mout;
}

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsubSimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc3x, std::size_t loc4x,
//                   std::size_t loc1y, std::size_t loc2y, std::size_t loc3y, std::size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(Mpo<2>::qloc[loc2x].size()/F.dim(), Mpo<2>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(S.Scomp(SOP4,loc4y),IdSub)});
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsubSimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc3x, std::size_t loc4x,
// 				  std::size_t loc1y, std::size_t loc2y, std::size_t loc3y, std::size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(Mpo<2>::qloc[loc2x].size()/F.dim(), Mpo<2>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP4,loc4y))}
// 		);
// 	return Mout;
// }


// bool KondoSU2xU1::
// validate (qType qnum) const
// {
// 	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
// 	return (qnum[0]-1+N_legs*Sx2*imploc.size())%2 == qnum[1]%2;
// }

} //end namespace VMPS

#endif
