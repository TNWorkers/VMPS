#ifndef KONDOMODELSU2XU1_H_
#define KONDOMODELSU2XU1_H_

//include "ParamHandler.h" // from HELPERS
#include "ParamReturner.h"
#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"
#include "bases/SpinBaseSU2xU1.h"
#include "bases/FermionBaseSU2xU1.h"
//include "Mpo.h"
//include "models/HubbardSU2xU1.h"
#include "models/KondoObservables.h"

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
class KondoSU2xU1 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,double>, public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(KondoSU2xU1)
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	///@{
	KondoSU2xU1 (): Mpo(), ParamReturner(KondoSU2xU1::sweep_defaults) {};
	KondoSU2xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{1,N};};
	static qarray<2> polaron (int L, int N=0) {assert(N%2==0); return qarray<2>{L+1,N};};
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local spin-operators are received
	 * \param F : Base class from which the local fermion-operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	 */
	static HamiltonianTermsXd<Symmetry> set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F,
	                                                    const ParamHandler &P, size_t loc=0);
	
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;
	
	///@{
	Mpo<Symmetry> c (size_t locx, size_t locy=0, double factor=1.);
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0, double factor=sqrt(2.));
	Mpo<Symmetry> cc (size_t locx, size_t locy=0);
	Mpo<Symmetry> cdagcdag (size_t locx, size_t locy=0);
	///@}
	
	///@{
	Mpo<Symmetry> n (size_t locx, size_t locy=0);
	Mpo<Symmetry> d (size_t locx, size_t locy=0);
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> ccdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
	///@{
	Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	//*\warning not implemented
//	Mpo<Symmetry> cdagcdagcc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
	///@{
	Mpo<Symmetry> Simp (size_t locx, size_t locy=0, double factor=1.);
	Mpo<Symmetry> Simpdag (size_t locx, size_t locy=0, double factor=sqrt(3.));
	Mpo<Symmetry> Ssub (size_t locx, size_t locy=0, double factor=1.);
	Mpo<Symmetry> Ssubdag (size_t locx, size_t locy=0, double factor=sqrt(3.));
	// for compatibility:
	Mpo<Symmetry> S (size_t locx, size_t locy=0, double factor=1.) {return Simp(locx,locy,factor);};
	Mpo<Symmetry> Sdag (size_t locx, size_t locy=0, double factor=sqrt(3.)) {return Simpdag(locx,locy,factor);};
	///@}
	
	///@{
	Mpo<Symmetry> SimpSimp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> SsubSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> SimpSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
//	///@{ \warning not implemented
//	Mpo<Symmetry> SimpSsubSimpSimp (size_t locx1, size_t locx2,
//									 size_t loc3x, size_t loc4x,
//									 size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0);
//	Mpo<Symmetry> SimpSsubSimpSsub (size_t locx1, size_t locx2,
//									 size_t loc3x, size_t loc4x,
//									 size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0);
//	///@}
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >
	make_local (KONDO_SUBSYSTEM SUBSYS, 
	            string name, 
	            size_t locx, size_t locy, 
	            const OperatorType &Op, 
	            double factor,
	            bool FERMIONIC, bool HERMITIAN) const;
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >
	make_corr (KONDO_SUBSYSTEM SUBSYS,
	           string name1, string name2, 
	           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
	           const OperatorType &Op1, const OperatorType &Op2, 
	           qarray<Symmetry::Nq> Qtot, 
	           double factor,
	           bool BOTH_HERMITIAN);
	
	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const map<string,any> KondoSU2xU1::defaults =
{
	{"t",1.}, {"tPrime",0.}, {"tRung",0.},
	{"J",1.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"mu",0.}, {"t0",0.},
	{"D",2ul}, {"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

//const map<string,any> KondoSU2xU1::sweep_defaults = 
//{
//	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",20ul}, {"eps_svd",1.e-7},
//	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
//	{"min_Nsv",0ul}, {"max_Nrich",-1},
//	{"max_halfsweeps",30ul}, {"min_halfsweeps",6ul},
//	{"Dinit",80ul}, {"Qinit",30ul}, {"Dlimit",1000ul},
//	{"tol_eigval",1.e-7}, {"tol_state",1.e-6},
//	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_HSQ}
//};

const map<string,any> VMPS::KondoSU2xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",6ul},
	{"Dinit",5ul}, {"Qinit",15ul}, {"Dlimit",100ul},
	{"tol_eigval",1.e-7}, {"tol_state",1.e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KondoSU2xU1::
KondoSU2xU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 ParamReturner(KondoSU2xU1::sweep_defaults)
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
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(B,F,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

bool KondoSU2xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[1],2); //electrons have spin 1/2
	frac Smax = S_elec;
	for (size_t l=0; l<N_sites; ++l) { Smax+=static_cast<int>(B[l].orbitals())*frac(B[l].get_D()-1,2); } //add local spins to Smax
	
	frac S_tot(qnum[0]-1,2);
	if (Smax.denominator()==S_tot.denominator() and S_tot<=Smax and qnum[0]<=2*2*static_cast<int>(this->N_phys) and qnum[0]>0) {return true;}
	else {return false;}
}

HamiltonianTermsXd<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	
	frac S = frac(B[loc].get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	size_t lp1 = (loc+1)%F.size();
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
	save_label(Vlabel);
	
	for (int i=0; i<F[loc].orbitals(); ++i)
	for (int j=0; j<F[lp1].orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			// auto Otmp = OperatorType::prod(OperatorType::outerprod(B.Id(),F.sign(),{1,0}),OperatorType::outerprod(B.Id(),F.c(i),{2,-1}),{2,-1});
			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
			// 								 OperatorType::outerprod(B.Id(),F.cdag(i),{2,+1}).plain<double>(),
			// 								 Otmp.plain<double>()));
			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id(),F.sign(),{1,0}),OperatorType::outerprod(B.Id(),F.cdag(i),{2,+1}),{2,+1});
			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
			// 								 OperatorType::outerprod(B.Id(),F.c(i),{2,-1}).plain<double>(),
			// 								 Otmp.plain<double>()));
			
			// Use this version:
			// auto cdagF = OperatorType::prod(F.cdag(i),F.sign(),{2,+1});
			// auto cF    = OperatorType::prod(F.c(i),   F.sign(),{2,-1});
			// Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.), cdagF.plain<double>(), F.c(i).plain<double>()));
			// SU(2) spinors commute on different sites, hence no sign flip here:
			// Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.), cF.plain<double>(),    F.cdag(i).plain<double>()));
			
			auto cF = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].c(i) ,{2,-1}),
			                             OperatorType::outerprod(B[loc].Id(), F[loc].sign(), {1,0}), {2,-1});
			auto cdagF = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].cdag(i), {2,+1}), 
			                                OperatorType::outerprod(B[loc].Id(), F[loc].sign(), {1,0}),{2,+1});
			
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
											 cdagF.plain<double>(),
											 OperatorType::outerprod(B[loc].Id(), F[loc].c(i), {2,-1}).plain<double>()));
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
											 cF.plain<double>(),
											 OperatorType::outerprod(B[loc].Id(), F[loc].cdag(i), {2,+1}).plain<double>()));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j),
											 OperatorType::outerprod(B[loc].Id(), F[loc].n(i), {1,0}).plain<double>(),
											 OperatorType::outerprod(B[loc].Id(), F[loc].n(i), {1,0}).plain<double>()));
		}
	}
	
	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);
	
	if (tPrime.x != 0.)
	{
		assert(F[loc].orbitals() == 1 and "Cannot do a ladder with t'!");
		
		auto cF = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].c(), {2,-1}),
		                             OperatorType::outerprod(B[loc].Id(), F[loc].sign(), {1,0}),{2,-1});
		auto cdagF = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].cdag(), {2,+1}),
		                                OperatorType::outerprod(B[loc].Id(), F[loc].sign(), {1,0}),{2,+1});
		
		Terms.nextn.push_back(make_tuple(+tPrime.x*sqrt(2.),
										 cdagF.plain<double>(),
										 OperatorType::outerprod(B[loc].Id(), F[loc].c(), {2,-1}).plain<double>(),
										 OperatorType::outerprod(B[loc].Id(), F[loc].sign(), {1,0}).plain<double>()));
		Terms.nextn.push_back(make_tuple(+tPrime.x*sqrt(2.),
										 cF.plain<double>(),
										 OperatorType::outerprod(B[loc].Id(), F[loc].cdag(), {2,+1}).plain<double>(),
										 OperatorType::outerprod(B[loc].Id(), F[loc].sign(), {1,0}).plain<double>()));
	}
	
	// local terms
	
	// t⟂
	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(tPerplabel);
	
	// V⟂
	auto [Vrung,Vperp,Vperplabel] = P.fill_array2d<double>("Vrung","V","Vperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Vperplabel);
	
	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);
	
	// mu
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F[loc].orbitals(),loc);
	save_label(mulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F[loc].orbitals(),loc);
	save_label(t0label);
	
	OperatorType KondoHamiltonian({1,0}, B[loc].get_basis().combine(F[loc].get_basis()));
	
	ArrayXXd Jperp    = B[loc].ZeroHopping();
	ArrayXXd Jperpsub = F[loc].ZeroHopping();
	
	//set Hubbard part of Kondo Hamiltonian
	KondoHamiltonian = OperatorType::outerprod(B[loc].Id(), F[loc].HubbardHamiltonian(Uorb,t0orb-muorb,tPerp,Vperp,Jperpsub), {1,0});
	
	//set Heisenberg part of Hamiltonian
	KondoHamiltonian += OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jperp), F[loc].Id(), {1,0});
	
	// Kondo-J
	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
	save_label(Jlabel);
	
	//set interaction part of Hamiltonian.
	for (int i=0; i<F[loc].orbitals(); ++i)
	{
		KondoHamiltonian += Jorb(i)*sqrt(3.) * OperatorType::outerprod(B[loc].Sdag(i), F[loc].S(i), {1,0});
	}
	
	Terms.name = "Kondo";
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
	
	return Terms;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
make_local (KONDO_SUBSYSTEM SUBSYS, 
            string name, 
            size_t locx, size_t locy, 
            const OperatorType &Op, 
            double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	assert(SUBSYS != IMPSUB);
	stringstream ss;
	ss << name << "(" << locx << "," << locy << ")";
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
	OperatorType OpExt, SignExt;
	
	if (SUBSYS == SUB)
	{
		OpExt   = OperatorType::outerprod(B[locx].Id(), Op, Op.Q());
		SignExt = OperatorType::outerprod(B[locx].Id(), F[locx].sign(), Symmetry::qvacuum());
	}
	else if (SUBSYS == IMP)
	{
		assert(!FERMIONIC and "Impurity cannot be fermionic!");
		OpExt = OperatorType::outerprod(Op, F[locx].Id(), Op.Q());
	}
	
	(FERMIONIC)? Mout.setLocal(locx, (factor * pow(-1.,locx+1) * OpExt).plain<double>(), SignExt.plain<double>())
		: Mout.setLocal(locx, (factor * OpExt).plain<double>());
	// (FERMIONIC)? Mout.setLocal(locx, (factor * OpExt).plain<double>(), SignExt.plain<double>())
	// 	: Mout.setLocal(locx, (factor * OpExt).plain<double>());

	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
make_corr (KONDO_SUBSYSTEM SUBSYS,
           string name1, string name2, 
           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
           const OperatorType &Op1, const OperatorType &Op2, 
           qarray<Symmetry::Nq> Qtot, 
           double factor,
           bool BOTH_HERMITIAN)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << name1 << "(" << locx1 << "," << locy1 << ")" << name2 << "(" << locx2 << "," << locy2 << ")";
	
	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
	Mpo<Symmetry> Mout(N_sites, Qtot, ss.str(), HERMITIAN);
	for(size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
	OperatorType Op1Ext;
	OperatorType Op2Ext;
	
	if (SUBSYS == SUB)
	{
		Op1Ext = OperatorType::outerprod(B[locx1].Id(), Op1, Op1.Q());
		Op2Ext = OperatorType::outerprod(B[locx2].Id(), Op2, Op2.Q());
	}
	else if (SUBSYS == IMP)
	{
		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), Op1.Q());
		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), Op2.Q());
	}
	else if (SUBSYS == IMPSUB)
	{
		Op2Ext = OperatorType::outerprod(Op1, F[locx1].Id(), Op1.Q());
		Op1Ext = OperatorType::outerprod(B[locx2].Id(), Op2, Op2.Q());
	}
	
	if (locx1 == locx2)
	{
		auto LocProd = OperatorType::prod(Op1Ext, Op2Ext, Qtot);
		Mout.setLocal(locx1, factor * LocProd.plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {factor * Op1Ext.plain<double>(), Op2Ext.plain<double>()});
	}
	
	return Mout;
}

//-----------------------------------------

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
n (size_t locx, size_t locy)
{
	return make_local(SUB, "n", locx,locy, F[locx].n(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
d (size_t locx, size_t locy)
{
	return make_local(SUB, "d", locx,locy, F[locx].d(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
c (size_t locx, size_t locy, double factor)
{
	return make_local(SUB, "c", locx,locy, F[locx].c(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
cdag (size_t locx, size_t locy, double factor)
{
	return make_local(SUB, "c†", locx,locy, F[locx].cdag(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
cc (size_t locx, size_t locy)
{
	stringstream ss;
	ss << "c" << UP << "c" << DN;
	return make_local(SUB, ss.str(), locx,locy, F[locx].Eta(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
cdagcdag (size_t locx, size_t locy)
{
	stringstream ss;
	ss << "c†" << UP << "c†" << DN;
	return make_local(SUB, ss.str(), locx,locy, F[locx].Etadag(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
Ssub (size_t locx, size_t locy, double factor)
{
	return make_local(SUB, "Ssub", locx,locy, F[locx].S(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
Ssubdag (size_t locx, size_t locy, double factor)
{
	return make_local(SUB, "Ssub†", locx,locy, F[locx].Sdag(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
Simp (size_t locx, size_t locy, double factor)
{
	return make_local(IMP, "Simp", locx,locy, B[locx].S(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
Simpdag (size_t locx, size_t locy, double factor)
{
	return make_local(IMP, "Simp†", locx,locy, B[locx].Sdag(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return make_corr (SUB, "n","n", locx1,locx2,locy1,locy2, F[locx1].n(locy1), F[locx2].n(locy2), Symmetry::qvacuum(), 1., true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
SsubSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return make_corr (SUB, "Ssub","Ssub", locx1,locx2,locy1,locy2, F[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
SimpSimp (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return make_corr (IMP, "Simp","Simp", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),B[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
SimpSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return make_corr (IMPSUB, "Simp","Ssub", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	// Not well implemented.. the same problems with sign as in the Hubbard model.
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
	auto cdag = OperatorType::outerprod(B[locx1].Id(), F[locx1].cdag(locy1),{2,+1});
	auto c    = OperatorType::outerprod(B[locx2].Id(), F[locx2].c(locy2),   {2,-1});
	auto sign = OperatorType::outerprod(B[locx2].Id(), F[locx2].sign(),     {1, 0});
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
	}
	else if(locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(cdag, sign, {2,+1}).plain<double>(), 
		                               pow(-1.,locx2-locx1+1) * c.plain<double>()}, 
		                               sign.plain<double>());
		// old:
		// Mout.setLocal({locx1, locx2}, {-sqrt(2.) * cdag.plain<double>(), 
		//                                OperatorType::prod(sign, c, {2,-1}).plain<double>()}, 
		//                                sign.plain<double>());
	}
	else if(locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.) * OperatorType::prod(c, sign, {2,-1}).plain<double>(), 
		                               pow(-1.,locx1-locx2+1) * cdag.plain<double>()}, 
		                               sign.plain<double>());
	// old:
		// Mout.setLocal({locx1, locx2}, {-sqrt(2.) * OperatorType::prod(sign, cdag, {2,+1}).plain<double>(), 
		//                                c.plain<double>()}, 
		//                                sign.plain<double>());
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
ccdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	//Not well implemented.. the same problems with sign as in the Hubbard model.
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c(" << locx1 << "," << locy1 << ")" << "c†(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }
	
	auto cdag = OperatorType::outerprod(B[locx1].Id(),F[locx1].cdag(locy1),{2,+1});
	auto c    = OperatorType::outerprod(B[locx2].Id(),F[locx2].c(locy2),{2,-1});
	auto sign = OperatorType::outerprod(B[locx2].Id(),F[locx2].sign(),{1,0});
	if(locx1 == locx2)
	{
		auto product = sqrt(2.)*OperatorType::prod(c,cdag,Symmetry::qvacuum());
		Mout.setLocal(locx1,product.plain<double>());
	}
	else if(locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(c, sign, {2,+1}).plain<double>(), 
		                               pow(-1.,locx2-locx1+1) * cdag.plain<double>()}, 
		                               sign.plain<double>());
	}
	else if(locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.)*OperatorType::prod(cdag, sign, {2,-1}).plain<double>(), 
		                               pow(-1.,locx1-locx2+1) * c.plain<double>()}, 
		                               sign.plain<double>());
	}
	return Mout;
}


// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cdagcdagcc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << "η†(" << locx1 << "," << locy1 << ")" << "η(" << locx2 << "," << locy2 << ")";

// 	Mpo<Symmetry> Mout(N_sites, N_legs);
// 	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Etadag = Operator::outerprod(Spins.Id(),F.Etadag(locy1),{1,2});
// 	auto Eta = Operator::outerprod(Spins.Id(),F.Eta(locy2),{1,-2});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(locx1 == locx2)
// 	{
// 		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
// 		Mout.setLocal(locx1,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({locx1, locx2}, {Etadag, Eta}, {{1,2},{1,-2}});
// 		return Mout;
// 	}
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsubSimpSimp (size_t locx1, size_t locx2, size_t loc3x, size_t loc4x,
//                   size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(Mpo<2>::qloc[locx2].size()/F.dim(), Mpo<2>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,locy2)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(S.Scomp(SOP4,loc4y),IdSub)});
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsubSimpSsub (size_t locx1, size_t locx2, size_t loc3x, size_t loc4x,
// 				  size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(Mpo<2>::qloc[locx2].size()/F.dim(), Mpo<2>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,locy2)),
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
