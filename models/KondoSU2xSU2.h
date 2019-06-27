#ifndef KONDOMODEL_SU2XSU2_H_
#define KONDOMODEL_SU2XSU2_H_

#include "symmetry/SU2.h"
#include "bases/SpinBaseSU2xSU2.h"
#include "bases/FermionBaseSU2xSU2.h"
#include "Mpo.h"
#include "ParamReturner.h"

namespace VMPS
{

class KondoSU2xSU2 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,double>, public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	MAKE_TYPEDEFS(KondoSU2xSU2)
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	KondoSU2xSU2 (): Mpo(), ParamReturner(KondoSU2xSU2::sweep_defaults) {};
	KondoSU2xSU2 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{1,1};};
	
//	static qarray<1> singlet (int N) {return qarray<1>{1};};
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local spin-operators are received
	 * \param F : Base class from which the local fermion-operators are received
	 * \param P : The parameters
	 * \param Terms : \p HamiltonianTerms instance
	 */
	static void set_operators (const vector<SpinBase<Symmetry> > &B, 
	                           const vector<FermionBase<Symmetry> > &F, 
	                           const ParamHandler &P, 
	                           HamiltonianTermsXd<Symmetry> &Terms);
	
//	Mpo<Symmetry> Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
//	Mpo<Symmetry> Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
//	
	Mpo<Symmetry> SimpSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SsubSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SimpSimp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	Mpo<Symmetry> ns (size_t locx, size_t locy=0);
	Mpo<Symmetry> nh (size_t locx, size_t locy=0);
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	static const std::map<string,std::any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	Mpo<Symmetry> make_corr (KONDO_SUBSYSTEM SUBSYS,
	                         string name1, string name2, 
	                         size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
	                         const OperatorType &Op1, const OperatorType &Op2, 
	                         qarray<Symmetry::Nq> Qtot, double factor,
	                         bool BOTH_HERMITIAN=false) const;
	
	Mpo<Symmetry>  make_local (KONDO_SUBSYSTEM SUBSYS, 
	                           string name, 
	                           size_t locx, size_t locy, 
	                           const OperatorType &Op, 
	                           double factor, bool FERMIONIC, bool HERMITIAN) const;
	
	vector<FermionBase<Symmetry> > F;
	vector<SpinBase   <Symmetry> > B;
};

const std::map<string,std::any> KondoSU2xSU2::defaults =
{
	{"t",1.}, {"tRung",0.},
	{"J",1.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"D",2ul}, {"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const map<string,any> KondoSU2xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",16ul}, {"eps_svd",1e-7},
	{"Dincr_abs",5ul}, {"Dincr_per",2ul}, {"Dincr_rel",1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",10ul},
	{"Dinit",5ul}, {"Qinit",6ul}, {"Dlimit",200ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KondoSU2xSU2::
KondoSU2xSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 ParamReturner(KondoSU2xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	B.resize(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = (l%2 == 0) ? FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), SUB_LATTICE::A) 
		                  : FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), SUB_LATTICE::B);
		B[l] = (l%2 == 0) ? SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell), SUB_LATTICE::B)
		                  : SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell), SUB_LATTICE::A);
		
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	set_operators(B,F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

void KondoSU2xSU2::
set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	Terms.set_name("Kondo");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[lp1].orbitals();
		std::size_t Fnextn_orbitals = F[lp2].orbitals();
		
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[lp1].orbitals();
		std::size_t Bnextn_orbitals = B[lp2].orbitals();
		
		stringstream Slabel;
		Slabel << "S=" << print_frac_nice(frac(B[loc].get_D()-1,2));
		Terms.save_label(loc, Slabel.str());
		
		// local terms
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung","t","tPerp",Forbitals,loc%Lcell,P.get<bool>("CYLINDER"));
		Terms.save_label(loc, tPerp.label);
		
		// V⟂
		param2d Vperp = P.fill_array2d<double>("Vrung","V","Vperp",Forbitals,loc%Lcell,P.get<bool>("CYLINDER"));
		Terms.save_label(loc, Vperp.label);
		
		// Hubbard U
		param1d U = P.fill_array1d<double>("U","Uorb",Forbitals,loc%Lcell);
		Terms.save_label(loc, U.label);
		
		if (F[loc].dim() > 1)
		{
			OperatorType KondoHamiltonian({1,1}, B[loc].get_basis().combine(F[loc].get_basis()));
			
			ArrayXXd Jperp    = B[loc].ZeroHopping();
			ArrayXXd Jperpsub = F[loc].ZeroHopping();
			
			//set Hubbard part of Kondo Hamiltonian
			KondoHamiltonian = OperatorType::outerprod(B[loc].Id(), F[loc].HubbardHamiltonian(U.a,tPerp.a,Vperp.a,Jperpsub), {1,1});
			
			//set Heisenberg part of Hamiltonian
		//	KondoHamiltonian += OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jperp), F[loc].Id(), {1,1});
			
			// Kondo-J
			param1d J = P.fill_array1d<double>("J","Jorb",Forbitals,loc%Lcell);
			Terms.save_label(loc, J.label);
			
			//set interaction part of Hamiltonian.
			for (int alfa=0; alfa<Forbitals; ++alfa)
			{
				assert(Borbitals == Forbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
				KondoHamiltonian += J(alfa) * sqrt(3.) * OperatorType::outerprod(B[loc].Sdag(alfa), F[loc].S(alfa), {1,1});
			}
			
			Terms.push_local(loc, 1., KondoHamiltonian.plain<double>());
		}
		
		// NN terms
		
		auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{Forbitals,F[lp1].orbitals()}},loc%Lcell);
		Terms.save_label(loc, tlabel);
		
		auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{Forbitals,F[lp1].orbitals()}},loc%Lcell);
		Terms.save_label(loc, Vlabel);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (int alfa=0; alfa<Forbitals;      ++alfa)
			for (int beta=0; beta<Fnext_orbitals; ++beta)
			{
				auto cdag_sign_loc = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].cdag(alfa), {2,2}),
				                                        OperatorType::outerprod(B[loc].Id(), F[loc].sign(),     {1,1}),
				                                        {2,2}).plain<double>();
				Terms.push_tight(loc, -tPara(alfa,beta) * sqrt(2.) * sqrt(2.),
				                      cdag_sign_loc, OperatorType::outerprod(B[lp1].Id(), F[lp1].c(beta), {2,2}).plain<double>());
				
				auto Tdag_loc = OperatorType::outerprod(B[loc].Id(), F[loc].Tdag(alfa), {1,3}).plain<double>();
				auto T_lp1    = OperatorType::outerprod(B[lp1].Id(), F[lp1].T   (beta), {1,3}).plain<double>();
				
				Terms.push_tight(loc, Vpara(alfa,beta) * std::sqrt(3.), Tdag_loc, T_lp1);
			}
		}
	}
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
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
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
	OperatorType OpExt;
	vector<SiteOperator<Symmetry,MatrixType::Scalar> > SignExt(locx);
	
	if (SUBSYS == SUB)
	{
		OpExt   = OperatorType::outerprod(B[locx].Id(), Op, Op.Q());
		for (size_t l=0; l<locx; ++l)
		{
			SignExt[l] = OperatorType::outerprod(B[l].Id(), F[l].sign(), Symmetry::qvacuum()).plain<double>();
		}
	}
	else if (SUBSYS == IMP)
	{
		assert(!FERMIONIC and "Impurity cannot be fermionic!");
		OpExt = OperatorType::outerprod(Op, F[locx].Id(), Op.Q());
	}
	
	Mout.set_locality(locx);
	Mout.set_localOperator(OpExt.plain<double>());
	
	(FERMIONIC)? Mout.setLocal(locx, (factor * OpExt).plain<double>(), SignExt)
	           : Mout.setLocal(locx, (factor * OpExt).plain<double>());
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
nh (size_t locx, size_t locy)
{
	return make_local(SUB, "nh", locx,locy, F[locx].nh(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
ns (size_t locx, size_t locy)
{
	return make_local(SUB, "ns", locx,locy, F[locx].ns(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
make_corr (KONDO_SUBSYSTEM SUBSYS,
           string name1, string name2, 
           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
           const OperatorType &Op1, const OperatorType &Op2, 
           qarray<Symmetry::Nq> Qtot, 
           double factor,
           bool BOTH_HERMITIAN) const
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

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
SsubSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (SUB, "Ssub","Ssub", locx1,locx2,locy1,locy2, F[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
SimpSimp (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (IMP, "Simp","Simp", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),B[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
SimpSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (IMPSUB, "Simp","Ssub", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
	auto cdag  = OperatorType::outerprod(B[locx1].Id(), F[locx1].cdag(locy1),{2,2});
	auto c     = OperatorType::outerprod(B[locx2].Id(), F[locx2].c(locy2),   {2,2});
	auto sign1 = OperatorType::outerprod(B[locx2].Id(), F[locx1].sign(),     {1,1});
	auto sign2 = OperatorType::outerprod(B[locx2].Id(), F[locx2].sign(),     {1,1});
	
	vector<SiteOperator<Symmetry,MatrixType::Scalar> > signs;
	for (size_t l=min(locx1,locx2)+1; l<max(locx1,locx2); l++)
	{
		signs.push_back(OperatorType::outerprod(B[l].Id(), F[l].sign(), {1,1}).plain<double>());
	}
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
	}
	else if(locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(cdag, sign1, {2,2}).plain<double>(), c.plain<double>()}, signs);
	}
	else if(locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.) * OperatorType::prod(c, sign2, {2,2}).plain<double>(), cdag.plain<double>()}, signs);
	}
	return Mout;
}

//HamiltonianTermsXd<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
//set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
//{
//	HamiltonianTermsXd<Symmetry> Terms;
//	
//	frac S = frac(B[loc].get_D()-1,2);
//	stringstream Slabel;
//	Slabel << "S=" << print_frac_nice(S);
//	Terms.info.push_back(Slabel.str());
//	
//	auto save_label = [&Terms] (string label)
//	{
//		if (label!="") {Terms.info.push_back(label);}
//	};
//	
//	size_t lp1 = (loc+1)%F.size();
//	
//	// NN terms
//	
//	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(tlabel);
//	
//	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(Vlabel);
//	
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	for (int j=0; j<F[lp1].orbitals(); ++j)
//	{
//		if (tPara(i,j) != 0.)
//		{
////			auto Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].cdag(i), {2,2}),
////			                               OperatorType::outerprod(B[loc].Id(), F[loc].sign() , {1,1}),
////			                               {2,2});
////			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.)*sqrt(2.),
////			                                 Otmp.plain<double>(),
////			                                 OperatorType::outerprod(B[loc].Id(), F[loc].c(i), {2,2}).plain<double>()));
//		}
//	}
//	
//	// local terms
//	
//	// t⟂
//	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(tPerplabel);
//	
//	// V⟂
//	auto [Vrung,Vperp,Vperplabel] = P.fill_array2d<double>("Vrung","V","Vperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(Vperplabel);
//	
//	// J⟂
//	auto [Jrung,Jperp,Jperplabel] = P.fill_array2d<double>("Jrung","J","Jperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(Jperplabel);
//	
//	// Hubbard U
//	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
//	save_label(Ulabel);
//	
//	OperatorType KondoHamiltonian({1,0}, B[loc].get_structured_basis().combine(F[loc].get_basis()));
//	
//	//set Hubbard part of Kondo Hamiltonian
//	KondoHamiltonian = OperatorType::outerprod(B[loc].Id(), F[loc].HubbardHamiltonian(Uorb,tPerp,Vperp,Jperp), {1,1});
//	
//	//set Heisenberg part of Hamiltonian
////	KondoHamiltonian += OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jperp), F[loc].Id(), {1,1});
//	
//	// Kondo-J
//	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
//	save_label(Jlabel);
//	
//	//set interaction part of Hamiltonian.
//	
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	{
//		KondoHamiltonian += Jorb(i)*sqrt(3.) * OperatorType::outerprod(B[loc].Sdag(i), F[loc].S(i), {1,1});
//	}
//	
//	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
//	
//	Terms.name = "Kondo SU(2)⊗SU(2)";
//	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
//	
//	return Terms;
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//Simp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
//{
//	assert(locx < this->N_sites);
//	std::stringstream ss;
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
//	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
//	
//	auto Sop = OperatorType::outerprod(B[locx].Scomp(Sa,locy).structured(), F[locx].Id(), {1});
//	
//	Mout.setLocal(locx, Sop.plain<double>());
//	return Mout;
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy) const
//{
//	assert(locx < this->N_sites);
//	std::stringstream ss;
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
//	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
//	
//	auto Sop = OperatorType::outerprod(B[locx].Id().structured(), F[locx].Scomp(Sa,locy), {1});
//	
//	Mout.setLocal(locx, Sop.plain<double>());
//	return Mout;
//}

///*Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::*/
///*SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const*/
///*{*/
///*	assert(locx1 < this->N_sites and locx2 < this->N_sites);*/
///*	std::stringstream ss;*/
///*	*/
///*	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());*/
///*	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }*/
///*	*/
///*	auto Sop1 = OperatorType::outerprod(B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Id(), {1});*/
///*	auto Sop2 = OperatorType::outerprod(B[locx1].Id().structured(), F[locx2].Scomp(SOP2,locy2), {1});*/
///*	*/
///*	Mout.setLocal({locx1,locx2}, {Sop1.plain<double>(),Sop2.plain<double>()});*/
///*	return Mout;*/
///*}*/

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, 
//           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//           const OperatorType &Op1, const OperatorType &Op2, 
//           bool BOTH_HERMITIAN) const
//{
//	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << name1 << "(" << locx1 << "," << locy1 << ")"
//	   << name2 << "(" << locx2 << "," << locy2 << ")";
//	
//	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
//	
//	OperatorType Op1Ext;
//	OperatorType Op2Ext;
//	
//	Mpo<Symmetry> Mout(F.size(), Symmetry::qvacuum(), ss.str(), HERMITIAN);
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l);}
//	
//	if (SUBSYS == SUB)
//	{
//		Op1Ext = OperatorType::outerprod(B[locx1].Id().structured(), Op1, {1});
//		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
//	}
//	else if (SUBSYS == IMP)
//	{
//		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
//		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), {1});
//	}
//	else if (SUBSYS == IMPSUB and locx1 != locx2)
//	{
//		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
//		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
//	}
//	else if (SUBSYS == IMPSUB and locx1 == locx2)
//	{
//		OperatorType OpExt = OperatorType::outerprod(Op1, Op2, {1});
//		
//		Mout.setLocal(locx1, OpExt.plain<double>());
//		return Mout;
//	}
//	
//	Mout.setLocal({locx1,locx2}, {Op1Ext.plain<double>(),Op2Ext.plain<double>()});
//	return Mout;
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	stringstream ss1; ss1 << SOP1 << "imp";
//	stringstream ss2; ss2 << SOP2 << "sub";
//	
//	return make_corr(IMPSUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Scomp(SOP2,locy2));
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	stringstream ss1; ss1 << SOP1 << "imp";
//	stringstream ss2; ss2 << SOP2 << "imp";
//	
//	return make_corr(IMP, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), B[locx2].Scomp(SOP2,locy2).structured());
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	stringstream ss1; ss1 << SOP1 << "sub";
//	stringstream ss2; ss2 << SOP2 << "sub";
//	
//	return make_corr(SUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
//}

} //end namespace VMPS

#endif
