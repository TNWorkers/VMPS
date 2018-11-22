#ifndef STRAWBERRY_KONDOMODEL
#define STRAWBERRY_KONDOMODEL

#include "models/KondoObservables.h"
//include "bases/FermionBase.h"
//include "bases/SpinBase.h"
#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
//include "Mpo.h"
//include "ParamHandler.h" // from TOOLS
#include "ParamReturner.h"

namespace VMPS
{

/** \class KondoU1xU1
  * \ingroup Kondo
  *
  * \brief Kondo Model
  *
  * MPO representation of 
  * \f[
  * H = - \sum_{<ij>\sigma} \left(c^\dagger_{i\sigma}c_{j\sigma} +h.c.\right)
  * - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z
  * \f]
  *
  * where further parameters from HubbardU1xU1 and HeisenbergU1 are possible.
  * \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin of the impurity.
  *
  * \note Take use of the \f$S_z\f$ U(1) symmetry and the U(1) particle conservation symmetry.
  * \note The default variable settings can be seen in \p KondoU1xU1::defaults.
  * \note \f$J<0\f$ is antiferromagnetic
  * \note If nnn-hopping is positive, the GS-energy is lowered.
  * \note The multi-impurity model can be received, by setting D=1 (S=0) for all sites without an impurity.
  */
class KondoU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,double>,
                   public KondoObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > >,
                   public ParamReturner

{
public:
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(KondoU1xU1)
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	KondoU1xU1 () : Mpo(){};
	KondoU1xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{0,N};};
	static qarray<2> polaron (int L, int N=0) {assert(N%2==0); return qarray<2>{L,N};};
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local spin-operators are received
	 * \param F : Base class from which the local fermion-operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	 */
	template<typename Symmetry_> 
//	static HamiltonianTermsXd<Symmetry_> set_operators (const vector<SpinBase<Symmetry_> > &B, const vector<FermionBase<Symmetry_> > &F,
//	                                                    const ParamHandler &P, size_t loc=0);
	static void set_operators(const std::vector<SpinBase<Symmetry_> > &B, const vector<FermionBase<Symmetry_> > &F,
	                          const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

const map<string,any> KondoU1xU1::defaults =
{
	{"t",1.}, {"tPrime",0.}, {"tRung",0.},
	{"J",1.}, {"U",0.}, 
	{"V",0.}, {"Vrung",0.}, 
	{"mu",0.}, {"t0",0.},
	{"Bz",0.}, {"Bzsub",0.}, {"Kz",0.},
	{"Inext",0.}, {"Iprev",0.}, {"I3next",0.}, {"I3prev",0.}, {"I3loc",0.}, 
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, {"LyF",1ul}
};

const map<string,any> VMPS::KondoU1xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",10ul}, {"eps_svd",1e-7},
	{"Dincr_abs",4ul}, {"Dincr_per",2ul}, {"Dincr_rel",1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",6ul},
	{"Dinit",5ul}, {"Qinit",10ul}, {"Dlimit",100ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT",true}, {"CONVTEST",DMRG::CONVTEST::VAR_2SITE}
};

KondoU1xU1::
KondoU1xU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0,0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 KondoObservables(L,params,KondoU1xU1::defaults),
 ParamReturner(KondoU1xU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("LyF",l%Lcell);
		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
	}
	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Terms[l] = set_operators(B,F,P,l%Lcell);
//		
//		stringstream ss;
//		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell) << ",LyF=" << P.get<size_t>("LyF",l%Lcell);
//		Terms[l].info.push_back(ss.str());
//	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	set_operators(B,F,P,Terms);
//	cout << Terms.print_info() << endl;

	//Workaround for wrong qOp/auxBasis if we have a Kondo_anpacked case
	bool RESET_Q_OP=false;
	for (size_t l=0; l<N_sites; l++)
	{
		if (P.get<size_t>("LyF",l%Lcell) == 0ul)
		{
			RESET_Q_OP = true;
			break;
		}
	}

	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"), RESET_Q_OP);
	this->precalc_TwoSiteData();
}

bool KondoU1xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[1],2); //electrons have spin 1/2
	frac Smax = S_elec;
	//add local spins to Smax
	for (size_t l=0; l<N_sites; ++l)
	{
		Smax += static_cast<int>(B[l].orbitals()) * frac(B[l].get_D()-1,2);
	}
	
	frac S_tot(qnum[0],2);
	if (Smax.denominator() == S_tot.denominator() and S_tot<=Smax and qnum[1]<=2*static_cast<int>(this->N_phys) and qnum[1]>0) {return true;}
	else {return false;}
}

template<typename Symmetry_>
void KondoU1xU1::
set_operators (const vector<SpinBase<Symmetry_> > &B, const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	Terms.set_name("Kondo");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[(loc+1)%N_sites].orbitals();
		std::size_t Fnextn_orbitals = F[(loc+2)%N_sites].orbitals();
		
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[(loc+1)%N_sites].orbitals();
		std::size_t Bnextn_orbitals = B[(loc+2)%N_sites].orbitals();
		
		stringstream Slabel, LyLabel, LyFlabel;
		Slabel << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
		LyLabel << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		LyFlabel << "LyF=" << P.get<size_t>("LyF",loc%Lcell);
		Terms.save_label(loc, Slabel.str());
		Terms.save_label(loc, LyLabel.str());
		Terms.save_label(loc, LyFlabel.str());
		
		// local terms
		
		// Kondo-J
		param1d J = P.fill_array1d<double>("J", "Jorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, J.label);
		
		// Hubbard-U
		param1d U = P.fill_array1d<double>("U", "Uorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, U.label);
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung", "t", "tPerp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		Terms.save_label(loc, tPerp.label);
		
		// V⟂
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		Terms.save_label(loc, Vperp.label);
		
		// mu
		param1d mu = P.fill_array1d<double>("mu", "muorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, mu.label);
		
		// t0
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", Forbitals, loc%Lcell);
		Terms.save_label(loc, t0.label);
		
		// Kz anisotropy
		param1d Kz = P.fill_array1d<double>("Kz", "Kzorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, Kz.label);
		
		// Bz substrate
		param1d Bzsub = P.fill_array1d<double>("Bzsub", "Bzsuborb", Forbitals, loc%Lcell);
		Terms.save_label(loc, Bzsub.label);
		
		// Bz impurities
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, Bz.label);
		
		// I3loc
		param1d I3loc = P.fill_array1d<double>("I3loc", "I3locOrb", Forbitals, loc%Lcell);
		Terms.save_label(loc, I3loc.label);
		
		ArrayXXd Jxyperp  = B[loc].ZeroHopping();
		ArrayXXd Jzperp   = B[loc].ZeroHopping();
		ArrayXd  Bxorb    = B[loc].ZeroField();
		ArrayXd  Bxsuborb = F[loc].ZeroField();
		ArrayXd  Kxorb    = B[loc].ZeroField();
		ArrayXXd Dyperp   = B[loc].ZeroHopping();
		ArrayXXd Jperp    = F[loc].ZeroHopping();
		
		if (Borbitals > 0 and Forbitals > 0)
		{
			auto Himp = kroneckerProduct(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bz.a,Bxorb,Kz.a,Kxorb,Dyperp), F[loc].Id());
			auto Hsub = kroneckerProduct(B[loc].Id(), F[loc].template HubbardHamiltonian<double>(U.a,t0.a-mu.a,Bzsub.a,Bxsuborb,tPerp.a,Vperp.a,Jperp));
			auto Hloc = Himp + Hsub;
			
			for (int alfa=0; alfa<Forbitals; ++alfa)
			{
				if (J(alfa) != 0.)
				{
					assert(Forbitals == Borbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
					Hloc += 0.5*J(alfa) * kroneckerProduct(B[loc].Scomp(SP,alfa), F[loc].Sm(alfa));
					Hloc += 0.5*J(alfa) * kroneckerProduct(B[loc].Scomp(SM,alfa), F[loc].Sp(alfa));
					Hloc +=     J(alfa) * kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Sz(alfa));
				}
			}
			
			for (int alfa=0; alfa<Forbitals; ++alfa)
			{
				if (I3loc(alfa) != 0.)
				{
					assert(Forbitals == Borbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
					Hloc += I3loc(alfa) * kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Sz(alfa));
				}
			}
			
			Terms.push_local(loc, 1., Hloc);
		}
		
		// NN terms
		
		// t∥
		param2d tPara = P.fill_array2d<double>("t", "tPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, tPara.label);
		
		// V∥
		param2d Vpara = P.fill_array2d<double>("V", "Vpara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, Vpara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Forbitals;      ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				Terms.push_tight(loc, -tPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,beta))
				                );
				Terms.push_tight(loc, -tPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,beta))
				                );
				Terms.push_tight(loc, -tPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(UP,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].cdag(UP,beta))
				                );
				Terms.push_tight(loc, -tPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(DN,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].cdag(DN,beta))
				                );
				
				Terms.push_tight(loc, Vpara(alfa,beta), 
				                 kroneckerProduct(B[loc].Id(),F[loc].n(alfa)), 
				                 kroneckerProduct(B[lp1].Id(),F[lp1].n(beta))
				                );
			}
		}
		
		// NN spin exchange terms
		
		param2d InextPara = P.fill_array2d<double>("Inext", "InextPara", {Borbitals, Fnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, InextPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Borbitals;      ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				Terms.push_tight(loc, 0.5*InextPara(alfa,beta),
				                 kroneckerProduct(B[loc].Scomp(SP,alfa), F[loc].Id()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].Sm(beta))
				                );
				Terms.push_tight(loc, 0.5*InextPara(alfa,beta),
				                 kroneckerProduct(B[loc].Scomp(SM,alfa), F[loc].Id()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].Sp(beta))
				                );
				Terms.push_tight(loc, InextPara(alfa,beta),
				                 kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Id()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].Sz(beta))
				                );
			}
		}
		
		param2d IprevPara = P.fill_array2d<double>("Iprev", "IprevPara", {Forbitals, Bnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, IprevPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Borbitals;  ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				Terms.push_tight(loc, 0.5*IprevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].Sm(alfa)),
				                 kroneckerProduct(B[lp1].Scomp(SP,beta), F[lp1].Id())
				                );
				Terms.push_tight(loc, 0.5*IprevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].Sp(alfa)),
				                 kroneckerProduct(B[lp1].Scomp(SM,beta), F[lp1].Id())
				                );
				Terms.push_tight(loc, IprevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].Sz(alfa)),
				                 kroneckerProduct(B[lp1].Scomp(SZ,beta), F[lp1].Id())
				                );
			}
		}
		
		// NN 3-orbital spin exchange terms
		
		param2d I3nextPara = P.fill_array2d<double>("I3next", "I3nextPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, I3nextPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Fnext_orbitals;  ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				assert(Borbitals == 1);
				
				Terms.push_tight(loc, 0.5*I3nextPara(alfa,beta),
					                  kroneckerProduct(B[loc].Scomp(SM,0), F[loc].cdag(UP,alfa) * F[loc].sign()),
					                  kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,beta))
					                 );
				Terms.push_tight(loc, 0.5*I3nextPara(alfa,beta),
						                         kroneckerProduct(B[loc].Scomp(SP,0), F[loc].cdag(DN,alfa) * F[loc].sign()),
						                         kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,beta))
						                         );
				Terms.push_tight(loc, 0.5*I3nextPara(alfa,beta),
						                         kroneckerProduct(B[loc].Scomp(SM,0), -1.*F[loc].c(DN,alfa) * F[loc].sign()),
						                         kroneckerProduct(B[lp1].Id(), F[lp1].cdag(UP,beta))
						             );
				Terms.push_tight(loc, 0.5*I3nextPara(alfa,beta),
						                         kroneckerProduct(B[loc].Scomp(SP,0), -1.*F[loc].c(UP,alfa) * F[loc].sign()),
						                         kroneckerProduct(B[lp1].Id(), F[lp1].cdag(DN,beta))
						             );
				Terms.push_tight(loc, I3nextPara(alfa,beta),
						                         kroneckerProduct(B[loc].Scomp(SZ,0), F[loc].Id()),
						                         kroneckerProduct(B[lp1].Id(), F[lp1].Sz(beta))
						             );
			}
		}
		
		param2d I3prevPara = P.fill_array2d<double>("I3prev", "I3prevPara", {Forbitals, Bnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, I3prevPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Borbitals;  ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				assert(Bnext_orbitals == 1);
				
				Terms.push_tight(loc, 0.5*I3prevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Scomp(SM,0), F[lp1].c(DN,beta))
				                );
				Terms.push_tight(loc, 0.5*I3prevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Scomp(SP,0), F[lp1].c(UP,beta))
				                );
				Terms.push_tight(loc, 0.5*I3prevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(DN,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Scomp(SM,0), F[lp1].cdag(UP,beta))
				                );
				Terms.push_tight(loc, 0.5*I3prevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(UP,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Scomp(SP,0), F[lp1].cdag(DN,beta))
				                );
				Terms.push_tight(loc, I3prevPara(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].Sz(alfa)),
				                 kroneckerProduct(B[lp1].Scomp(SZ,0), F[lp1].Id())
				                );
			}
		}
		
		// NNN terms
		
		param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {Forbitals, Fnextn_orbitals}, loc%Lcell);
		Terms.save_label(loc, tPrime.label);
		
		if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Forbitals;       ++alfa)
			for (std::size_t beta=0; beta<Fnextn_orbitals; ++beta)
			{
				Terms.push_nextn(loc, -tPrime(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].sign()),
				                 kroneckerProduct(B[lp2].Id(), F[lp2].c(UP,beta))
				                );
				Terms.push_nextn(loc, -tPrime(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].sign()),
				                 kroneckerProduct(B[lp2].Id(), F[lp2].c(DN,beta))
				                );
				Terms.push_nextn(loc, -tPrime(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(UP,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].sign()),
				                 kroneckerProduct(B[lp2].Id(), F[lp2].cdag(UP,beta))
				                );
				Terms.push_nextn(loc, -tPrime(alfa,beta),
				                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(DN,alfa) * F[loc].sign()),
				                 kroneckerProduct(B[lp1].Id(), F[lp1].sign()),
				                 kroneckerProduct(B[lp2].Id(), F[lp2].cdag(DN,beta))
				                );
			}
		}
	}
}

//template<typename Symmetry_>
//HamiltonianTermsXd<Symmetry_> KondoU1xU1::
//set_operators (const vector<SpinBase<Symmetry_> > &B, const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc)
//{
//	HamiltonianTermsXd<Symmetry_> Terms;
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
//			Terms.tight.push_back(make_tuple(-tPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].c(UP,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(-tPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].c(DN,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(-tPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(UP,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(-tPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(DN,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,i))
//			                                 ));
//		}
//		
//		if (Vpara(i,j) != 0.)
//		{
//			if (Vpara(i,j) != 0.)
//			{
//				Terms.tight.push_back(make_tuple(Vpara(i,j), 
//				                                 kroneckerProduct(B[loc].Id(),F[loc].n(i)), 
//				                                 kroneckerProduct(B[loc].Id(),F[loc].n(i))
//				                                 ));
//			}
//		}
//	}
//	
//	// NN spin exchange terms
//	
//	auto [Inext,InextPara,InextLabel] = P.fill_array2d<double>("Inext","InextPara",{{B[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(InextLabel);
//	
//	auto [I3next,I3nextPara,I3nextLabel] = P.fill_array2d<double>("I3next","I3nextPara",{{B[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(I3nextLabel);
//	
//	for (int i=0; i<B[loc].orbitals(); ++i)
//	for (int j=0; j<F[lp1].orbitals(); ++j)
//	{
//		if (InextPara(i,j) != 0.)
//		{
//			Terms.tight.push_back(make_tuple(0.5*InextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SP,i), F[loc].Id()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sm(i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*InextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SM,i), F[loc].Id()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sp(i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(InextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SZ,i), F[loc].Id()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sz(i))
//			                                 ));
//		}
//		
//		if (I3nextPara(i,j) != 0.)
//		{
//			Terms.tight.push_back(make_tuple(0.5*I3nextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SM,i), F[loc].cdag(UP,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].c(DN,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*I3nextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SP,i), F[loc].cdag(DN,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].c(UP,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*I3nextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SM,i), -1.*F[loc].c(DN,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*I3nextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SP,i), -1.*F[loc].c(UP,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(I3nextPara(i,j),
//			                                 kroneckerProduct(B[loc].Scomp(SZ,i), F[loc].Id()),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sz(i))
//			                                 ));
//		}
//	}
//	
//	auto [Iprev,IprevPara,IprevLabel] = P.fill_array2d<double>("Iprev","IprevPara",{{F[loc].orbitals(),B[lp1].orbitals()}},loc);
//	save_label(IprevLabel);
//	
//	auto [I3prev,I3prevPara,I3prevLabel] = P.fill_array2d<double>("I3prev","I3prevPara",{{B[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(I3prevLabel);
//	
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	for (int j=0; j<B[lp1].orbitals(); ++j)
//	{
//		if (IprevPara(i,j) != 0.)
//		{
//			Terms.tight.push_back(make_tuple(0.5*IprevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sm(i)),
//			                                 kroneckerProduct(B[loc].Scomp(SP,i), F[loc].Id())
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*IprevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sp(i)),
//			                                 kroneckerProduct(B[loc].Scomp(SM,i), F[loc].Id())
//			                                 ));
//			Terms.tight.push_back(make_tuple(IprevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sz(i)),
//			                                 kroneckerProduct(B[loc].Scomp(SZ,i), F[loc].Id())
//			                                 ));
//		}
//		
//		if (I3prevPara(i,j) != 0.)
//		{
//			Terms.tight.push_back(make_tuple(0.5*I3prevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Scomp(SM,i), F[loc].c(DN,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*I3prevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Scomp(SP,i), F[loc].c(UP,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*I3prevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(DN,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Scomp(SM,i), F[loc].cdag(UP,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(0.5*I3prevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), -1.*F[loc].c(UP,i) * F[loc].sign()),
//			                                 kroneckerProduct(B[loc].Scomp(SP,i), F[loc].cdag(DN,i))
//			                                 ));
//			Terms.tight.push_back(make_tuple(I3prevPara(i,j),
//			                                 kroneckerProduct(B[loc].Id(), F[loc].Sz(i)),
//			                                 kroneckerProduct(B[loc].Scomp(SZ,i), F[loc].Id())
//			                                 ));
//		}
//	}
//	
//	// NNN terms
//	
//	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
//	save_label(tPrime.label);
//	
//	if (tPrime.x!=0)
//	{
//		assert(F[loc].orbitals() <= 1 and "Cannot do a ladder with t' terms!");
//		
//		Terms.nextn.push_back(make_tuple(-tPrime.x,
//		                                 kroneckerProduct(B[loc].Id(),F[loc].cdag(UP,0) * F[loc].sign()),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].c(UP,0)),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].sign())
//		                                 ));
//		Terms.nextn.push_back(make_tuple(-tPrime.x,
//		                                 kroneckerProduct(B[loc].Id(),F[loc].cdag(DN,0) * F[loc].sign()),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].c(DN,0)),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].sign())
//		                                 ));
//		Terms.nextn.push_back(make_tuple(-tPrime.x,
//		                                 kroneckerProduct(B[loc].Id(),-1.*F[loc].c(UP,0) * F[loc].sign()),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].cdag(UP,0)),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].sign())
//		                                 ));
//		Terms.nextn.push_back(make_tuple(-tPrime.x,
//		                                 kroneckerProduct(B[loc].Id(),-1.*F[loc].c(DN,0) * F[loc].sign()),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].cdag(DN,0)),
//		                                 kroneckerProduct(B[loc].Id(),F[loc].sign())
//		                                 ));
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
//	// Hubbard U
//	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
//	save_label(Ulabel);
//	
//	// mu
//	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F[loc].orbitals(),loc);
//	save_label(mulabel);
//	
//	// t0
//	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F[loc].orbitals(),loc);
//	save_label(t0label);
//	
//	// Kz anisotropy
//	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",F[loc].orbitals(),loc);
//	save_label(Kzlabel);
//	
//	// Bz substrate
//	auto [Bzsub,Bzsuborb,Bzsublabel] = P.fill_array1d<double>("Bzsub","Bzsuborb",F[loc].orbitals(),loc);
//	save_label(Bzsublabel);
//	
//	// Bz impurities
//	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F[loc].orbitals(),loc);
//	save_label(Bzlabel);
//	
//	ArrayXXd Jxyperp  = B[loc].ZeroHopping();
//	ArrayXXd Jzperp   = B[loc].ZeroHopping();
//	ArrayXd  Bxorb    = B[loc].ZeroField();
//	ArrayXd  Bxsuborb = F[loc].ZeroField();
//	ArrayXd  Kxorb    = B[loc].ZeroField();
//	ArrayXXd Dyperp   = B[loc].ZeroHopping();
//	ArrayXXd Jperp    = F[loc].ZeroHopping();
//	
//	if (B[loc].orbitals() > 0 and F[loc].orbitals() > 0)
//	{
//		auto Himp = kroneckerProduct(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp), F[loc].Id());
//		auto Hsub = kroneckerProduct(B[loc].Id(), F[loc].template HubbardHamiltonian<double>(Uorb,t0orb-muorb,Bzsuborb,Bxsuborb,tPerp,Vperp,Jperp));
//		auto Hloc = Himp + Hsub;
//		
//		// Kondo-J
//		auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
//		save_label(Jlabel);
//		
//		for (int i=0; i<F[loc].orbitals(); ++i)
//		{
//			if (Jorb(i) != 0.)
//			{
//				Hloc += 0.5*Jorb(i) * kroneckerProduct(B[loc].Scomp(SP,i), F[loc].Sm(i));
//				Hloc += 0.5*Jorb(i) * kroneckerProduct(B[loc].Scomp(SM,i), F[loc].Sp(i));
//				Hloc +=     Jorb(i) * kroneckerProduct(B[loc].Scomp(SZ,i), F[loc].Sz(i));
//			}
//		}
//		
//		auto [I3loc,I3locOrb,I3locLabel] = P.fill_array1d<double>("I3loc","I3locOrb",F[loc].orbitals(),loc);
//		save_label(I3locLabel);
//		
//		for (int i=0; i<F[loc].orbitals(); ++i)
//		{
//			if (I3locOrb(i) != 0.)
//			{
//				Hloc += I3locOrb(i) * kroneckerProduct(B[loc].Scomp(SZ,i), F[loc].Sz(i));
//			}
//		}
//		
//		Terms.local.push_back(make_tuple(1.,Hloc));
//	}
//	
//	Terms.name = "Kondo";
//	
//	return Terms;
//}

} //end namespace VMPS

#endif
