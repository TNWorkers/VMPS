#ifndef KONDOMODEL_U0XU1_H_
#define KONDOMODEL_U0XU1_H_

#include "symmetry/SU2.h"
#include "bases/SpinBase.h"
#include "bases/FermionBaseU0xSU2.h"
#include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
#include "ParamReturner.h"

namespace VMPS
{
/** \class KondoU0xSU2
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
  * \note The default variable settings can be seen in \p KondoU0xSU2::defaults.
  * \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
  * \note If the nnn-hopping is positive, the ground state energy is lowered.
  * \warning \f$J<0\f$ is antiferromagnetic
  */
class KondoU0xSU2 : public Mpo<Sym::SU2<Sym::ChargeSU2>,double>, public ParamReturner
{
public:
	typedef Sym::SU2<Sym::ChargeSU2> Symmetry;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	MAKE_TYPEDEFS(KondoU0xSU2)
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	KondoU0xSU2 (): Mpo(), ParamReturner(KondoU0xSU2::sweep_defaults) {};
	KondoU0xSU2 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<1> singlet (int N) {return qarray<1>{1};};
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local spin-operators are received
	 * \param F : Base class from which the local fermion-operators are received
	 * \param P : The parameters
	 * \param Terms : \p HamiltonianTerms instance
	 */
//	static HamiltonianTermsXd<Symmetry> set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F,
//	                                                    const ParamHandler &P, size_t loc=0);
	static void set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F,
	                           const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms);
	
	Mpo<Symmetry> Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	Mpo<Symmetry> Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	
	Mpo<Symmetry> SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	static const std::map<string,std::any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	Mpo<Symmetry> make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
	                         const OperatorType &Op1, const OperatorType &Op2, bool BOTH_HERMITIAN=false) const;
	
	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> KondoU0xSU2::defaults =
{
	{"t",1.}, {"tRung",0.}, {"tPrime",0.}, {"tPrimePrime",0.},
	{"J",1.}, {"U",0.}, 
	{"V",0.}, {"Vrung",0.},
	{"Bz",0.}, {"Bzsub",0.}, {"Kz",0.}, {"Bx",0.}, {"Bxsub",0.}, {"Kx",0.},
	{"Inext",0.}, {"Iprev",0.}, {"I3next",0.}, {"I3prev",0.}, {"I3loc",0.}, 
	{"D",2ul}, {"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul},
	{"subL",SUB_LATTICE::A}
};

const map<string,any> KondoU0xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",15ul}, {"eps_svd",1e-7},
	{"Dincr_abs",5ul}, {"Dincr_per",2ul}, {"Dincr_rel",1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",10ul},
	{"Dinit",5ul}, {"Qinit",6ul}, {"Dlimit",200ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KondoU0xSU2::
KondoU0xSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 ParamReturner(KondoU0xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	B.resize(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		
		if (P.HAS("subL",l%Lcell))
		{
//			cout << "l=" << l << ", " << P.get<SUB_LATTICE>("subL",l%Lcell) << endl;
			F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<SUB_LATTICE>("subL",l%Lcell));
		}
		else
		{
//			cout << "l=" << l << ", " << "make default A/B" << endl;
			F[l] = (l%2 == 0) ? FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), SUB_LATTICE::A) 
			                  : FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), SUB_LATTICE::B);
		}
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		
		setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	set_operators(B,F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

void KondoU0xSU2::
set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	Terms.set_name("Kondo U(0)⊗SU(2)");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lm1 = (loc==0)? N_sites-1 : loc-1;
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		
		std::size_t Fprev_orbitals  = F[lm1].orbitals();
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[lp1].orbitals();
		std::size_t Fnextn_orbitals = F[lp2].orbitals();
		std::size_t F3next_orbitals = F[lp3].orbitals();
		
		std::size_t Bprev_orbitals  = B[lm1].orbitals();
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[lp1].orbitals();
		std::size_t Bnextn_orbitals = B[lp2].orbitals();
		std::size_t B3next_orbitals = B[lp3].orbitals();
		
		frac S = frac(B[loc].get_D()-1,2);
		stringstream Slabel;
		Slabel << "S=" << print_frac_nice(S);
		Terms.save_label(loc, Slabel.str());
		
		stringstream sublabel;
		sublabel << "lat=" << F[loc].sublattice();
		Terms.save_label(loc, sublabel.str());
		
		// local terms
		
		// Kondo-J
		param1d J = P.fill_array1d<double>("J", "Jorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, J.label);
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung", "t", "tPerp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		Terms.save_label(loc, tPerp.label);
		
		// Hubbard-U
		param1d U = P.fill_array1d<double>("U", "Uorb", Forbitals, loc%Lcell);
		Terms.save_label(loc, U.label);
		
		// Bx substrate
		param1d Bxsub = P.fill_array1d<double>("Bxsub", "Bxsuborb", Forbitals, loc%Lcell);
		Terms.save_label(loc, Bxsub.label);
		
		// Bx impurities
		param1d Bx = P.fill_array1d<double>("Bx","Bxorb", Borbitals, loc%Lcell);
		Terms.save_label(loc, Bx.label);
		
		// Kx anisotropy
		param1d Kx = P.fill_array1d<double>("Kx","Kxorb", Borbitals, loc%Lcell);
		Terms.save_label(loc, Kx.label);
		
		// Bz substrate
		param1d Bzsub = P.fill_array1d<double>("Bzsub", "Bzsuborb", Forbitals, loc%Lcell);
		Terms.save_label(loc, Bzsub.label);
		
		// Bz impurities
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", Borbitals, loc%Lcell);
		Terms.save_label(loc, Bz.label);
		
		// Kz anisotropy
		param1d Kz = P.fill_array1d<double>("Kz","Kzorb", Borbitals, loc%Lcell);
		Terms.save_label(loc, Kz.label);
		
		ArrayXXd muPerp  = B[loc].ZeroHopping();
		ArrayXXd Jxyperp = B[loc].ZeroHopping();
		ArrayXXd Jzperp  = B[loc].ZeroHopping();
		ArrayXXd DyPerp  = B[loc].ZeroHopping();
		
		//set Heisenberg part of Kondo Hamiltonian
		auto KondoHamiltonian = OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bz.a,Bx.a,muPerp,Kz.a,Kx.a,DyPerp).structured(),
		                                                F[loc].Id(),
		                                                {1});
		
		ArrayXXd Vperp      = F[loc].ZeroHopping();
		ArrayXXd Jxysubperp = F[loc].ZeroHopping();
		ArrayXXd Jzsubperp  = F[loc].ZeroHopping();
		
		//set Hubbard part of Kondo Hamiltonian
		KondoHamiltonian += OperatorType::outerprod(B[loc].Id().structured(),
		                                            F[loc].HubbardHamiltonian(U.a,tPerp.a,Vperp,Jxysubperp,Jzsubperp,Bzsub.a,Bxsub.a),
		                                            {1});
		
		//set Kondo part of Hamiltonian
		for (int alfa=0; alfa<Forbitals; ++alfa)
		{
			if (J(alfa) != 0.)
			{
				assert(Borbitals == Forbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
				KondoHamiltonian += 0.5*J(alfa) * OperatorType::outerprod(B[loc].Scomp(SP,alfa).structured(), F[loc].Sm(alfa), {1});
				KondoHamiltonian += 0.5*J(alfa) * OperatorType::outerprod(B[loc].Scomp(SM,alfa).structured(), F[loc].Sp(alfa), {1});
				KondoHamiltonian +=     J(alfa) * OperatorType::outerprod(B[loc].Scomp(SZ,alfa).structured(), F[loc].Sz(alfa), {1});
			}
		}
		
		Terms.push_local(loc, 1., KondoHamiltonian.plain<double>());
		
		// NN terms
		
		if (P.HAS("tFull"))
		{
			for (size_t hop=loc; hop<N_sites; ++hop)
			{
				size_t N_TransOps;
				if (hop == loc) {N_TransOps=0;} else {N_TransOps=hop-loc-1;}
				vector<SiteOperator<Symmetry,double> > TransOps(N_TransOps);
				for (size_t i=0; i<N_TransOps; i++)
				{
					TransOps[i] = OperatorType::outerprod(B[loc+i+1].Id().structured(), F[loc+i+1].sign(), {1}).plain<double>();
				}
				
				if (hop == loc)
				{
//					SiteOperator<Symmetry,double> Ssqrt = SiteOperatorQ<Symmetry,MatrixXd>::prod(B[loc].Sdag(0),B[loc].S(0),Symmetry::qvacuum()).plain<double>();
//					Terms.push_local(loc,std::sqrt(3.)*P.get<Eigen::ArrayXXd>("Jfull")(loc,loc),Ssqrt);
				}
				else
				{
					auto PsiDagUp_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(UP,0), {2});
					auto PsiDagDn_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(DN,0), {2});
					auto Sign_loc     = OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign(), {1});
					auto PsiUp_hop    = OperatorType::outerprod(B[hop].Id().structured(), F[hop].psi(UP,0), {2});
					auto PsiDn_hop    = OperatorType::outerprod(B[hop].Id().structured(), F[hop].psi(DN,0), {2});
					
					auto Otmp_loc = OperatorType::prod(PsiDagUp_loc, Sign_loc, {2});
					
					Terms.push(hop-loc, loc, -P.get<Eigen::ArrayXXd>("tFull")(loc,hop) * sqrt(2.),
					           Otmp_loc.plain<double>(), TransOps, PsiUp_hop.plain<double>());
					
					Otmp_loc = OperatorType::prod(PsiDagDn_loc, Sign_loc, {2});
					
					Terms.push(hop-loc, loc, -P.get<Eigen::ArrayXXd>("tFull")(loc,hop) * sqrt(2.),
					           Otmp_loc.plain<double>(), TransOps, PsiDn_hop.plain<double>());
				}
			}
			Terms.save_label(loc, "tᵢⱼ");
		}
		
		// V∥
		param2d Vpara = P.fill_array2d<double>("V", "Vpara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, Vpara.label);
		
		// t∥
		if (!P.HAS("tFull"))
		{
			param2d tPara = P.fill_array2d<double>("t", "tPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
			Terms.save_label(loc, tPara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
				{
					auto PsiDagUp_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(UP,alfa), {2});
					auto PsiDagDn_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(DN,alfa), {2});
					auto Sign_loc     = OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign(), {1});
					auto PsiUp_lp1    = OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].psi(UP,beta), {2});
					auto PsiDn_lp1    = OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].psi(DN,beta), {2});
					
					auto Otmp_loc = OperatorType::prod(PsiDagUp_loc, Sign_loc, {2});
					
					Terms.push_tight(loc, -tPara(alfa,beta) * sqrt(2.), Otmp_loc.plain<double>(), PsiUp_lp1.plain<double>());
					
					if (tPara(alfa,beta) != 0.)
					{
						assert(F[loc].sublattice() != F[lp1].sublattice());
					}
					
					//c†DNcDN
					Otmp_loc = OperatorType::prod(PsiDagDn_loc, Sign_loc, {2});
					
					Terms.push_tight(loc, -tPara(alfa,beta) * sqrt(2.), Otmp_loc.plain<double>(), PsiDn_lp1.plain<double>());
				}
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
				                 OperatorType::outerprod(B[loc].Scomp(SP,alfa).structured(), F[loc].Id(), {1}).plain<double>(),
				                 OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].Sm(beta), {1}).plain<double>()
				                );
				Terms.push_tight(loc, 0.5*InextPara(alfa,beta),
				                 OperatorType::outerprod(B[loc].Scomp(SM,alfa).structured(), F[loc].Id(), {1}).plain<double>(),
				                 OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].Sp(beta), {1}).plain<double>()
				                );
				Terms.push_tight(loc, InextPara(alfa,beta),
				                 OperatorType::outerprod(B[loc].Scomp(SZ,alfa).structured(), F[loc].Id(), {1}).plain<double>(),
				                 OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].Sz(beta), {1}).plain<double>()
				                );
			}
		}
		
		param2d IprevPara = P.fill_array2d<double>("Iprev", "IprevPara", {Fprev_orbitals, Borbitals}, loc%Lcell);
		Terms.save_label(loc, IprevPara.label);
		
		if (lm1 < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Fprev_orbitals;  ++alfa)
			for (std::size_t beta=0; beta<Borbitals;       ++beta)
			{
				Terms.push_tight(lm1, 0.5*IprevPara(alfa,beta),
				                 OperatorType::outerprod(B[lm1].Id().structured(), F[lm1].Sm(alfa), {1}).plain<double>(),
				                 OperatorType::outerprod(B[loc].Scomp(SP,beta).structured(), F[loc].Id(), {1}).plain<double>()
				                );
				Terms.push_tight(lm1, 0.5*IprevPara(alfa,beta),
				                 OperatorType::outerprod(B[lm1].Id().structured(), F[lm1].Sp(alfa), {1}).plain<double>(),
				                 OperatorType::outerprod(B[loc].Scomp(SM,beta).structured(), F[loc].Id(), {1}).plain<double>()
				                );
				Terms.push_tight(lm1, IprevPara(alfa,beta),
				                 OperatorType::outerprod(B[lm1].Id().structured(), F[lm1].Sz(alfa), {1}).plain<double>(),
				                 OperatorType::outerprod(B[loc].Scomp(SZ,beta).structured(), F[loc].Id(), {1}).plain<double>()
				                );
			}
		}
		
		// NN 3-orbital spin exchange terms
		
		param2d I3nextPara = P.fill_array2d<double>("I3next", "I3nextPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		Terms.save_label(loc, I3nextPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Forbitals;      ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				assert(Borbitals == 1);
				
				auto Sm_PsiDagUp_loc = OperatorType::outerprod(B[loc].Scomp(SM,0).structured(), F[loc].psidag(UP,alfa), {2});
				auto Sp_PsiDagDn_loc = OperatorType::outerprod(B[loc].Scomp(SP,0).structured(), F[loc].psidag(DN,alfa), {2});
				auto Sign_loc        = OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign(), {1});
				auto PsiUp_lp1       = OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].psi(UP,beta), {2});
				auto PsiDn_lp1       = OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].psi(DN,beta), {2});
				
				auto Otmp_loc = OperatorType::prod(Sm_PsiDagUp_loc, Sign_loc, {2});
				
				Terms.push_tight(loc, 0.5 * sqrt(2.) * I3nextPara(alfa,beta), Otmp_loc.plain<double>(), PsiDn_lp1.plain<double>());
				
				Otmp_loc = OperatorType::prod(Sp_PsiDagDn_loc, Sign_loc, {2});
				
				Terms.push_tight(loc, 0.5 * sqrt(2.) * I3nextPara(alfa,beta), Otmp_loc.plain<double>(), PsiUp_lp1.plain<double>());
				
				Terms.push_tight(loc, I3nextPara(alfa,beta),
				                      OperatorType::outerprod(B[loc].Scomp(SZ,0).structured(), F[loc].Id(), {1}).plain<double>(),
				                      OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].Sz(beta), {1}).plain<double>()
				                );
			}
		}
		
		param2d I3prevPara = P.fill_array2d<double>("I3prev", "I3prevPara", {Fprev_orbitals, Forbitals}, loc%Lcell);
		Terms.save_label(loc, I3prevPara.label);
		
		if (lm1 < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<Fprev_orbitals;  ++alfa)
			for (std::size_t beta=0; beta<Forbitals;       ++beta)
			{
				assert(Borbitals == 1);
				
				auto Sm_PsiDagUp_lm1 = OperatorType::outerprod(B[lm1].Scomp(SM,0).structured(), F[lm1].psidag(UP,alfa), {2});
				auto Sp_PsiDagDn_lm1 = OperatorType::outerprod(B[lm1].Scomp(SP,0).structured(), F[lm1].psidag(DN,alfa), {2});
				auto Sign_lm1        = OperatorType::outerprod(B[lm1].Id().structured(), F[lm1].sign(), {1});
				auto PsiUp_loc       = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psi(UP,beta), {2});
				auto PsiDn_loc       = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psi(DN,beta), {2});
				
				auto Otmp_lm1 = OperatorType::prod(Sm_PsiDagUp_lm1, Sign_lm1, {2});
				
				Terms.push_tight(lm1, 0.5 * sqrt(2.) * I3nextPara(alfa,beta), Otmp_lm1.plain<double>(), PsiDn_loc.plain<double>());
				
				Otmp_lm1 = OperatorType::prod(Sp_PsiDagDn_lm1, Sign_lm1, {2});
				
				Terms.push_tight(lm1, 0.5 * sqrt(2.) * I3nextPara(alfa,beta), Otmp_lm1.plain<double>(), PsiUp_loc.plain<double>());
				
				Terms.push_tight(lm1, I3nextPara(alfa,beta),
				                      OperatorType::outerprod(B[lm1].Scomp(SZ,0).structured(), F[lm1].Id(), {1}).plain<double>(),
				                      OperatorType::outerprod(B[loc].Id().structured(), F[loc].Sz(beta), {1}).plain<double>()
				                );
			}
		}
		
		// tPrime
		if (!P.HAS("tFull") and P.HAS("tPrime",loc%Lcell))
		{
			assert(F[loc].sublattice() != F[lp2].sublattice());
			
			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {Forbitals, F3next_orbitals}, loc%Lcell);
			Terms.save_label(loc, tPrime.label);
			
			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
			{
				auto Sign_loc     = OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign(), {1});
				auto Sign_lp1     = OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].sign(), {1});
				
				vector<SiteOperator<Symmetry,double> > TransOps(1);
				TransOps[0] = Sign_lp1.plain<double>();
				
				for (std::size_t alfa=0; alfa<Forbitals;       ++alfa)
				for (std::size_t beta=0; beta<F3next_orbitals; ++beta)
				{
					auto PsiDagUp_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(UP,alfa), {2});
					auto PsiDagDn_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(DN,alfa), {2});
					auto PsiUp_lp2    = OperatorType::outerprod(B[lp2].Id().structured(), F[lp2].psi(UP,beta), {2});
					auto PsiDn_lp2    = OperatorType::outerprod(B[lp2].Id().structured(), F[lp2].psi(DN,beta), {2});
					
					auto PsiDagUp_loc_signed = OperatorType::prod(PsiDagUp_loc, Sign_loc, {2});
					auto PsiDagDn_loc_signed = OperatorType::prod(PsiDagDn_loc, Sign_loc, {2});
					
					Terms.push(2, loc, -tPrime(alfa,beta)*sqrt(2.), PsiDagUp_loc_signed.plain<double>(), TransOps, PsiUp_lp2.plain<double>());
					Terms.push(2, loc, -tPrime(alfa,beta)*sqrt(2.), PsiDagDn_loc_signed.plain<double>(), TransOps, PsiDn_lp2.plain<double>());
				}
			}
		}
		
		// tPrimePrime
		if (!P.HAS("tFull") and P.HAS("tPrimePrime",loc%Lcell))
		{
			assert(F[loc].sublattice() != F[lp3].sublattice());
			
			param2d tPrimePrime = P.fill_array2d<double>("tPrimePrime", "tPrimePrime_array", {Forbitals, F3next_orbitals}, loc%Lcell);
			Terms.save_label(loc, tPrimePrime.label);
			
			if (loc < N_sites-3 or !P.get<bool>("OPEN_BC"))
			{
				auto Sign_loc     = OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign(), {1});
				auto Sign_lp1     = OperatorType::outerprod(B[lp1].Id().structured(), F[lp1].sign(), {1});
				auto Sign_lp2     = OperatorType::outerprod(B[lp2].Id().structured(), F[lp2].sign(), {1});
				
				vector<SiteOperator<Symmetry,double> > TransOps(2);
				TransOps[0] = Sign_lp1.plain<double>();
				TransOps[1] = Sign_lp2.plain<double>();
				
				for (std::size_t alfa=0; alfa<Forbitals;       ++alfa)
				for (std::size_t beta=0; beta<F3next_orbitals; ++beta)
				{
					auto PsiDagUp_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(UP,alfa), {2});
					auto PsiDagDn_loc = OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(DN,alfa), {2});
					auto PsiUp_lp3    = OperatorType::outerprod(B[lp3].Id().structured(), F[lp3].psi(UP,beta), {2});
					auto PsiDn_lp3    = OperatorType::outerprod(B[lp3].Id().structured(), F[lp3].psi(DN,beta), {2});
					
					auto PsiDagUp_loc_signed = OperatorType::prod(PsiDagUp_loc, Sign_loc, {2});
					auto PsiDagDn_loc_signed = OperatorType::prod(PsiDagDn_loc, Sign_loc, {2});
					
					Terms.push(3, loc, -tPrimePrime(alfa,beta)*sqrt(2.), PsiDagUp_loc_signed.plain<double>(), TransOps, PsiUp_lp3.plain<double>());
					Terms.push(3, loc, -tPrimePrime(alfa,beta)*sqrt(2.), PsiDagDn_loc_signed.plain<double>(), TransOps, PsiDn_lp3.plain<double>());
				}
			}
		}
	}
}

//HamiltonianTermsXd<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
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
//			//-------------------------------------------------------------------------------------------------------------------------------------//
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), F.cdag(UP,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.c(UP,j))));
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), F.cdag(DN,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.c(DN,j))));
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), -1.*F.c(UP,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.cdag(UP,j))));
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), -1.*F.c(DN,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.cdag(DN,j))));
//			
//			// Mout += -t*std::sqrt(2.)*(Operator::prod(psidag(UP,i),psi(UP,i+1),{1})+Operator::prod(psidag(DN,i),psi(DN,i+1),{1}));
//			//-------------------------------------------------------------------------------------------------------------------------------------//
//			
//			//c†UPcUP
//			
//			auto Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(UP,i), {2}),
//										   OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign()      , {1}),
//										   {2});
//			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
//											 Otmp.plain<double>(),
//											 OperatorType::outerprod(B[loc].Id().structured(), F[loc].psi(UP,i), {2}).plain<double>()));
//			
//			//c†DNcDN
//			Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(DN,i), {2}),
//									  OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign()      , {1}),
//									  {2});
//			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
//											 Otmp.plain<double>(),
//											 OperatorType::outerprod(B[loc].Id().structured(), F[loc].psi(DN,i), {2}).plain<double>()));
//			
//			//-cUPc†UP
//			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id().structured(),F.psi(UP,i),{2}),
//			// 						  OperatorType::outerprod(B.Id().structured(),F.sign()   ,{1}),
//			// 						  {2});
//			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
//			// 								 Otmp.plain<double>(),
//			// 								 OperatorType::outerprod(B.Id().structured(),F.psidag(UP,j),{2}).plain<double>()));
//			
//			//-cDNc†DN
//			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id().structured(),F.psi(DN,i),{2}),
//			// 						  OperatorType::outerprod(B.Id().structured(),F.sign()   ,{1}),
//			// 						  {2});
//			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
//			// 								 Otmp.plain<double>(),
//			// 								 OperatorType::outerprod(B.Id().structured(),F.psidag(DN,j),{2}).plain<double>()));
//			
//			//-------------------------------------------------------------------------------------------------------------------------------------//
//		}
//	}
//	
//	// local terms
//	
//	// t⟂
//	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(tPerplabel);
//	
//	// Hubbard U
//	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
//	save_label(Ulabel);

//	// Bx substrate
//	auto [Bxsub,Bxsuborb,Bxsublabel] = P.fill_array1d<double>("Bxsub","Bxsuborb",F[loc].orbitals(),loc);
//	save_label(Bxsublabel);
//	
//	// Bx impurities
//	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F[loc].orbitals(),loc);
//	save_label(Bxlabel);
//	
//	// Kx anisotropy
//	auto [Kx,Kxorb,Kxlabel] = P.fill_array1d<double>("Kx","Kxorb",B[loc].orbitals(),loc);
//	save_label(Kxlabel);
//	
//	// Bz substrate
//	auto [Bzsub,Bzsuborb,Bzsublabel] = P.fill_array1d<double>("Bzsub","Bzsuborb",F[loc].orbitals(),loc);
//	save_label(Bzsublabel);
//	
//	// Bz impurities
//	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F[loc].orbitals(),loc);
//	save_label(Bzlabel);
//	
//	// Kz anisotropy
//	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",B[loc].orbitals(),loc);
//	save_label(Kzlabel);
//	
//	// OperatorType KondoHamiltonian({1},B[loc].get_structured_basis().combine(F[loc].get_basis()));
//	
//	ArrayXXd Jxyperp = B[loc].ZeroHopping();
//	ArrayXXd Jzperp  = B[loc].ZeroHopping();
//	ArrayXXd Dyperp  = B[loc].ZeroHopping();
//	
//	//set Heisenberg part of Kondo Hamiltonian
//	auto KondoHamiltonian = OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp).structured(),
//	                                                F[loc].Id(),
//	                                                {1});
//	
//	ArrayXXd Vperp      = F[loc].ZeroHopping();
//	ArrayXXd Jxysubperp = F[loc].ZeroHopping();
//	ArrayXXd Jzsubperp  = F[loc].ZeroHopping();
//	
//	//set Hubbard part of Kondo Hamiltonian
//	KondoHamiltonian += OperatorType::outerprod(B[loc].Id().structured(),
//	                                            F[loc].HubbardHamiltonian(Uorb,tPerp,Vperp,Jxysubperp,Jzsubperp,Bzsuborb,Bxsuborb),
//	                                            {1});
//	
//	//set Heisenberg part of Hamiltonian
////	KondoHamiltonian += OperatorType::outerprod(B.HeisenbergHamiltonian(0.,P.get<bool>("CYLINDER")),F[loc].Id(),{1,0});
//	
//	// Kondo-J
//	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
//	save_label(Jlabel);
//	
//	//set interaction part of Hamiltonian.
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	{
//		if (Jorb(i) != 0.)
//		{
//			KondoHamiltonian +=     Jorb(i) * OperatorType::outerprod(B[loc].Scomp(SZ,i).structured(), F[loc].Sz(i), {1});
//			KondoHamiltonian += 0.5*Jorb(i) * OperatorType::outerprod(B[loc].Scomp(SP,i).structured(), F[loc].Sm(i), {1});
//			KondoHamiltonian += 0.5*Jorb(i) * OperatorType::outerprod(B[loc].Scomp(SM,i).structured(), F[loc].Sp(i), {1});
//		}
//	}
//	
//	Terms.name = "Kondo U(0)⊗SU(2)";
//	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
//	
//	return Terms;
//}

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
Simp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	assert(locx < this->N_sites);
	std::stringstream ss;
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
	
	auto Sop = OperatorType::outerprod(B[locx].Scomp(Sa,locy).structured(), F[locx].Id(), {1});
	
	Mout.setLocal(locx, Sop.plain<double>());
	return Mout;
}

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	assert(locx < this->N_sites);
	std::stringstream ss;
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
	
	auto Sop = OperatorType::outerprod(B[locx].Id().structured(), F[locx].Scomp(Sa,locy), {1});
	
	Mout.setLocal(locx, Sop.plain<double>());
	return Mout;
}

/*Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::*/
/*SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const*/
/*{*/
/*	assert(locx1 < this->N_sites and locx2 < this->N_sites);*/
/*	std::stringstream ss;*/
/*	*/
/*	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());*/
/*	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }*/
/*	*/
/*	auto Sop1 = OperatorType::outerprod(B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Id(), {1});*/
/*	auto Sop2 = OperatorType::outerprod(B[locx1].Id().structured(), F[locx2].Scomp(SOP2,locy2), {1});*/
/*	*/
/*	Mout.setLocal({locx1,locx2}, {Sop1.plain<double>(),Sop2.plain<double>()});*/
/*	return Mout;*/
/*}*/

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, 
           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
           const OperatorType &Op1, const OperatorType &Op2, 
           bool BOTH_HERMITIAN) const
{
	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
	stringstream ss;
	ss << name1 << "(" << locx1 << "," << locy1 << ")"
	   << name2 << "(" << locx2 << "," << locy2 << ")";
	
	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
	OperatorType Op1Ext;
	OperatorType Op2Ext;
	
	Mpo<Symmetry> Mout(F.size(), Symmetry::qvacuum(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l);}
	
	if (SUBSYS == SUB)
	{
		Op1Ext = OperatorType::outerprod(B[locx1].Id().structured(), Op1, {1});
		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
	}
	else if (SUBSYS == IMP)
	{
		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), {1});
	}
	else if (SUBSYS == IMPSUB and locx1 != locx2)
	{
		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
	}
	else if (SUBSYS == IMPSUB and locx1 == locx2)
	{
		OperatorType OpExt = OperatorType::outerprod(Op1, Op2, {1});
		
		Mout.setLocal(locx1, OpExt.plain<double>());
		return Mout;
	}
	
	Mout.setLocal({locx1,locx2}, {Op1Ext.plain<double>(),Op2Ext.plain<double>()});
	return Mout;
}

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "imp";
	stringstream ss2; ss2 << SOP2 << "sub";
	
	return make_corr(IMPSUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Scomp(SOP2,locy2));
}

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "imp";
	stringstream ss2; ss2 << SOP2 << "imp";
	
	return make_corr(IMP, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), B[locx2].Scomp(SOP2,locy2).structured());
}

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "sub";
	stringstream ss2; ss2 << SOP2 << "sub";
	
	return make_corr(SUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
}

} //end namespace VMPS

#endif
