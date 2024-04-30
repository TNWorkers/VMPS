#ifndef KONDOMODEL_SU2XSU2_H_
#define KONDOMODEL_SU2XSU2_H_

#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
#include "symmetry/SU2.h"
#include "bases/SpinBase.h"
#include "bases/FermionBase.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS
#include "models/KondoObservables.h"

namespace VMPS
{

class KondoU1xSU2 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::SU2<Sym::ChargeSU2> > ,double>,
					 public KondoObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::SU2<Sym::ChargeSU2> > >,
					 public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	MAKE_TYPEDEFS(KondoU1xSU2)
	static constexpr MODEL_FAMILY FAMILY = MODEL_FAMILY::KONDO;
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	KondoU1xSU2 (): Mpo(), KondoObservables(), ParamReturner(KondoU1xSU2::sweep_defaults) {};
	KondoU1xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	static qarray<2> singlet (int N=0, int L=0)
	{
		assert(N%2==0);
		int T = abs(0.5*(N-L));
		return qarray<2>{0,2*T+1};
	};
	
	template<typename Symmetry_> 
	static void set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
	static const map<string,any> sweep_defaults;
};

const std::map<string,std::any> KondoU1xSU2::defaults =
{
	{"t",1.}, {"tRung",0.}, {"tPrime",0.}, {"tPrimePrime",0.},
	{"J",1.}, {"Jz",0.}, {"U",0.}, 
	{"V",0.}, {"Vrung",0.},
	{"Bz",0.}, {"Bzsub",0.}, {"Kz",0.},
	{"Inext",0.}, {"Iprev",0.}, {"I3next",0.}, {"I3prev",0.}, {"I3loc",0.}, 
	{"D",2ul}, {"maxPower",1ul}, {"CYLINDER",false}, {"Ly",1ul}, {"LyF",1ul},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",1}, 
};

const map<string,any> KondoU1xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",16ul}, {"eps_svd",1e-7},
	{"Dincr_abs",5ul}, {"Dincr_per",2ul}, {"Dincr_rel",1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",10ul},
	{"Dinit",5ul}, {"Qinit",6ul}, {"Dlimit",200ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST",DMRG::CONVTEST::VAR_2SITE}
};

KondoU1xSU2::
KondoU1xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0,1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 KondoObservables(L,params,KondoU1xSU2::defaults),
 ParamReturner(KondoU1xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("LyF",l%Lcell);
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);
	}
	
	this->set_name("Kondo");
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	set_operators(B, F, G, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_> 
void KondoU1xSU2::
set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P,
               PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
/*		auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));*/
/*		auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));*/
/*		auto Glp2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp2)));*/
		
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[lp1].orbitals();
		std::size_t Fnextn_orbitals = F[lp2].orbitals();
		
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[lp1].orbitals();
		std::size_t Bnextn_orbitals = B[lp2].orbitals();
		
		stringstream Slabel;
		Slabel << "S=" << print_frac_nice(frac(B[loc].get_D()-1,2));
		labellist[loc].push_back(Slabel.str());
		
		auto push_full = [&N_sites, &loc, &B, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
																						 const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > &first,
																						 const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > &last,
																						 vector<double> factor, bool FERMIONIC) -> void
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>(xxxFull);
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t j=0; j<first.size(); j++)
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				if (range != 0)
				{
					vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						if (FERMIONIC) {ops[i] = kroneckerProduct(B[(loc+i)%N_sites].Id(), F[(loc+i)%N_sites].sign());}
						else {ops[i] = kroneckerProduct(B[(loc+i)%N_sites].Id(), F[(loc+i)%N_sites].Id());}
					}
					ops[range] = last[j][(loc+range)%N_sites];
					pushlist.push_back(std::make_tuple(loc, ops, factor[j] * value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};
		
		if (P.HAS("tFull"))
		{
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagUP_sign_local = kroneckerProduct(B[loc].Id(),(F[loc].cdag(UP,G[loc],0) * F[loc].sign()));
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagDN_sign_local = kroneckerProduct(B[loc].Id(),(F[loc].cdag(DN,G[loc],0) * F[loc].sign()));
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cUP_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cDN_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				//auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cUP_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].c(UP,G[i],0));
			}
			for (size_t i=0; i<N_sites; i++)
			{
				//auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cDN_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].c(DN,G[i],0));
			}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {cdagUP_sign_local, cdagDN_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {cUP_ranges, cDN_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.), -std::sqrt(2.)}, PROP::FERMIONIC);
		}
		
		if (!P.HAS("tFull"))
		{
			param2d tPara = P.fill_array2d<double>("t", "tPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPara.label);
			
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
				{
					auto PsiDagUp_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,G[loc],alfa));
					auto PsiDagDn_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,G[loc],alfa));
					auto Sign_loc     = kroneckerProduct(B[loc].Id(), F[loc].sign());
					auto PsiUp_lp1    = kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,G[lp1],beta));
					auto PsiDn_lp1    = kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,G[lp1],beta));
					
					auto Otmp_loc = PsiDagUp_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, PsiUp_lp1), -tPara(alfa,beta) * sqrt(2.)) );
					
					//c†DNcDN
					Otmp_loc = PsiDagDn_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, PsiDn_lp1), -tPara(alfa,beta) * sqrt(2.)) );
				}
			}
		}
		
		// local terms
		
		// Kondo-J
		param1d J = P.fill_array1d<double>("J", "Jorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(J.label);
		
		// Kondo-Jz
		param1d Jz = P.fill_array1d<double>("Jz", "Jzorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(Jz.label);
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung", "t", "tPerp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		labellist[loc].push_back(tPerp.label);
		
		// Hubbard-U
		param1d U = P.fill_array1d<double>("U", "Uorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(U.label);
		
		// Bz substrate
		param1d Bzsub = P.fill_array1d<double>("Bzsub", "Bzsuborb", Forbitals, loc%Lcell);
		labellist[loc].push_back(Bzsub.label);
		
		// Bz impurities
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", Borbitals, loc%Lcell);
		labellist[loc].push_back(Bz.label);
		
		// Kz anisotropy
		param1d Kz = P.fill_array1d<double>("Kz","Kzorb", Borbitals, loc%Lcell);
		labellist[loc].push_back(Kz.label);
		
		ArrayXXd muPerp  = B[loc].ZeroHopping();
		ArrayXXd nuPerp  = B[loc].ZeroHopping();
		ArrayXXd Jxyperp = B[loc].ZeroHopping();
		ArrayXXd Jzperp  = B[loc].ZeroHopping();
		
		//set Heisenberg part of Kondo Hamiltonian
		auto KondoHamiltonian = kroneckerProduct(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bz.a,muPerp,nuPerp,Kz.a), F[loc].Id());
		
		ArrayXXd Vperp      = F[loc].ZeroHopping();
		ArrayXXd Jxysubperp = F[loc].ZeroHopping();
		ArrayXXd Jzsubperp  = F[loc].ZeroHopping();
		
		//set Hubbard part of Kondo Hamiltonian
		KondoHamiltonian += kroneckerProduct(B[loc].Id(), F[loc].HubbardHamiltonian(U.a,tPerp.a,Vperp,Jzsubperp,Jxysubperp,Bzsub.a));
		
		//set Kondo part of Hamiltonian
		for (int alfa=0; alfa<Forbitals; ++alfa)
		{
			if (J(alfa) != 0.)
			{
				assert(Borbitals == Forbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
				KondoHamiltonian += 0.5*J(alfa) * kroneckerProduct(B[loc].Scomp(SP,alfa), F[loc].Sm(alfa));
				KondoHamiltonian += 0.5*J(alfa) * kroneckerProduct(B[loc].Scomp(SM,alfa), F[loc].Sp(alfa));
				KondoHamiltonian +=     J(alfa) * kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Sz(alfa));
			}
			if (Jz(alfa) != 0.)
			{
				KondoHamiltonian += Jz(alfa) * kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Sz(alfa));
			}
		}
		pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(KondoHamiltonian), 1.));
	}
}

} //end namespace VMPS

#endif
