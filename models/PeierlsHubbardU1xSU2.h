#ifndef HUBBARDMODELU1XSU2CHARGE_H_
#define HUBBARDMODELU1XSU2CHARGE_H_

#include "models/HubbardSU2xU1.h"
#include "symmetry/SU2.h"
#include "bases/FermionBase.h"
#include "models/HubbardObservables.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class PeierlsHubbardU1xSU2 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::SU2<Sym::ChargeSU2> >,complex<double> >,
                                   public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::SU2<Sym::ChargeSU2> >,complex<double> >,
                                   public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	MAKE_TYPEDEFS(PeierlsHubbardU1xSU2)
	typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	PeierlsHubbardU1xSU2() : Mpo(){};
	
	PeierlsHubbardU1xSU2(Mpo<Symmetry, complex<double> > &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry,complex<double> >(Mpo_input),
	 HubbardObservables(this->N_sites,params,PeierlsHubbardU1xSU2::defaults),
	 ParamReturner(PeierlsHubbardU1xSU2::sweep_defaults)
	{
		ParamHandler P(params,PeierlsHubbardU1xSU2::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->calc(P.get<size_t>("maxPower"));
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	PeierlsHubbardU1xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	template<typename Symmetry_> 
	static void set_operators (const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,complex<double>>,complex<double>>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static qarray<2> singlet (int N=0, int L=0) {return qarray<2>{0,N};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 1;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> PeierlsHubbardU1xSU2::defaults = 
{
	{"t",1.+0.i}, {"tPrime",0.i}, {"tRung",1.+0.i}, {"tPrimePrime",0.i},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.},
	{"Jz",0.}, {"Jzrung",0.}, {"Jxy",0.}, {"Jxyrung",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"Bz",0.}, {"Bx",0.},
	{"X",0.}, {"Xrung",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",0},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> PeierlsHubbardU1xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",6ul},
	{"Minit",1ul}, {"Qinit",1ul}, {"Mlimit",500ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

PeierlsHubbardU1xSU2::
PeierlsHubbardU1xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double> > (L, qarray<Symmetry::Nq>({0,1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,PeierlsHubbardU1xSU2::defaults),
 ParamReturner(PeierlsHubbardU1xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	param1d U = P.fill_array1d<double>("U", "Uorb", F[0].orbitals(), 0);
	if (isfinite(U.a.sum()))
	{
		this->set_name("Hubbard");
	}
	else
	{
		this->set_name("U=∞-Hubbard");
	}
	
	PushType<SiteOperator<Symmetry,complex<double>>,complex<double>> pushlist;
	std::vector<std::vector<std::string>> labellist;
	PeierlsHubbardU1xSU2::set_operators(F, G, P, pushlist, labellist, boundary); // F, G are set in HubbardObservables
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void PeierlsHubbardU1xSU2::
set_operators (const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P, PushType<SiteOperator<Symmetry_,complex<double>>,complex<double>>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		
		std::size_t orbitals       = F[loc].orbitals();
		std::size_t next_orbitals  = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		std::size_t nnextn_orbitals = F[lp3].orbitals();
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss.str());
		
		auto push_full = [&N_sites, &loc, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
																					 const vector<SiteOperatorQ<Symmetry_,MatrixType>> &first,
																					 const vector<vector<SiteOperatorQ<Symmetry_,MatrixType>>> &last,
																					 vector<complex<double>> factor, 
																					 vector<bool> CONJ,
																					 bool FERMIONIC) -> void
		{
			ArrayXXcd Full = P.get<Eigen::ArrayXXcd>(xxxFull);
			vector<vector<std::pair<size_t,complex<double>> > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t j=0; j<first.size(); j++)
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				complex<double> value = R[loc][h].second;
				
				if (range != 0)
				{
					vector<SiteOperatorQ<Symmetry_,MatrixType> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						if (FERMIONIC) {ops[i] = F[(loc+i)%N_sites].sign().template cast<complex<double>>();}
						else {ops[i] = F[(loc+i)%N_sites].Id().template cast<complex<double>>();}
					}
					ops[range] = last[j][(loc+range)%N_sites];
					complex<double> total_value = factor[j] * value;
					if (CONJ[j]) total_value = conj(total_value);
					pushlist.push_back(std::make_tuple(loc, ops, total_value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};
		
		if (P.HAS("tFull"))
		{
			SiteOperatorQ<Symmetry_,MatrixType> cdagup_sign_local = (F[loc].cdag(UP,G[loc],0) * F[loc].sign()).template cast<complex<double>>();;
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cup_ranges(N_sites);
			SiteOperatorQ<Symmetry_,MatrixType> cdagdn_sign_local = (F[loc].cdag(DN,G[loc],0) * F[loc].sign()).template cast<complex<double>>();;
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cdn_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				//auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cup_ranges[i] = F[i].c(UP,G[i],0).template cast<complex<double> >();
			}
			for (size_t i=0; i<N_sites; i++)
			{
				//auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cdn_ranges[i] = F[i].c(DN,G[i],0).template cast<complex<double> >();
			}
			
			vector<SiteOperatorQ<Symmetry_,MatrixType> >          frst {cdagup_sign_local, cdagdn_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {cup_ranges,        cdn_ranges};
			push_full("tFull", "tᵢⱼ", frst, last, {-std::sqrt(2.),-std::sqrt(2.)}, {false, false}, PROP::FERMIONIC);
		}
		if (P.HAS("Jfull"))
		{
			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].Sp(0).template cast<complex<double>>(), 
			                                                    F[loc].Sm(0).template cast<complex<double>>(), 
			                                                    F[loc].Sz(0).template cast<complex<double>>()};
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Sm_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				Sp_ranges[i] = F[i].Sp(0).template cast<complex<double>>();;
				Sm_ranges[i] = F[i].Sm(0).template cast<complex<double>>();;
				Sz_ranges[i] = F[i].Sz(0).template cast<complex<double>>();;
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {Sm_ranges, Sp_ranges, Sz_ranges};
			push_full("Jfull", "Jᵢⱼ", first, last, {0.5,0.5,1.}, {false, false, false}, PROP::BOSONIC);
		}
		// Local terms: U, t0, μ, t⟂, V⟂, J⟂
		
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", orbitals, loc%Lcell);
		param1d V = P.fill_array1d<double>("V", "Vorb", orbitals, loc%Lcell);
		//param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		//param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param2d tPerp = P.fill_array2d<complex<double> >("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jxyperp = P.fill_array2d<double>("JxyRung", "Jxy", "JxyPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp = P.fill_array2d<double>("JzRung", "Jz", "JzPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("JRung", "J", "JPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(Uph.label);
		//labellist[loc].push_back(t0.label);
		//labellist[loc].push_back(mu.label);
		labellist[loc].push_back(tPerp.label);
		labellist[loc].push_back(Jxyperp.label);
		labellist[loc].push_back(Jzperp.label);
		labellist[loc].push_back(Jperp.label);
		labellist[loc].push_back(Bz.label);
		labellist[loc].push_back(Vperp.label);
		
		auto Hloc = Mpo<Symmetry_,complex<double> >::get_N_site_interaction
		(
			F[loc].template HubbardHamiltonian<complex<double>,Symmetry_>(
				Uph.a.cast<complex<double>>(), 
				tPerp.a.cast<complex<double>>(),
				Vperp.a.cast<complex<double>>(),
				Jzperp.a.cast<complex<double>>(),
				Jxyperp.a.cast<complex<double>>(),
				Bz.a.cast<complex<double>>())
		);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.+0.i));
	}
}

} // end namespace VMPS::models

#endif
