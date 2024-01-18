#ifndef HUBBARDMODELU1_H_COMPLEX
#define HUBBARDMODELU1_H_COMPLEX

#include "models/PeierlsHubbardU1xU1.h"

namespace VMPS
{
class PeierlsHubbardU1 : public Mpo<Sym::U1<Sym::ChargeU1>,complex<double> >,
                         public HubbardObservables<Sym::U1<Sym::ChargeU1>,complex<double> >,
                         public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::ChargeU1> Symmetry;
	MAKE_TYPEDEFS(PeierlsHubbardU1)
	typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
//private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	PeierlsHubbardU1() : Mpo(){};
	
	PeierlsHubbardU1(Mpo<Symmetry,complex<double>> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry,complex<double>>(Mpo_input),
	 HubbardObservables(this->N_sites,params,PeierlsHubbardU1::defaults),
	 ParamReturner(PeierlsHubbardU1::sweep_defaults)
	{
		ParamHandler P(params,PeierlsHubbardU1::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->calc(P.get<size_t>("maxPower"));
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	PeierlsHubbardU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, 
	                           PushType<SiteOperator<Symmetry_,complex<double>>,complex<double>> &pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static qarray<1> singlet (int N=0) {return qarray<1>{N};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> PeierlsHubbardU1::defaults = 
{
	{"t",1.+0.i}, {"tPrime",0.+0.i}, {"tRung",1.+0.i}, 
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, {"Bx",0.}, {"By",0.},
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"X",0.}, {"Xperp",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",1},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> PeierlsHubbardU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Mincr_abs", 50ul}, {"Mincr_per", 2ul}, {"Mincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",1ul},
	{"Minit",2ul}, {"Qinit",2ul}, {"Mlimit",1000ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST",DMRG::CONVTEST::VAR_2SITE}
};

PeierlsHubbardU1::
PeierlsHubbardU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double>> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,PeierlsHubbardU1::defaults),
 ParamReturner(PeierlsHubbardU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	this->set_name("Peierls-Hubbard");
	
	PushType<SiteOperator<Symmetry,complex<double>>,complex<double>> pushlist;
	std::vector<std::vector<std::string>> labellist;
	PeierlsHubbardU1xU1::set_operators(F, P, pushlist, labellist, boundary);
	add_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void PeierlsHubbardU1::
add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,complex<double>>,complex<double>> &pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		std::size_t lp1 = (loc+1)%N_sites;
		std::size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals = F[loc].orbitals();
		std::size_t next_orbitals = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		param1d Bx = P.fill_array1d<double>("Bx", "Bxorb", orbitals, loc%Lcell);
		labellist[loc].push_back(Bx.label);
		auto H_Bx = F[loc].template coupling_Bx<complex<double>,Symmetry>(Bx.a);
		
		param1d By = P.fill_array1d<double>("By", "Byorb", orbitals, loc%Lcell);
		labellist[loc].push_back(By.label);
		auto H_By = F[loc].coupling_By(By.a); // already complex
		
		auto Hloc = Mpo<Symmetry_,complex<double>>::get_N_site_interaction(H_Bx+H_By);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.+0.i));
	}
}

} // end namespace VMPS::models

#endif
