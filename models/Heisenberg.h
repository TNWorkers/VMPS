#ifndef VANILLA_HEISENBERG
#define VANILLA_HEISENBERG

#include "models/HeisenbergU1.h"
#include "symmetry/U0.h"

namespace VMPS
{

/** \class Heisenberg
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model
  *
  * MPO representation of
  * \f[
  * H =  J \sum_{<ij>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right) 
  *      J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
  *     -B_z \sum_i S^z_i
  *     -B_x \sum_i S^x_i
  *     +K_z \sum_i \left(S^z_i\right)^2
  *     +K_x \sum_i \left(S^x_i\right)^2
  *     -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  *     -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  * \f]
  *
  * \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  * \note Uses no symmetries. Any parameter constellations are allowed. For variants with symmetries, see VMPS::HeisenbergU1 or VMPS::HeisenbergSU2.
  * \note The default variable settings can be seen in \p Heisenberg::defaults.
  * \note \f$J>0\f$ is antiferromagnetic
  * \note This is the real version of the Heisenbergmodel without symmetries, so \f$J_x = J_y\f$ is mandatory. For general couplings use VMPS::HeisenbergXYZ.
  */
class Heisenberg : public Mpo<Sym::U0,double>, public HeisenbergObservables<Sym::U0>, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(Heisenberg)
	
	static qarray<0> singlet() {return qarray<0>{};};
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	Heisenberg() : Mpo<Symmetry>(), HeisenbergObservables(), ParamReturner(Heisenberg::sweep_defaults) {};
	Heisenberg (const size_t &L, const vector<Param> &params, const BC & boundary=BC::OPEN);
	///@}

	static void add_operators (const std::vector<SpinBase<Symmetry>> &B, const ParamHandler &P, PushType<SiteOperator<Symmetry,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
	
	static refEnergy ref (const vector<Param> &params, double L=numeric_limits<double>::infinity());
};

const std::map<string,std::any> Heisenberg::defaults = 
{
	{"J",1.}, {"Jprime",0.}, {"Jrung",1.},
	{"Bz",0.}, {"Bx",0.},
	{"Kz",0.}, {"Kx",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.}, // Dzialoshinsky-Moriya terms
	{"t",0.}, {"mu",0.}, {"Delta",0.}, // Kitaev chain terms
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const std::map<string,std::any> Heisenberg::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",40ul}, {"min_halfsweeps",1ul},
	{"Dinit",10ul}, {"Qinit",10ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-5}, {"tol_state",1.e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

Heisenberg::
Heisenberg (const size_t &L, const vector<Param> &params, const BC & boundary)
	:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN, boundary),
 HeisenbergObservables(L,params,Heisenberg::defaults),
 ParamReturner(Heisenberg::sweep_defaults)
{
	ParamHandler P(params,Heisenberg::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis(),l);
	}

	if (P.HAS_ANY_OF({"Dy", "Dyperp", "Dyprime"}))
	{
		this->set_name("Dzyaloshinsky-Moriya");
	}
	else if (P.HAS_ANY_OF({"t", "mu", "Delta"}))
	{
		this->set_name("KitaevChain");
	}
	else
	{
		this->set_name("Heisenberg");
	}

	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
	HeisenbergU1::set_operators(B, P, pushlist, labellist, boundary);
	add_operators(B, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));

	this->precalc_TwoSiteData();
}

void Heisenberg::
add_operators(const std::vector<SpinBase<Symmetry>> &B, const ParamHandler &P, PushType<SiteOperator<Symmetry,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = B[loc].orbitals();
		std::size_t next_orbitals  = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[lp2].orbitals();
		
		// Local terms: B, K, DM⟂
		
		param1d Bx = P.fill_array1d<double>("Bx", "Bxorb", orbitals, loc%Lcell);
		param1d Kx = P.fill_array1d<double>("Kx", "Kxorb", orbitals, loc%Lcell);
		param2d Dyperp = P.fill_array2d<double>("Dyrung", "Dy", "Dyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(Bx.label);
		labellist[loc].push_back(Kx.label);
		labellist[loc].push_back(Dyperp.label);
		
		ArrayXd Bz_array = B[loc].ZeroField();
		ArrayXd mu_array = B[loc].ZeroField();
		ArrayXd Kz_array = B[loc].ZeroField();
		ArrayXXd Jperp_array = B[loc].ZeroHopping();

		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction(B[loc].HeisenbergHamiltonian(Jperp_array, Jperp_array, Bz_array, Bx.a, mu_array, Kz_array, Kx.a, Dyperp.a));
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Nearest-neighbour terms: DM=Dzyaloshinsky-Moriya		
		param2d Dypara = P.fill_array2d<double>("Dy", "Dypara", {orbitals, next_orbitals}, loc%Lcell);
		labellist[loc].push_back(Dypara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals; alfa++)
			for (std::size_t beta=0; beta<next_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SX,alfa), B[lp1].Scomp(SZ,beta)), +Dypara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SZ,alfa), B[lp1].Scomp(SX,beta)), -Dypara(alfa,beta)));
			}
		}
		
		// Next-nearest-neighbour terms: DM		
		param2d Dyprime = P.fill_array2d<double>("Dyprime", "Dyprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		labellist[loc].push_back(Dyprime.label);
		
		if (loc < N_sites-2 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals; ++alfa)
			for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SX,alfa), B[lp1].Id(), B[lp2].Scomp(SZ,beta)), +Dyprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SZ,alfa), B[lp1].Id(), B[lp2].Scomp(SX,beta)), -Dyprime(alfa,beta)));
			}
		}
	}
}

refEnergy Heisenberg::
ref (const vector<Param> &params, double L)
{
	ParamHandler P(params,{{"D",2ul},{"Ly",1ul},{"m",0.},{"J",1.},{"Jxy",0.},{"Jz",0.},
	                       {"Jprime",0.},{"Bz",0.},{"Bx",0.},{"Kx",0.},{"Kz",0.},{"Dy",0.},{"Dyprime",0.}});
	refEnergy out;
	
	size_t Ly = P.get<size_t>("Ly");
	size_t D = P.get<size_t>("D");
	double J = P.get<double>("J");
	double Jxy = P.get<double>("Jxy");
	double Jz = P.get<double>("Jz");
	double Jprime = P.get<double>("Jprime");
	
	// Heisenberg chain and ladder
	if (isinf(L) and J > 0. and P.ARE_ALL_ZERO<double>({"m","Jprime","Jxy","Jz","Bz","Bx","Kx","Kz","Dy","Dyprime"}))
	{
		// out.source = "T. Xiang, Thermodynamics of quantum Heisenberg spin chains, Phys. Rev. B 58, 9142 (1998)";
		out.source = "F. B. Ramos and J. C. Xavier, N-leg spin-S Heisenberg ladders, Phys. Rev. B 89, 094424 (2014)";
		out.method = "literature";
		
		if (D == 2)
		{
			if (Ly == 1) {out.value = -log(2)+0.25; out.method = "analytical";}
			if (Ly == 2) {out.value = -0.578043140180; out.method = "IDMRG high precision";}
			if (Ly == 3) {out.value = -0.600537;}
			if (Ly == 4) {out.value = -0.618566;}
			if (Ly == 5) {out.value = -0.62776;}
			if (Ly == 6) {out.value = -0.6346;}
		}
		else if (D == 3)
		{
			if (Ly == 1) {out.value = -1.40148403897;}
			if (Ly == 2) {out.value = -1.878372746;}
			if (Ly == 3) {out.value = -2.0204;}
			if (Ly == 4) {out.value = -2.0957;}
			if (Ly == 5) {out.value = -2.141;}
			if (Ly == 6) {out.value = -2.169;}
		}
		else if (D == 4)
		{
			if (Ly == 1) {out.value = -2.828337;}
			if (Ly == 2) {out.value = -3.930067;}
			if (Ly == 3) {out.value = -4.2718;}
			if (Ly == 4) {out.value = -4.446;}
			if (Ly == 5) {out.value = -4.553;}
			if (Ly == 6) {out.value = -4.60;}
		}
		else if (D == 5)
		{
			if (Ly == 1) {out.value = -4.761248;}
			if (Ly == 2) {out.value = -6.73256;}
			if (Ly == 3) {out.value = -7.3565;}
			if (Ly == 4) {out.value = -7.669;}
			if (Ly == 5) {out.value = -7.865;}
			if (Ly == 6) {out.value = -7.94;}
		}
		else if (D == 6)
		{
			if (Ly == 1) {out.value = -7.1924;}
			if (Ly == 2) {out.value = -10.2852;}
			if (Ly == 3) {out.value = -11.274;}
			if (Ly == 4) {out.value = -11.76;}
			if (Ly == 5) {out.value = -12.08;}
			if (Ly == 6) {out.value = -12.1;}
		}
		
		out.value *= J;
	}
	// XX chain
	else if (isinf(L) and D == 2 and Jxy > 0. and P.ARE_ALL_ZERO<double>({"m","J","Jprime","Jz","Bz","Bx","Kx","Kz","Dy","Dyprime"}))
	{
		out.value = -M_1_PI*Jxy;
		out.source = "S. Paul, A. K. Ghosh, Ground state properties of the bond alternating spin-1/2 anisotropic Heisenberg chain, Condensed Matter Physics, 2017, Vol. 20, No 2, 23701: 1–16";
		out.method = "analytical";
	}
	// Majumdar-Ghosh chain
	else if (D == 2 and J > 0. and Jprime == 0.5*J and P.ARE_ALL_ZERO<double>({"m","Jxy","Jz","Bz","Bx","Kx","Kz","Dy","Dyprime"}))
	{
		out.value = -0.375*J;
		out.source = "https://en.wikipedia.org/wiki/Majumdar-Ghosh_model";
		out.method = "analytical";
	}
	
	return out;
}

} // end namespace VMPS

#endif
