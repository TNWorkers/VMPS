#ifndef VANILLA_HEISENBERG
#define VANILLA_HEISENBERG

#include "models/HeisenbergU1.h"

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
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	Heisenberg() : Mpo<Symmetry>(), HeisenbergObservables(), ParamReturner(Heisenberg::sweep_defaults) {};
	Heisenberg (const size_t &L, const vector<Param> &params);
	///@}
	
	static void add_operators (HamiltonianTermsXd<Symmetry> &Terms, const vector<SpinBase<Symmetry> > &B, const ParamHandler &P, size_t loc=0);
	
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
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::map<string,std::any> Heisenberg::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",40ul}, {"min_halfsweeps",6ul},
	{"Dinit",10ul}, {"Qinit",10ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-5}, {"tol_state",1.e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

Heisenberg::
Heisenberg (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<0>({}), "", true),
 HeisenbergObservables(L,params,Heisenberg::defaults),
 ParamReturner(Heisenberg::sweep_defaults)
{
	ParamHandler P(params,Heisenberg::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = HeisenbergU1::set_operators(B,P,l%Lcell);
		add_operators(Terms[l],B,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

void Heisenberg::
add_operators (HamiltonianTermsXd<Symmetry> &Terms, const vector<SpinBase<Symmetry> > &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	size_t lp1 = (loc+1)%B.size();
	
	// Dzyaloshinsky-Moriya terms
	
	auto [Dy,Dypara,Dylabel] = P.fill_array2d<double>("Dy","Dypara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Dylabel);
	
	for (int i=0; i<B[loc].orbitals(); ++i)
	for (int j=0; j<B[lp1].orbitals(); ++j)
	{
		if (Dypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(+Dypara(i,j), B[loc].Scomp(SX), B[loc].Scomp(SZ)));
			Terms.tight.push_back(make_tuple(-Dypara(i,j), B[loc].Scomp(SZ), B[loc].Scomp(SX)));
		}
	}
	
	param0d Dyprime = P.fill_array0d<double>("Dyprime","Dyprime",loc);
	save_label(Dyprime.label);
	
	if (Dyprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Dy' terms!");
		
		Terms.nextn.push_back(make_tuple(+Dyprime.x, B[loc].Scomp(SX), B[loc].Scomp(SZ), B[loc].Id()));
		Terms.nextn.push_back(make_tuple(-Dyprime.x, B[loc].Scomp(SZ), B[loc].Scomp(SX), B[loc].Id()));
	}
	
	// local terms
	
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",B[loc].orbitals(),loc);
	save_label(Bxlabel);
	
	auto [Kx,Kxorb,Kxlabel] = P.fill_array1d<double>("Kx","Kxorb",B[loc].orbitals(),loc);
	save_label(Kxlabel);
	
	auto [Dyrung,Dyperp,Dyperplabel] = P.fill_array2d<double>("Dyrung","Dy","Dyperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Dyperplabel);
	
	Terms.name = (P.HAS_ANY_OF({"Dy","Dyperp","Dyprime"},loc))? "Dzyaloshinsky-Moriya":"Heisenberg";
	
	ArrayXd Bzorb = B[loc].ZeroField();
	ArrayXd Kzorb = B[loc].ZeroField();
	ArrayXXd Jperp = B[loc].ZeroHopping();
	
	Terms.local.push_back(make_tuple(1., B[loc].HeisenbergHamiltonian(Jperp,Jperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp)));
}

refEnergy Heisenberg::
ref (const vector<Param> &params, double L)
{
	ParamHandler P(params,{{"D",2ul},{"Ly",1ul},{"m",0.}});
	refEnergy out;
	
	if (isinf(L) and P.get<double>("m") == 0. and P.HAS_NONE_OF({"Bz","Bx","Kx","Kz","Dy","Dyprime"}))
	{
		out.source = "Tao Xiang, Thermodynamics of quantum Heisenberg spin chains, Phys. Rev. B 58, 9142 (1998)";
		out.method = "literature";
		
		if (P.get<size_t>("D") == 2)
		{
			if (P.get<size_t>("Ly") == 1) {out.value = 0.25-log(2); out.method = "analytical";}
			if (P.get<size_t>("Ly") == 2) {out.value = -0.578043140180;}
			if (P.get<size_t>("Ly") == 3) {out.value = -0.600537;}
			if (P.get<size_t>("Ly") == 4) {out.value = -0.618566;}
			if (P.get<size_t>("Ly") == 5) {out.value = -0.62776;}
			if (P.get<size_t>("Ly") == 6) {out.value = -0.6346;}
		}
		else if (P.get<size_t>("D") == 3)
		{
			if (P.get<size_t>("Ly") == 1) {out.value = -1.40148403897;}
			if (P.get<size_t>("Ly") == 2) {out.value = -1.878372746;}
			if (P.get<size_t>("Ly") == 3) {out.value = -2.0204;}
			if (P.get<size_t>("Ly") == 4) {out.value = -2.0957;}
			if (P.get<size_t>("Ly") == 5) {out.value = -2.141;}
			if (P.get<size_t>("Ly") == 6) {out.value = -2.169;}
		}
		else if (P.get<size_t>("D") == 4)
		{
			if (P.get<size_t>("Ly") == 1) {out.value = -2.828337;}
			if (P.get<size_t>("Ly") == 2) {out.value = -3.930067;}
			if (P.get<size_t>("Ly") == 3) {out.value = -4.2718;}
			if (P.get<size_t>("Ly") == 4) {out.value = -4.446;}
			if (P.get<size_t>("Ly") == 5) {out.value = -4.553;}
			if (P.get<size_t>("Ly") == 6) {out.value = -4.60;}
		}
		else if (P.get<size_t>("D") == 5)
		{
			if (P.get<size_t>("Ly") == 1) {out.value = -4.761248;}
			if (P.get<size_t>("Ly") == 2) {out.value = -6.73256;}
			if (P.get<size_t>("Ly") == 3) {out.value = -7.3565;}
			if (P.get<size_t>("Ly") == 4) {out.value = -7.669;}
			if (P.get<size_t>("Ly") == 5) {out.value = -7.865;}
			if (P.get<size_t>("Ly") == 6) {out.value = -7.94;}
		}
		else if (P.get<size_t>("D") == 6)
		{
			if (P.get<size_t>("Ly") == 1) {out.value = -7.1924;}
			if (P.get<size_t>("Ly") == 2) {out.value = -10.2852;}
			if (P.get<size_t>("Ly") == 3) {out.value = -11.274;}
			if (P.get<size_t>("Ly") == 4) {out.value = -11.76;}
			if (P.get<size_t>("Ly") == 5) {out.value = -12.08;}
			if (P.get<size_t>("Ly") == 6) {out.value = -12.1;}
		}
	}
	
	return out;
}

} // end namespace VMPS

#endif
