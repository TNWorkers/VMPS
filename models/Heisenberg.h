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
  * H = -J \sum_{<ij>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right) 
  *     -J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
  *     -B_z \sum_i S^z_i
  *     -B_x \sum_i S^x_i
  *     +K_z \sum_i \left(S^z_i\right)^2
  *     +K_x \sum_i \left(S^x_i\right)^2
  *     -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  *     -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  * \f]
  *
  * \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  * \note Uses no symmetry. Any parameter constellations are allowed. For variants with symmetries, see VMPS::HeisenbergU1 or VMPS::HeisenbergSU2.
  * \note The default variable settings can be seen in \p Heisenberg::defaults.
  * \note \f$J<0\f$ is antiferromagnetic
  * \note This is the real version of the Heisenbergmodel without symmetries, so \f$J_x = J_y\f$ is mandatory. For general couplings use VMPS::HeisenbergXYZ.
  */
class Heisenberg : public Mpo<Sym::U0,double>, public HeisenbergObservables<Sym::U0>
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	Heisenberg() : Mpo<Symmetry>(), HeisenbergObservables() {};
	Heisenberg (const size_t &L, const vector<Param> &params);
	///@}

	///@{
	/**Push params for DMRG algorithms via these functions to an instance of DmrgSolver.*/
	DMRG::CONTROL::DYN get_DynParam(const vector<Param> &params={}) const;
	DMRG::CONTROL::GLOB get_GlobParam(const vector<Param> &params={}) const;
	///@}

	static void add_operators (HamiltonianTermsXd<Symmetry> &Terms, const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> Heisenberg::defaults = 
{
	{"J",-1.}, {"Jprime",0.}, {"Jperp",0.},
	{"D",2ul}, {"Bz",0.}, {"Bx",0.}, {"Kz",0.}, {"Kx",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyperp",0.}, // Dzialoshinsky-Moriya terms
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::map<string,std::any> Heisenberg::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",40ul}, {"max_halfsweeps",6ul},
	{"Dinit",10ul}, {"Qinit",10ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-5}, {"tol_state",1.e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

Heisenberg::
Heisenberg (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<0>({}), "", true),
 HeisenbergObservables(L,params,Heisenberg::defaults)
{
	ParamHandler P(params,Heisenberg::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = HeisenbergU1::set_operators(B[l],P,l%Lcell);
		add_operators(Terms[l],B[l],P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

DMRG::CONTROL::GLOB Heisenberg::
get_GlobParam(const vector<Param> &params) const
{
	ParamHandler P(params,Heisenberg::sweep_defaults);
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

DMRG::CONTROL::DYN Heisenberg::
get_DynParam(const vector<Param> &params) const
{
	ParamHandler P(params,Heisenberg::sweep_defaults);
	DMRG::CONTROL::DYN out;
	double tmp1        = P.get<double>("max_alpha");
	out.max_alpha_rsvd = [tmp1] (size_t i) { return tmp1; };
	tmp1               = P.get<double>("min_alpha");
	out.min_alpha_rsvd = [tmp1] (size_t i) { return tmp1; };
	tmp1               = P.get<double>("eps_svd");
	out.eps_svd        = [tmp1] (size_t i) { return tmp1; };
	size_t tmp2        = P.get<size_t>("Dincr_abs");
	out.Dincr_abs      = [tmp2] (size_t i) { return tmp2; };
	tmp2               = P.get<size_t>("Dincr_per");
	out.Dincr_per      = [tmp2] (size_t i) { return tmp2; };
	tmp1               = P.get<double>("Dincr_rel");
	out.Dincr_rel      = [tmp1] (size_t i) { return tmp1; };
	tmp2               = P.get<size_t>("min_Nsv");
	out.min_Nsv        = [tmp2] (size_t i) { return tmp2; };
	int tmp3           = P.get<int>("max_Nrich");
	out.max_Nrich	   = [tmp3] (size_t i) { return tmp3; };
	return out;
}

void Heisenberg::
add_operators (HamiltonianTermsXd<Symmetry> &Terms, const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	stringstream ss;
	ss << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc)-1,2));
	save_label(ss.str());
	
	// Dzyaloshinsky-Moriya terms
	
	auto [Dy,Dypara,Dylabel] = P.fill_array2d<double>("Dy","Dypara",B.orbitals(),loc);
	save_label(Dylabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Dypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(+Dypara(i,j), B.Scomp(SX), B.Scomp(SZ)));
			Terms.tight.push_back(make_tuple(-Dypara(i,j), B.Scomp(SZ), B.Scomp(SX)));
		}
	}
	
	param0d Dyprime = P.fill_array0d<double>("Dyprime","Dyprime",loc);
	save_label(Dyprime.label);
	
	if (Dyprime.x != 0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with Dy' terms!");
		
		Terms.nextn.push_back(make_tuple(+Dyprime.x, B.Scomp(SX), B.Scomp(SZ), B.Id()));
		Terms.nextn.push_back(make_tuple(-Dyprime.x, B.Scomp(SZ), B.Scomp(SX), B.Id()));
	}
	
	// local terms
	
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",B.orbitals(),loc);
	save_label(Bxlabel);
	
	auto [Kx,Kxorb,Kxlabel] = P.fill_array1d<double>("Kx","Kxorb",B.orbitals(),loc);
	save_label(Kxlabel);
	
	param0d Dyperp = P.fill_array0d<double>("Dy","Dyperp",loc);
	save_label(Dyperp.label);
	
	Terms.name = (P.HAS_ANY_OF({"Dy","Dyperp","Dyprime"},loc))? "Dzyaloshinsky-Moriya":"Heisenberg";
	
	ArrayXd Bzorb = B.ZeroField();
	ArrayXd Kzorb = B.ZeroField();
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(0.,0.,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp.x, P.get<bool>("CYLINDER"))));
}

} // end namespace VMPS

#endif
