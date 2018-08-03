#ifndef STRAWBERRY_HEISENBERGU1XXZ
#define STRAWBERRY_HEISENBERGU1XXZ

#include "models/HeisenbergU1.h"

namespace VMPS
{
	
/** \class HeisenbergU1XXZ
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model with XXZ-coupling
  *
  * MPO representation of
  \f[
  H =  J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) + J_z \sum_{<ij>} S^z_iS^z_j 
      +J'_{xy} \sum_{<<ij>>} \left(S^x_iS^x_j+S^y_iS^y_j\right) + J'_z \sum_{<<ij>>} S^z_iS^z_j 
      -B_z \sum_i S^z_i
      +K_z \sum_i \left(S^z_i\right)^2
      -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
      -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Makes use of the \f$S^z\f$ U(1) symmetry.
  \note The default variable settings can be seen in \p HeisenbergU1XXZ::defaults.
  \note \f$J>0\f$ is antiferromagnetic.
*/
class HeisenbergU1XXZ : public HeisenbergU1
{
public:
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1XXZ)
	
public:
	
	HeisenbergU1XXZ() : HeisenbergU1() {};
	HeisenbergU1XXZ (const size_t &L, const vector<Param> &params);
	
	template<typename Symmetry_>
	static void add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergU1XXZ::defaults = 
{
	{"Jxy",1.}, {"Jxyprime",0.}, {"Jxyrung",1.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"Bz",0.}, {"Kz",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
	
	// for consistency during inheritance (should not be set for XXZ!):
	{"J",0.}, {"Jprime",0.}
};

HeisenbergU1XXZ::
HeisenbergU1XXZ (const size_t &L, const vector<Param> &params)
:HeisenbergU1(L)
{
	ParamHandler P(params,HeisenbergU1XXZ::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(B,P,l%Lcell);
		add_operators(Terms[l],B,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HeisenbergU1XXZ::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	size_t lp1 = (loc+1)%B.size();
	
	// Jxy/Jz terms
	
	auto [Jxy,Jxypara,Jxylabel] = P.fill_array2d<double>("Jxy","Jxypara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jxylabel);
	
	auto [Jz,Jzpara,Jzlabel] = P.fill_array2d<double>("Jz","Jzpara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jzlabel);
	
	for (int i=0; i<B[loc].orbitals(); ++i)
	for (int j=0; j<B[lp1].orbitals(); ++j)
	{
		if (Jxypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(0.5*Jxypara(i,j), B[loc].Scomp(SP,i), B[loc].Scomp(SM,i)));
			Terms.tight.push_back(make_tuple(0.5*Jxypara(i,j), B[loc].Scomp(SM,i), B[loc].Scomp(SP,i)));
		}
		
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jzpara(i,j), B[loc].Scomp(SZ,i), B[loc].Scomp(SZ,i)));
		}
	}
	
	// Jxy'/Jz' terms
	
	param0d Jxyprime = P.fill_array0d<double>("Jxyprime","Jxyprime",loc);
	save_label(Jxyprime.label);
	
	if (Jxyprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jxy' terms!");
		
		Terms.nextn.push_back(make_tuple(0.5*Jxyprime.x, B[loc].Scomp(SP), B[loc].Scomp(SM), B[loc].Id()));
		Terms.nextn.push_back(make_tuple(0.5*Jxyprime.x, B[loc].Scomp(SM), B[loc].Scomp(SP), B[loc].Id()));
	}
	
	param0d Jzprime = P.fill_array0d<double>("Jzprime","Jzprime",loc);
	save_label(Jzprime.label);
	
	if (Jzprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jz' terms!");
		
		Terms.nextn.push_back(make_tuple(Jzprime.x, B[loc].Scomp(SZ), B[loc].Scomp(SZ), B[loc].Id()));
	}
	
	// local terms
	
//	param0d Jxyperp = P.fill_array0d<double>("Jxy","Jxyperp",loc);
//	save_label(Jxyperp.label);
//	
//	param0d Jzperp = P.fill_array0d<double>("Jz","Jzperp",loc);
//	save_label(Jzperp.label);
	
	auto [Jxy_,Jxyperp,Jxyperplabel] = P.fill_array2d<double>("Jxyrung","Jxy","Jxyperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jxyperplabel);
	
	auto [Jz_,Jzperp,Jzperplabel] = P.fill_array2d<double>("Jzrung","Jz","Jzperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jzperplabel);
	
	ArrayXd Bzorb   = B[loc].ZeroField();
	ArrayXd Bxorb   = B[loc].ZeroField();
	ArrayXd Kzorb   = B[loc].ZeroField();
	ArrayXd Kxorb   = B[loc].ZeroField();
	ArrayXXd Dyperp = B[loc].ZeroHopping();
	
	Terms.local.push_back(make_tuple(1., B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp)));
	
	Terms.name = (P.HAS_ANY_OF({"Jxy","Jxypara","Jxyperp"},loc))? "XXZ":"Ising";
}

} //end namespace VMPS

#endif
