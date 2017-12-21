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
  H = -J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j 
      -J'_{xy} \sum_{<<ij>>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J'_z \sum_{<<ij>>} S^z_iS^z_j 
      -B_z \sum_i S^z_i
      +K_z \sum_i \left(S^z_i\right)^2
      -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
      -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Take use of the \f$S^z\f$ U(1) symmetry.
  \note The default variable settings can be seen in \p HeisenbergU1XXZ::defaults.
  \note \f$J<0\f$ is antiferromagnetic.
*/
class HeisenbergU1XXZ : public HeisenbergU1
{
public:
	typedef Sym::U1<double> Symmetry;
	
public:
	
	HeisenbergU1XXZ() : HeisenbergU1() {};
	HeisenbergU1XXZ (const size_t &L, const vector<Param> &params);
	
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> add_operators (HamiltonianTermsXd<Symmetry_> &Terms, 
	                                                    const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergU1XXZ::defaults = 
{
	{"Jxy",-1.}, {"Jz",0.},
	{"Jxyprime",0.}, {"Jzprime",0.},
	{"Jxyperp",0.}, {"Jzperp",0.},
	
	{"Dy",0.}, {"Dyperp",0.}, {"Dyprime",0.},
	{"D",2ul}, {"Bz",0.}, {"Kz",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1}, 
	
	// for consistency during inheritance (should not be set for XXZ!):
	{"J",0.}, {"Jprime",0.}, {"Jperp",0.}, {"Jpara",0.}
};

HeisenbergU1XXZ::
HeisenbergU1XXZ (const size_t &L, const vector<Param> &params)
:HeisenbergU1(L)
{
	ParamHandler P(params,HeisenbergU1XXZ::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
		add_operators(Terms[l],B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergU1XXZ::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// Jxy/Jz terms
	
	auto [Jxy,Jxypara,Jxylabel] = P.fill_array2d<double>("Jxy","Jxypara",B.orbitals(),loc);
	save_label(Jxylabel);
	
	auto [Jz,Jzpara,Jzlabel] = P.fill_array2d<double>("Jz","Jzpara",B.orbitals(),loc);
	save_label(Jzlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jxypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-0.5*Jxypara(i,j), B.Scomp(SP,i), B.Scomp(SM,j)));
			Terms.tight.push_back(make_tuple(-0.5*Jxypara(i,j), B.Scomp(SM,i), B.Scomp(SP,j)));
		}
		
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-Jzpara(i,j), B.Scomp(SZ,i), B.Scomp(SZ,j)));
		}
	}
	
	// Jxy'/Jz' terms
	
	param0d Jxyprime = P.fill_array0d<double>("Jxyprime","Jxyprime",loc);
	save_label(Jxyprime.label);
	
	if (Jxyprime.x != 0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with Jxy' terms!");
		
		Terms.nextn.push_back(make_tuple(-0.5*Jxyprime.x, B.Scomp(SP), B.Scomp(SM), B.Id()));
		Terms.nextn.push_back(make_tuple(-0.5*Jxyprime.x, B.Scomp(SM), B.Scomp(SP), B.Id()));
	}
	
	param0d Jzprime = P.fill_array0d<double>("Jzprime","Jzprime",loc);
	save_label(Jzprime.label);
	
	if (Jzprime.x != 0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with Jz' terms!");
		
		Terms.nextn.push_back(make_tuple(-Jzprime.x, B.Scomp(SZ), B.Scomp(SZ), B.Id()));
	}
	
	// local terms
	
	param0d Jxyperp = P.fill_array0d<double>("Jxy","Jxyperp",loc);
	save_label(Jxyperp.label);
	
	param0d Jzperp = P.fill_array0d<double>("Jz","Jzperp",loc);
	save_label(Jzperp.label);
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jxyperp.x,Jzperp.x,0.,0.,0.,0.,0., P.get<bool>("CYLINDER"))));
	
	Terms.name = (P.HAS_ANY_OF({"Jxy","Jxypara","Jxyperp"},loc))? "XXZ":"Ising";
	
	return Terms;
}

} //end namespace VMPS

#endif
