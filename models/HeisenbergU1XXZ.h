#ifndef STRAWBERRY_HEISENBERGU1XXZ
#define STRAWBERRY_HEISENBERGU1XXZ

#include "models/HeisenbergU1.h"

namespace VMPS
{

class HeisenbergU1XXZ : public HeisenbergU1
{
public:
	typedef Sym::U1<double> Symmetry;
	
public:
	
	HeisenbergU1XXZ() : HeisenbergU1() {};
	HeisenbergU1XXZ (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params);
	
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
	{"D",2ul}, {"Bz",0.}, {"K",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true},
	
	// for consistency during inheritance:
	{"J",0.}, {"Jprime",0.}, {"Jperp",0.}, {"Jpara",0.}
};

HeisenbergU1XXZ::
HeisenbergU1XXZ (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params)
:HeisenbergU1(L)
{
	ParamHandler P(params,HeisenbergU1XXZ::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		B[l] = SpinBase<Symmetry>(N_legs, P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
		add_operators(Terms[l],B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
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
	
	// J-terms
	
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
			Terms.tight.push_back(make_tuple(-Jzpara(i,j),     B.Scomp(SZ,i), B.Scomp(SZ,j)));
		}
	}
	
	double Jxyperp = P.get_default<double>("Jxyperp");
	
	if (P.HAS("Jxy",loc))
	{
		Jxyperp = P.get<double>("Jxy",loc);
	}
	else if (P.HAS("Jxyperp",loc))
	{
		Jxyperp = P.get<double>("Jxyperp",loc);
		stringstream ss; ss << "Jxy⟂=" << Jxyperp; Terms.info.push_back(ss.str());
	}
	
	double Jzperp = P.get_default<double>("Jzperp");
	
	if (P.HAS("Jz",loc))
	{
		Jzperp = P.get<double>("Jz",loc);
	}
	else if (P.HAS("Jzperp",loc))
	{
		Jzperp = P.get<double>("Jzperp",loc);
		stringstream ss; ss << "Jz⟂=" << Jzperp; Terms.info.push_back(ss.str());
	}
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jxyperp,Jzperp,0.,0.,0.,0., P.get<bool>("CYLINDER"))));
	
	Terms.name = (P.HAS_ANY_OF({"Jxy","Jxypara","Jxyperp"},loc))? "XXZ":"Ising";
	
	return Terms;
}

} //end namespace VMPS

#endif
