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
 \f[
  H = -J \sum_{<ij>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right) 
      -J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
      -B_z \sum_i S^z_i
      -B_x \sum_i S^x_i
      +K_z \sum_i \left(S^z_i\right)^2
      +K_x \sum_i \left(S^x_i\right)^2
      -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
      -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Uses no symmetry. Any parameter constellations are allowed. For variants with symmetries, see VMPS::HeisenbergU1 or VMPS::HeisenbergSU2.
  \note The default variable settings can be seen in \p Heisenberg::defaults.
  \note \f$J<0\f$ is antiferromagnetic
  \note This is the real version of the Heisenbergmodel without symmetries, so \f$J_x = J_y\f$ is mandatory. For general couplings use VMPS::HeisenbergXYZ.
*/
class Heisenberg : public MpoQ<Sym::U0,double>
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///\{
	Heisenberg() : MpoQ<Symmetry>() {};
	Heisenberg (const size_t &L, const vector<Param> &params);
	///\}
	
	static void add_operators (HamiltonianTermsXd<Symmetry> &Terms, const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc=0);
	
	///@{
	/**Observables.*/
	MpoQ<Symmetry> SzSz (size_t loc1, size_t loc2);
	MpoQ<Symmetry> Sz   (size_t loc);
	///@}
	
	static const std::map<string,std::any> defaults;
	
protected:
	
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> Heisenberg::defaults = 
{
	{"J",-1.}, {"Jprime",0.}, {"Jperp",0.},
	{"D",2ul}, {"Bz",0.}, {"Bx",0.}, {"Kz",0.}, {"Kx",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyperp",0.}, // Dzialoshinsky-Moriya terms
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1}
};


Heisenberg::
Heisenberg (const size_t &L, const vector<Param> &params)
:MpoQ<Symmetry> (L, qarray<0>({}), labeldummy, "")
{
	ParamHandler P(params,Heisenberg::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		cout << "l=" << l << ", N_phys=" << P.get<size_t>("Ly",l%Lcell) << endl;
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = HeisenbergU1::set_operators(B[l],P,l%Lcell);
		add_operators(Terms[l],B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	cout << "N_phys=" << N_phys << endl;
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

MpoQ<Sym::U0> Heisenberg::
Sz (size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<Symmetry > Mout(N_sites, qarray<0>{}, labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B[l].get_basis(),l); }
	Mout.setLocal(loc, B[loc].Scomp(SZ));
	return Mout;
}

MpoQ<Sym::U0> Heisenberg::
SzSz (size_t loc1, size_t loc2)
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "Sz(" << loc1 << ")" <<  "Sz(" << loc2 << ")";
	MpoQ<Symmetry > Mout(N_sites, qarray<0>{}, labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B[l].get_basis(),l); }
	Mout.setLocal({loc1, loc2}, {B[loc1].Scomp(SZ), B[loc2].Scomp(SZ)});
	return Mout;
}

void Heisenberg::
add_operators (HamiltonianTermsXd<Symmetry> &Terms, const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
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
