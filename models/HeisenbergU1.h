#ifndef STRAWBERRY_HEISENBERGU1
#define STRAWBERRY_HEISENBERGU1

#include <array>

#include "Mpo.h"
#include "symmetry/U1.h"
#include "bases/SpinBase.h"
#include "DmrgExternal.h"
#include "ParamHandler.h" // from HELPERS
#include "models/HeisenbergObservables.h"
#include "symmetry/kind_dummies.h"

namespace VMPS
{

/** \class HeisenbergU1
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model
  *
  * MPO representation of
  * \f[
  * H = -J \sum_{<ij>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right) 
  *     -J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
  *     -B_z \sum_i S^z_i
  *     +K_z \sum_i \left(S^z_i\right)^2
  *     -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  *     -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  * \f]
  *
  * \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  * \note Take use of the \f$S^z\f$ U(1) symmetry.
  * \note The default variable settings can be seen in \p HeisenbergU1::defaults.
  * \note \f$J<0\f$ is antiferromagnetic
  * \note Homogeneous \f$J\f$ is required here. For a XXZ couplings, use VMPS::HeisenbergU1XXZ.
  */
class HeisenbergU1 : public Mpo<Sym::U1<Sym::SpinU1>,double>, public HeisenbergObservables<Sym::U1<Sym::SpinU1> >
{
public:
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	
private:
	typedef Symmetry::qType qType;
	typedef SiteOperator<Symmetry,SparseMatrix<double> > OperatorType;
	
public:
	
	///@{
	HeisenbergU1() : Mpo<Symmetry>() {};
	HeisenbergU1 (const size_t &L);
	HeisenbergU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	 */
	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc=0);
		
	/**Validates whether a given total quantum number \p qnum is a possible target quantum number for an MpsQ.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<1> qnum) const;
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergU1::defaults = 
{
	{"J",-1.}, {"Jprime",0.}, {"Jperp",0.},
	{"Bz",0.}, {"Kz",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1}
};

HeisenbergU1::
HeisenbergU1 (const size_t &L)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", halve<1>),
 HeisenbergObservables(L)
{}

HeisenbergU1::
HeisenbergU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", halve<1>),
 HeisenbergObservables(L,params,HeisenbergU1::defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

bool HeisenbergU1::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0],2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if (Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HeisenbergU1::
set_operators (const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// J terms
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",B.orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-0.5*Jpara(i,j), B.Scomp(SP,i), B.Scomp(SM,j)));
			Terms.tight.push_back(make_tuple(-0.5*Jpara(i,j), B.Scomp(SM,i), B.Scomp(SP,j)));
			Terms.tight.push_back(make_tuple(-Jpara(i,j),     B.Scomp(SZ,i), B.Scomp(SZ,j)));
		}
	}
	
	// J' terms
	
	param0d Jprime = P.fill_array0d<double>("Jprime","Jprime",loc);
	save_label(Jprime.label);
	
	if (Jprime.x != 0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with J' terms!");
		
		Terms.nextn.push_back(make_tuple(-0.5*Jprime.x, B.Scomp(SP), B.Scomp(SM), B.Id()));
		Terms.nextn.push_back(make_tuple(-0.5*Jprime.x, B.Scomp(SM), B.Scomp(SP), B.Id()));
		Terms.nextn.push_back(make_tuple(-Jprime.x,     B.Scomp(SZ), B.Scomp(SZ), B.Id()));
	}
	
	// local terms
	
	param0d Jperp = P.fill_array0d<double>("J","Jperp",loc);
	save_label(Jperp.label);
	
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",B.orbitals(),loc);
	save_label(Bzlabel);
	
	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",B.orbitals(),loc);
	save_label(Kzlabel);
	
	Terms.name = "Heisenberg";
	
	ArrayXd Bxorb = B.ZeroField();
	ArrayXd Kxorb = B.ZeroField();
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jperp.x,Jperp.x,Bzorb,Bxorb,Kzorb,Kxorb,0., P.get<bool>("CYLINDER"))));
	
	return Terms;
}

} //end namespace VMPS

#endif
