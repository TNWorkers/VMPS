#ifndef VANILLA_HEISENBERG
#define VANILLA_HEISENBERG

#include "models/HeisenbergU1.h"

namespace VMPS
{

/** \class Heisenberg
  * \ingroup Models
  *
  * \brief Heisenberg Model
  *
  * MPO representation of
  \f[
  H = -J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right)
      - B_z \sum_i S^z_i - B_x \sum_i S^x_i
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Uses no symmetry. Any parameter constellations are allowed. For variants with symmetries, see VMPS::HeisenbergU1 or VMPS::HeisenbergSU2.
  \note The default variable settings can be seen in \p Heisenberg::defaults.
  \note \f$J<0\f$ is antiferromagnetic
  \todo In principal one could allow here a general \f$xyz\f$-model. Until now this is not possible, because the Generator of VMPS::HeisenbergU1 is used.
        Maybe change the order, that VMPS::HeisenbergU1 uses the Generator of VMPS::Heisenberg.
*/
class Heisenberg : public MpoQ<Sym::U0,double>
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	//---constructors---
	///\{
	/**Do nothing.*/
	Heisenberg() : MpoQ<Symmetry>() {};
	
	/**
	   \param Lx_input : chain length
	   \describe_params
	   \param Ly_input : amount of legs in ladder
	   \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	Heisenberg (variant<size_t,std::array<size_t,2> > L, vector<Param> params);
	///\}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,Heisenberg>                Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
	///@}
	
	///@{
	/**Observables.*/
	MpoQ<Symmetry> SzSz (size_t loc1, size_t loc2);
	MpoQ<Symmetry> Sz   (size_t loc);
	///@}
	
protected:
	
	SpinBase<Symmetry> B;
};

Heisenberg::
Heisenberg (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0], 
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1], 
                 qarray<0>({}), vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "")
{
	ParamHandler P(params,HeisenbergU1::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		B = SpinBase<Symmetry>(N_legs, P.get<size_t>("D",l%Lcell));
		setLocBasis(B.get_basis(),l);

		Terms[l] = HeisenbergU1::set_operators(B,P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
	}

	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

MpoQ<Sym::U0> Heisenberg::
Sz (size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<Symmetry > Mout(N_sites, N_legs, qarray<0>{}, vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B.get_basis(),l); }
	Mout.setLocal(loc, B.Scomp(SZ));
	return Mout;
}

MpoQ<Sym::U0> Heisenberg::
SzSz (size_t loc1, size_t loc2)
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "Sz(" << loc1 << ")" <<  "Sz(" << loc2 << ")";
	MpoQ<Symmetry > Mout(N_sites, N_legs, qarray<0>{}, vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B.get_basis(),l); }
	Mout.setLocal({loc1, loc2}, {B.Scomp(SZ), B.Scomp(SZ)});
	return Mout;
}
	
} // end namespace VMPS

#endif
