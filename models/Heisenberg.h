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
	Heisenberg(size_t Lx_input, initializer_list<Param> params, size_t Ly_input=1, bool CALC_SQUARE=true);
	///\}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<Sym::U0,double>                           StateXd;
	typedef MpsQ<Sym::U0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Sym::U0,Heisenberg>                Solver;
	typedef MpsQCompressor<Sym::U0,double,double>          CompressorXd;
	typedef MpsQCompressor<Sym::U0,complex<double>,double> CompressorXcd;
	typedef MpoQ<Sym::U0>                                  Operator;
	///@}
	
	///@{
	/**Observables.*/	
	MpoQ<Sym::U0> SzSz (size_t loc1, size_t loc2);
	MpoQ<Sym::U0> Sz   (size_t loc);
	///@}
	
private:
	
	double Jxy, Jz;
	double Bz, Bx;
	size_t D;
	std::map<string,std::any> defaults = 
	{
		{"J",0.}, {"Jxy",0.}, {"Jz",0.},
		{"Jprime",0.}, {"Jxyprime",0.}, {"Jzprime",0.},
		{"Jperp",0.}, {"Jxyperp",0.}, {"Jzperp",0.},
		{"Jpara",0.}, {"Jxypara",0.}, {"Jzpara",0.},
		{"D",2ul}, {"Bz",0.}, {"Bx",0.}, {"K",0.}
	};

	SpinBase<Symmetry> B;
};

Heisenberg::
Heisenberg (size_t Lx_input, initializer_list<Param> params, size_t Ly_input, bool CALC_SQUARE)
	:MpoQ<Symmetry> (Lx_input, Ly_input, qarray<0>({}), vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "")
{
	ParamHandler P(params,defaults);
	B = SpinBase<Symmetry>(N_legs, P.get<size_t>("D"));
	
	for (size_t l=0; l<N_sites; ++l) { setLocBasis(B.basis(),l); }
	
	HamiltonianTermsXd<Symmetry> Terms = HeisenbergU1::set_operators(B,P);
	this->label = Terms.info;
	SuperMatrix<Symmetry,double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G, this->W, this->Gvec, CALC_SQUARE);	
}

MpoQ<Sym::U0> Heisenberg::
Sz (size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<Sym::U0 > Mout(N_sites, N_legs, qarray<0>{}, vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B.basis(),l); }
	Mout.setLocal(loc, B.Scomp(SZ));
	return Mout;
}

MpoQ<Sym::U0> Heisenberg::
SzSz (size_t loc1, size_t loc2)
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "Sz(" << loc1 << ")" <<  "Sz(" << loc2 << ")";
	MpoQ<Sym::U0 > Mout(N_sites, N_legs, qarray<0>{}, vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B.basis(),l); }
	Mout.setLocal({loc1, loc2}, {B.Scomp(SZ), B.Scomp(SZ)});
	return Mout;
}
	
} // end namespace VMPS

#endif
