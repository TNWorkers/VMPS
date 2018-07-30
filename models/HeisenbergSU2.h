#ifndef STRAWBERRY_HEISENBERGSU2
#define STRAWBERRY_HEISENBERGSU2

#include "symmetry/SU2.h"
#include "bases/SpinBaseSU2.h"
#include "Mpo.h"
#include "DmrgExternal.h"
#include "ParamHandler.h" // from TOOLS
#include "ParamReturner.h"

namespace VMPS
{

/** \class HeisenbergSU2
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model
  *
  * MPO representation of 
  * \f[
  * H =  J \sum_{<ij>} \left(\mathbf{S_i}\mathbf{S_j}\right)
        +J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right)
  * \f]
  *
  * \note Makes use of the spin-SU(2) symmetry, which implies no magnetic fields. For B-fields see VMPS::HeisenbergU1.
  * \note The default variable settings can be seen in \p HeisenbergSU2::defaults.
  * \note \f$J>0\f$ is antiferromagnetic
  */
class HeisenbergSU2 : public Mpo<Sym::SU2<Sym::SpinSU2>,double>, public ParamReturner
{
public:
	typedef Sym::SU2<Sym::SpinSU2> Symmetry;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::SparseMatrix<double> SparseMatrixType;
	
public:
	
	//---constructors---
	
	///\{
	/**Do nothing.*/
	HeisenbergSU2() : Mpo<Symmetry>(), ParamReturner(HeisenbergSU2::sweep_defaults) {};
	
	/**
	   \param L : chain length
	   \describe_params
	*/
	HeisenbergSU2 (const size_t &L, const vector<Param> &params={});
	///\}
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc=0);
	
	///@{
	/**Observables.*/
	Mpo<Symmetry,double> S (std::size_t locx, std::size_t locy=0);
	Mpo<Symmetry,double> Sdag (std::size_t locx, std::size_t locy=0, double factor=sqrt(3.));
	Mpo<Symmetry,double> SS (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	///@}
	
	/**Validates whether a given total quantum number \p qnum is a possible target quantum number for an MpsQ.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<1> qnum) const;
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
	
protected:
	
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> HeisenbergSU2::defaults = 
{
	{"J",1.}, {"Jprime",0.}, {"Jrung",1.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::map<string,std::any> HeisenbergSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",6ul},
	{"Dinit",4ul}, {"Qinit",10ul}, {"Dlimit",100ul},
	{"tol_eigval",1.e-7}, {"tol_state",1.e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HeisenbergSU2::
HeisenbergSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1}), "", true),
 ParamReturner(HeisenbergSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis().qloc(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
S (std::size_t locx, std::size_t locy)
{
	assert(locx<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx << "," << locy << ")";
	
	SiteOperator Op = B[locx].S(locy).plain<double>();
	
	Mpo<Symmetry> Mout(N_sites, Op.Q, ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx,Op);
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
Sdag (std::size_t locx, std::size_t locy, double factor)
{
	assert(locx<this->N_sites);
	std::stringstream ss;
	ss << "Sdag(" << locx << "," << locy << ")";
	
	SiteOperator Op = factor * B[locx].Sdag(locy).plain<double>();
	
	Mpo<Symmetry> Mout(N_sites, Op.Q, ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx,Op);
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
SS (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites,Symmetry::qvacuum(),ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	if (locx1 == locx2)
	{
		auto product = std::sqrt(3.)*OperatorType::prod(B[locx1].Sdag(locy1), B[locx2].S(locy2),Symmetry::qvacuum());
		// auto product = Operator::prod(B[locx1].Sdag(locy1), B[locx2].S(locy2), Symmetry::qvacuum());
		Mout.setLocal(locx1, product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.)*B[locx1].Sdag(locy1)).plain<double>(), B[locx2].S(locy2).plain<double>()});
		// Mout.setLocal({locx1, locx2}, {(B[locx1].Sdag(locy1)).plain<double>(), B[locx2].S(locy2).plain<double>()});
		return Mout;
	}
}

bool HeisenbergSU2::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0]-1,2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if(Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

HamiltonianTermsXd<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
set_operators (const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	Terms.name = "Heisenberg";
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	stringstream ss;
	ss << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc)-1,2));
	save_label(ss.str());
	
	// J-terms
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",B.orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(std::sqrt(3)*Jpara(i,j), B.Sdag(i).plain<double>(), 
			                                                          B.S(j).plain<double>()));
		}
	}
	
	// J'-terms
	
	param0d Jprime = P.fill_array0d<double>("Jprime","Jprime",loc);
	save_label(Jprime.label);
	
	assert((B.orbitals() == 1 or Jprime.x == 0) and "Cannot interpret Ly>1 and J'!=0");
	
	if (Jprime.x != 0)
	{
		Terms.nextn.push_back(make_tuple(std::sqrt(3)*Jprime.x, B.Sdag(0).plain<double>(), 
		                                                        B.S(0).plain<double>(), 
		                                                        B.Id().plain<double>()));
	}
	
	// perp terms
	
	auto [Jrung,Jperp,Jperplabel] = P.fill_array2d<double>("Jrung","J","Jperp",B.orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jperplabel);
	
	if (B.orbitals() > 1)
	{
		Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jperp).plain<double>()));
	}
	
	return Terms;
}

} //end namespace VMPS

#endif
