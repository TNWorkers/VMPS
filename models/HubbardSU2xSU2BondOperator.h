#ifndef STRAWBERRY_HUBBARDSO4BONDOPERATOR
#define STRAWBERRY_HUBBARDSO4BONDOPERATOR

#include "models/HubbardSU2xSU2.h"

namespace VMPS
{
template<typename Scalar>
class HubbardSU2xSU2BondOperator : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,Scalar>
{
public:
	
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HubbardSU2xSU2BondOperator() : Mpo<Symmetry,Scalar>() {};
	HubbardSU2xSU2BondOperator (const size_t &L, const vector<Param> &params);
	
	void set_operators (const std::vector<FermionBase<Symmetry> > &F, const ParamHandler &P, HamiltonianTerms<Symmetry,Scalar> &Terms);
	
	static const std::map<string,std::any> defaults;
	
private:
	
	vector<FermionBase<Symmetry> > F;
};

template<typename Scalar>
const std::map<string,std::any> HubbardSU2xSU2BondOperator<Scalar>::defaults = 
{
	{"x",0ul}, {"shift",0.}, 
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
};

template<typename Scalar>
HubbardSU2xSU2BondOperator<Scalar>::
HubbardSU2xSU2BondOperator (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry,Scalar> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::NON_HAMILTONIAN)
{
	ParamHandler P(params,HubbardSU2xSU2BondOperator::defaults);
	
	size_t Lcell = P.size();
	HamiltonianTerms<Symmetry,Scalar> Terms(this->N_sites, P.get<bool>("OPEN_BC"));
	F.resize(this->N_sites);
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		this->N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = (l%2 == 0) ? FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::A):
		                    FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::B);
		this->setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	set_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

template<typename Scalar>
void HubbardSU2xSU2BondOperator<Scalar>::
set_operators (const std::vector<FermionBase<Symmetry> > &F, const ParamHandler &P, HamiltonianTerms<Symmetry,Scalar> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	Terms.set_name("HubbardSU2xSU2BondOperator");
	
	param0d x = P.fill_array0d<size_t>("x", "x_", 0);
	param0d shift = P.fill_array0d<double>("shift", "shift_", 0);
	
	stringstream ss;
	ss << "x=" << x() << ", shift=" << shift();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		if (abs(shift()) > ::mynumeric_limits<double>::epsilon())
		{
			Terms.push_local(l, shift()/N_sites, F[x()].Id().plain<double>().cast<Scalar>());
		}
		Terms.save_label(l,ss.str());
	}
	
//	Terms.push_local(x(), shift(), F[x()].Id().plain<double>().cast<Scalar>());
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Terms.save_label(l,ss.str());
//	}
	
	for (size_t loc=0; loc<N_sites; ++loc)
	{
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			SiteOperator<Symmetry,Scalar> cdag_sign_loc = OperatorType::prod(F[x()].cdag(0), F[x()].sign(), {2,2}).plain<double>().cast<Scalar>();
			SiteOperator<Symmetry,Scalar> c_tight       = F[x()+1].c(0).plain<double>().cast<Scalar>();
			
			double coupling = (loc==x())? 2.:1e-15;
			
			Terms.push_tight(loc, coupling, cdag_sign_loc, c_tight);
		}
	}
}

} //end namespace VMPS

#endif
