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
	HubbardSU2xSU2BondOperator (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN);

	void set_operators (const std::vector<FermionBase<Symmetry> > &F, const ParamHandler &P,
							   PushType<SiteOperator<Symmetry,Scalar>,Scalar>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
	
private:
	
	vector<FermionBase<Symmetry> > F;
};

template<typename Scalar>
const std::map<string,std::any> HubbardSU2xSU2BondOperator<Scalar>::defaults = 
{
	{"x",0ul}, {"shift",0.}, 
	{"maxPower",1ul}, {"CYLINDER",false}, {"Ly",1ul}, 
};

template<typename Scalar>
HubbardSU2xSU2BondOperator<Scalar>::
HubbardSU2xSU2BondOperator (const size_t &L, const vector<Param> &params, const BC &boundary)
:Mpo<Symmetry,Scalar> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::NON_HAMILTONIAN, boundary)
{
	ParamHandler P(params,HubbardSU2xSU2BondOperator::defaults);	
	size_t Lcell = P.size();
	
	F.resize(this->N_sites);
	
	for (size_t l=0; l<this->N_sites; ++l)
	{
		this->N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell));
		this->setLocBasis(F[l].get_basis().qloc(),l);
	}

	this->set_name("HubbardSU2xSU2BondOperator");

	PushType<SiteOperator<Symmetry,Scalar>,Scalar> pushlist;
    std::vector<std::vector<std::string>> labellist;
    set_operators(F, P, pushlist, labellist, boundary);

	this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<bool>("maxPower"));
}

template<typename Scalar>
void HubbardSU2xSU2BondOperator<Scalar>::set_operators (const std::vector<FermionBase<Symmetry> > &F, const ParamHandler &P,
														PushType<SiteOperator<Symmetry,Scalar>,Scalar>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	param0d x = P.fill_array0d<size_t>("x", "x_", 0);
	param0d shift = P.fill_array0d<double>("shift", "shift_", 0);
	
	stringstream ss;
	ss << "x=" << x() << ", shift=" << shift();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		if (abs(shift()) > ::mynumeric_limits<double>::epsilon())
		{
			auto Hloc = Mpo<Symmetry,Scalar>::get_N_site_interaction(F[x()].Id().template plain<double>().template cast<Scalar>());
			pushlist.push_back(std::make_tuple(l, Hloc, shift()/N_sites));
		}
		labellist[l].push_back(ss.str());
	}
	
//	Terms.push_local(x(), shift(), F[x()].Id().plain<double>().cast<Scalar>());
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Terms.save_label(l,ss.str());
//	}
	
	for (size_t loc=0; loc<N_sites; ++loc)
	{
		auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			SiteOperator<Symmetry,Scalar> cdag_sign_loc = OperatorType::prod(F[x()].cdag(Gloc,0), F[x()].sign(), {2,2}).template plain<double>().template cast<Scalar>();
			SiteOperator<Symmetry,Scalar> c_tight       = F[x()+1].c(Gloc,0).template plain<double>().template cast<Scalar>();
			
			double coupling = (loc==x())? 2.:1e-15;
			
			pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,Scalar>::get_N_site_interaction(cdag_sign_loc, c_tight), coupling));
		}
	}
}

} //end namespace VMPS

#endif
