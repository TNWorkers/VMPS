#ifndef STRAWBERRY_HEISENBERGSU2
#define STRAWBERRY_HEISENBERGSU2

#include "symmetry/SU2.h"
#include "bases/SpinBaseSU2.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

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
	MAKE_TYPEDEFS(HeisenbergSU2)
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
	static qarray<1> singlet() {return qarray<1>{1};};
	
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
	   \describe_boundary
	*/
	HeisenbergSU2 (const size_t &L, const vector<Param> &params={}, const BC & boundary=BC::OPEN);
	///\}
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param pushlist : All the local operators for the Mpo will be pushed into \p pushlist.
	 * \param labellist : All the labels for the Mpo will be put into \p labellist. Mpo::generate_label will produce a nice label from the data in labellist.
	 * \describe_boundary 
	*/
    static void set_operators (const std::vector<SpinBase<Symmetry> > &B, const ParamHandler &P, PushType<SiteOperator<Symmetry,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);

	
	///@{
	/**Observables.*/
	Mpo<Symmetry,double> S     (std::size_t locx,  std::size_t locy=0, double factor=1.) const;
	Mpo<Symmetry,double> Sdag  (std::size_t locx,  std::size_t locy=0, double factor=sqrt(3.)) const;
	Mpo<Symmetry,double> SdagS (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0) const;
	///@}
	
	Mpo<Symmetry,complex<double> > S_ky    (const vector<complex<double> > &phases) const;
	Mpo<Symmetry,complex<double> > Sdag_ky (const vector<complex<double> > &phases, double factor=sqrt(3.)) const;
	
	/**Validates whether a given total quantum number \p qnum is a possible target quantum number for an Mps.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<1> qnum) const;
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
	
protected:
	
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> HeisenbergSU2::defaults = 
{
	{"J",1.}, {"Jprime",0.}, {"Jprimeprime",0.}, {"Jrung",1.},
	{"D",2ul}, {"CALC_SQUARE",false}, {"CYLINDER",false}, {"Ly",1ul}
};

const std::map<string,std::any> HeisenbergSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",50ul}, {"min_halfsweeps",20ul},
	{"Dinit",5ul}, {"Qinit",6ul}, {"Dlimit",100ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_HSQ}
};

HeisenbergSU2::
HeisenbergSU2 (const size_t &L, const vector<Param> &params, const BC & boundary)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN, boundary),
 ParamReturner(HeisenbergSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	B.resize(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis().qloc(),l);
	}
	
    this->set_name("Heisenberg");

    PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
    set_operators(B, P, pushlist, labellist, boundary);
    
    this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<bool>("CALC_SQUARE"));

	this->precalc_TwoSiteData();
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
S (std::size_t locx, std::size_t locy, double factor) const
{
	assert(locx<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx << "," << locy << ",factor=" << factor << ")";
	
	SiteOperator Op = factor * B[locx].S(locy).plain<double>();
	
	Mpo<Symmetry> Mout(N_sites, Op.Q, ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx,Op);
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
Sdag (std::size_t locx, std::size_t locy, double factor) const
{
	assert(locx<this->N_sites);
	std::stringstream ss;
	ss << "S†(" << locx << "," << locy << ",factor=" << factor << ")";
	
	SiteOperator Op = factor * B[locx].Sdag(locy).plain<double>();
	
	Mpo<Symmetry> Mout(N_sites, Op.Q, ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx,Op);
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
SdagS (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	std::stringstream ss;
	ss << "S†(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites,Symmetry::qvacuum(),ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	if (locx1 == locx2)
	{
		auto product = std::sqrt(3.)*OperatorType::prod(B[locx1].Sdag(locy1), B[locx2].S(locy2),Symmetry::qvacuum());
		Mout.setLocal(locx1, product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.)*B[locx1].Sdag(locy1)).plain<double>(), B[locx2].S(locy2).plain<double>()});
		return Mout;
	}
}

Mpo<Sym::SU2<Sym::SpinSU2>,complex<double> > HeisenbergSU2::
S_ky (const vector<complex<double> > &phases) const
{
	stringstream ss;
	ss << "S" << "_ky(";
	for (int l=0; l<phases.size(); ++l)
	{
		ss << phases[l];
		if (l!=phases.size()-1) {ss << ",";}
		else                    {ss << ")";}
	}
	
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = B[l].S(0);
	}
	
	// all Ops[l].Q() must match
	Mpo<Symmetry,complex<double> > Mout(N_sites, Ops[0].Q(), ss.str(), false, false, false, BC::OPEN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
	for (int l=0; l<OpsPlain.size(); ++l)
	{
		OpsPlain[l] = Ops[l].plain<double>().cast<complex<double> >();
	}
	
	Mout.setLocalSum(OpsPlain, phases);
	
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2>,complex<double> > HeisenbergSU2::
Sdag_ky (const vector<complex<double> > &phases, double factor) const
{
	stringstream ss;
	ss << "S†" << "_ky(";
	for (int l=0; l<phases.size(); ++l)
	{
		ss << phases[l];
		if (l!=phases.size()-1) {ss << ",";}
		else                    {ss << ")";}
	}
	
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = B[l].Sdag(0);
	}
	
	// all Ops[l].Q() must match
	Mpo<Symmetry,complex<double> > Mout(N_sites, Ops[0].Q(), ss.str(), false, false, false, BC::OPEN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	vector<complex<double> > phases_x_factor = phases;
	for (int l=0; l<phases.size(); ++l)
	{
		phases_x_factor[l] = phases[l] * factor;
	}
	
	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
	for (int l=0; l<OpsPlain.size(); ++l)
	{
		OpsPlain[l] = Ops[l].plain<double>().cast<complex<double> >();
	}
	
	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
	return Mout;
}

bool HeisenbergSU2::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0]-1,2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if (Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

void HeisenbergSU2::
set_operators (const vector<SpinBase<Symmetry> > &B, const ParamHandler &P, PushType<SiteOperator<Symmetry,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		
		std::size_t orbitals       = B[loc].orbitals();
		std::size_t next1_orbitals = B[lp1].orbitals();
		std::size_t next2_orbitals = B[lp2].orbitals();
		std::size_t next3_orbitals = B[lp3].orbitals();
		
		stringstream ss1, ss2;
		ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss1.str());
		labellist[loc].push_back(ss2.str());
		
		// Case where a full coupling matrix is provided: Jᵢⱼ (all the code below this funtion will be skipped then.)
		if (P.HAS("Jfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Jfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
                
                if(range != 0)
                {
                    vector<SiteOperator<Symmetry,double> > ops(range+1);
                    ops[0] = B[loc].Sdag(0).plain<double>();
                    for (size_t i=1; i<range; ++i)
                    {
                        ops[i] = B[(loc+i)%N_sites].Id().plain<double>();
                    }
                    
                    ops[range] = B[(loc+range)%N_sites].S(0).plain<double>();
                        
                    pushlist.push_back(std::make_tuple(loc, ops, std::sqrt(3.) * value));
				}
			}
			
			stringstream ss;
			ss << "Jᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
			continue;
		}
		
		// Local Terms: J⟂
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		labellist[loc].push_back(Jperp.label);
		
		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction(B[loc].HeisenbergHamiltonian(Jperp.a).plain<double>());
        pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Nearest-neighbour terms: J
		param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next1_orbitals}, loc%Lcell);
		labellist[loc].push_back(Jpara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for(std::size_t alfa=0; alfa<orbitals; ++alfa)
			for(std::size_t beta=0; beta<next1_orbitals; ++beta)
			{
				auto ops = Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Sdag(alfa).plain<double>(), B[lp1].S(beta).plain<double>());
                pushlist.push_back(std::make_tuple(loc,ops,std::sqrt(3.)*Jpara(alfa,beta)));
			}
		}

		// Next-nearest-neighbour terms: J'
		
		param2d Jprime = P.fill_array2d<double>("Jprime", "Jprime_array", {orbitals, next2_orbitals}, loc%Lcell);
		labellist[loc].push_back(Jprime.label);
		
		if (loc < N_sites-2 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals; ++alfa)
			for (std::size_t beta=0; beta<next2_orbitals; ++beta)
			{
				auto ops = Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Sdag(alfa).plain<double>(), B[lp1].Id().plain<double>(), B[lp2].S(beta).plain<double>());
				pushlist.push_back(std::make_tuple(loc,ops,std::sqrt(3.)*Jprime(alfa,beta)));
			}
		}
		
		// 3rd-neighbour terms: J''
		
		param2d Jprimeprime = P.fill_array2d<double>("Jprimeprime", "Jprimeprime_array", {orbitals, next3_orbitals}, loc%Lcell);
		labellist[loc].push_back(Jprimeprime.label);
		
		if (loc < N_sites-3 or !static_cast<bool>(boundary))
		{
			
			for(std::size_t alfa=0; alfa<orbitals; ++alfa)
			for(std::size_t beta=0; beta<next3_orbitals; ++beta)
			{
				auto ops = Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Sdag(alfa).plain<double>(), B[lp1].Id().plain<double>(), B[lp2].Id().plain<double>(), B[lp3].S(beta).plain<double>());
                pushlist.push_back(std::make_tuple(loc,ops,std::sqrt(3.)*Jprimeprime(alfa,beta)));
			}
		}
	}
}

} //end namespace VMPS

#endif
