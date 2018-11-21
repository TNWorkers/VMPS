#ifndef STRAWBERRY_HEISENBERGSU2
#define STRAWBERRY_HEISENBERGSU2

#include "symmetry/SU2.h"
#include "bases/SpinBaseSU2.h"
#include "Mpo.h"
//include "DmrgExternal.h"
//include "ParamHandler.h" // from TOOLS
#include "ParamReturner.h"
//include "symmetry/kind_dummies.h"

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
	
	typedef DmrgSolver<Symmetry,HeisenbergSU2,double>  Solver;
	typedef VumpsSolver<Symmetry,HeisenbergSU2,double> uSolver;
	
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
    static void set_operators(const std::vector<SpinBase<Symmetry> > &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms);
	
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
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",4ul},
	{"Dinit",5ul}, {"Qinit",10ul}, {"Dlimit",100ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HeisenbergSU2::
HeisenbergSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
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
    
    HamiltonianTerms<Symmetry, double> Terms(N_sites, P.get<bool>("OPEN_BC"));
    set_operators(B,P,Terms);
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

void HeisenbergSU2::
set_operators(const vector<SpinBase<Symmetry>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms)
{
    std::size_t Lcell = P.size();
    std::size_t N_sites = Terms.size();
    Terms.set_name("HeisenbergSU2");
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t orbitals = B[loc].orbitals();
        std::size_t next_orbitals = B[(loc+1)%N_sites].orbitals();
        std::size_t nextn_orbitals = B[(loc+2)%N_sites].orbitals();
        
        stringstream ss1, ss2;
        ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
        ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
        Terms.save_label(loc, ss1.str());
        Terms.save_label(loc, ss2.str());
        
        // Local Terms: J⟂

        param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        Terms.save_label(loc, Jperp.label);
        
        for(int alpha=0; alpha<orbitals; ++alpha)
        {
            for(int beta=0; beta<orbitals; ++beta)
            {
                Terms.push_local(loc, std::sqrt(3.)*Jperp.a(alpha), OperatorType::prod(B[loc].Sdag(alpha), B[loc].S(beta), {1}).plain<double>());
            }
        }
        
        // Nearest-neighbour terms: J
        
        param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
        Terms.save_label(loc, Jpara.label);
        
        if(loc < N_sites-1 || !P.get<bool>("OPEN_BC"))
        {
            for(std::size_t alpha=0; alpha<orbitals; ++alpha)
            {
                for(std::size_t beta=0; beta<next_orbitals; ++beta)
                {
                    Terms.push_tight(loc,std::sqrt(3.)*Jpara.a(alpha,beta),
                                     B[loc].Sdag(alpha).plain<double>(),
                                     B[(loc+1)%N_sites].S(beta).plain<double>());
                }
            }
        }
        
        // Next-nearest-neighbour terms: J'
        
        param2d Jprime = P.fill_array2d<double>("Jprime", "Jprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
        Terms.save_label(loc, Jprime.label);
        
        if(loc < N_sites-2 || !P.get<bool>("OPEN_BC"))
        {
            for(std::size_t alpha=0; alpha<orbitals; ++alpha)
            {
                for(std::size_t beta=0; beta<nextn_orbitals; ++beta)
                {
                    Terms.push_nextn(loc,std::sqrt(3.)*Jprime.a(alpha, beta),
                                     B[loc].Sdag(alpha).plain<double>(),
                                     B[(loc+1)%N_sites].Id().plain<double>(),
                                     B[(loc+2)%N_sites].S(beta).plain<double>());
                }
            }
        }
    }
}

} //end namespace VMPS

#endif
