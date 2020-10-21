#ifndef KONDONECKLACEU1_H_
#define KONDONECKLACEU1_H_

#define OPLABELS

#include<map>
#include<string>

#include "KondoNecklaceObservables.h"
#include "symmetry/U1.h"
#include "bases/SpinBase.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

	class KondoNecklaceU1 : public Mpo<Sym::U1<Sym::SpinU1>,double>, public KondoNecklaceObservables<Sym::U1<Sym::SpinU1> >, public ParamReturner
{
public:
    typedef Sym::U1<Sym::SpinU1> Symmetry;
    MAKE_TYPEDEFS(KondoNecklaceU1)
    static constexpr MODEL_FAMILY FAMILY = HEISENBERG;
    
private:
    typedef typename Symmetry::qType qType;
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
    typedef Eigen::SparseMatrix<double,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixType;
    typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
    
public:
    /**
     *  Empty constructor, constructs Terms for a lattice of size 0
     */
    KondoNecklaceU1() : Mpo<Symmetry>(), ParamReturner(KondoNecklaceU1::sweep_defaults) {};

    /**
     *  Constructor
     *  @param L        Lattice size
     *  @param params   Vector of parameters for the construction of the model
     */
    KondoNecklaceU1(const std::size_t L, const std::vector<Param>& params={}, const BC boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION VERB=DMRG::VERBOSITY::ON_EXIT);


    /**
     *  Fills operators into an instance of HamiltonianTerms with respect to the Kondo Necklace model, according to given parameters.
     *  @param Bsub     Spin bases for the substrate
     *  @param Bimp     Spin bases for the impurities
     *  @param P        A ParamHandler that stores all parameter values
     *  @param Terms    An instance of HamiltonianTerms with respective Symmetry that is filled
     */
    static void set_operators(const std::vector<SpinBase<Symmetry>>& Bsub, const std::vector<SpinBase<Symmetry>>& Bimp, const ParamHandler& P,
							  PushType<SiteOperator<Symmetry,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary);

    /**
     *  Default values for model parameters, such as Heisenberg and Kondo couplings
     */
    static const std::map<string,std::any> defaults;

    /**
     *  Default values for the sweeping process
     */
    static const std::map<string,std::any> sweep_defaults;

    /**
     *  Checks whether a quantum number D = 2S+1 can be reached by any combination of substrate and impurity spins and thereby validates this quantum number
     */
    bool validate(qType qnum);
};

const std::map<string,std::any> KondoNecklaceU1::defaults = 
{
	{"Jlocxy",1.}, {"Jlocz",1.}, 
	{"Jparaxy",1.}, {"Jparaz",1.}, {"Jperpxy",0.}, {"Jperpz",0.}, {"Jprimexy",1.}, {"Jprimez",1.},
	{"Bz",0.}, {"Bzsub",0.},
	{"Dimp",2ul}, {"Dsub",2ul}, {"Ly",1ul},
	{"maxPower",2ul}, {"CYLINDER",false}
};

const std::map<string,std::any> KondoNecklaceU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1e-11}, {"lim_alpha",12ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",1ul}, {"max_Nrich",-1},
	{"max_halfsweeps",100ul}, {"min_halfsweeps",24ul},
	{"Dinit",5ul}, {"Qinit",6ul}, {"Dlimit",250ul},
	{"tol_eigval",1e-9}, {"tol_state",1e-8},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_HSQ}
};

KondoNecklaceU1::KondoNecklaceU1(const std::size_t L, const std::vector<Param>& params, const BC boundary, const DMRG::VERBOSITY::OPTION VERB)
: Mpo<Symmetry>(L, Symmetry::qvacuum(), "KondoNecklaceU1", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
  KondoNecklaceObservables<Symmetry>(L,params,KondoNecklaceU1::defaults),
  ParamReturner(KondoNecklaceU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
    this->set_verbosity(VERB);
	std::size_t Lcell = P.size();
    Bsub.resize(N_sites);
	Bimp.resize(N_sites);
	for (size_t loc=0; loc<N_sites; ++loc)
	{
		N_phys += P.get<size_t>("Ly",loc%Lcell);
		setLocBasis((Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc(),loc);
	}

    PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist(N_sites);
    set_operators(Bsub, Bimp, P, pushlist, labellist, boundary);
    
    this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<std::size_t>("maxPower"));
    
    this->precalc_TwoSiteData();
}
    
void KondoNecklaceU1::set_operators(const std::vector<SpinBase<Symmetry>>& Bsub, const std::vector<SpinBase<Symmetry>>& Bimp, const ParamHandler &P,
									 PushType<SiteOperator<Symmetry,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
    std::size_t Lcell = P.size();
    std::size_t N_sites = Bsub.size();
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t orbitals = Bsub[loc].orbitals();
        std::size_t next_orbitals = Bsub[(loc+1)%N_sites].orbitals();
        std::size_t nextn_orbitals = Bsub[(loc+2)%N_sites].orbitals();
        

        
        std::stringstream ss1, ss2, ss3;
        ss1 << "S_sub=" << print_frac_nice(frac(P.get<size_t>("Dsub",loc%Lcell)-1,2));
        ss2 << "S_imp=" << print_frac_nice(frac(P.get<size_t>("Dimp",loc%Lcell)-1,2));
        ss3 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
        labellist[loc].push_back(ss1.str());
        labellist[loc].push_back(ss2.str());
        labellist[loc].push_back(ss3.str());

		// Bz substrate
		param1d Bzsub = P.fill_array1d<double>("Bzsub", "Bzsuborb", orbitals, loc%Lcell);
		labellist[loc].push_back(Bzsub.label);
		
		// Bz impurities
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		labellist[loc].push_back(Bz.label);

		auto Himp = kroneckerProduct(Bsub[loc].Id(),Bimp[loc].HeisenbergHamiltonian(Bimp[loc].ZeroHopping(),Bimp[loc].ZeroHopping(),Bz.a,Bimp[loc].ZeroField(),Bimp[loc].ZeroField()));
		auto Hsub = kroneckerProduct(Bsub[loc].HeisenbergHamiltonian(Bsub[loc].ZeroHopping(),Bsub[loc].ZeroHopping(),Bzsub.a,Bsub[loc].ZeroField(),Bsub[loc].ZeroField()),Bimp[loc].Id());
		auto Hloc = Himp + Hsub;
        
        // Local Terms: J_Kondo
        param1d Jlocxy = P.fill_array1d<double>("Jlocxy", "Jlocxy_array", orbitals, loc%Lcell);
		param1d Jlocz = P.fill_array1d<double>("Jlocz", "Jlocz_array", orbitals, loc%Lcell);
        if(Jlocxy.x != 0.)
        {
            labellist[loc].push_back(Jlocxy.label);
			labellist[loc].push_back(Jlocz.label);
            for(int alpha=0; alpha<orbitals; ++alpha)
            {
				Hloc += 0.5*Jlocxy(alpha) * kroneckerProduct(Bsub[loc].Scomp(SP,alpha), Bimp[loc].Scomp(SM,alpha));
				Hloc += 0.5*Jlocxy(alpha) * kroneckerProduct(Bsub[loc].Scomp(SM,alpha), Bimp[loc].Scomp(SP,alpha));
				Hloc +=     Jlocz(alpha)  * kroneckerProduct(Bsub[loc].Scomp(SZ,alpha), Bimp[loc].Scomp(SZ,alpha));
            }
        }
		pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(Hloc), 1.));
				
		auto push_full = [&N_sites, &loc, &Bimp, &Bsub, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
																						 const vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > &first,
																						 const vector<vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > > &last,
																						 vector<double> factor) -> void
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>(xxxFull);
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}

			for (size_t j=0; j<first.size(); j++)
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				if (range != 0)
				{
					vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						ops[i] = kroneckerProduct(Bsub[(loc+i)%N_sites].Id(), Bimp[(loc+i)%N_sites].Id());
					}
					ops[range] = last[j][(loc+range)%N_sites];
					pushlist.push_back(std::make_tuple(loc, ops, factor[j] * value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};

		// Case where a full coupling matrix is providedf: Jᵢⱼ (all the code below this funtion will be skipped then.)
		if (P.HAS("Jxyfull"))
		{
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > first {kroneckerProduct(Bsub[loc].Scomp(SP,0),Bimp[loc].Id()),
																	kroneckerProduct(Bsub[loc].Scomp(SM,0),Bimp[loc].Id())};
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > Sm_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Sm_ranges[i] = kroneckerProduct(Bsub[i].Scomp(SM,0),Bimp[i].Id());}
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > Sp_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = kroneckerProduct(Bsub[i].Scomp(SP,0),Bimp[i].Id());}
			vector<vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > > last {Sm_ranges,Sp_ranges};
			push_full("Jxyfull", "Jxyᵢⱼ", first, last, {0.5,0.5});
		}
		if (P.HAS("Jzfull"))
		{
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > first {kroneckerProduct(Bsub[loc].Scomp(SZ,0),Bimp[loc].Id())};
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > Sz_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Sz_ranges[i] = kroneckerProduct(Bsub[i].Scomp(SZ,0),Bimp[i].Id());}
			vector<vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > > last {Sz_ranges};
			push_full("Jzfull", "Jzᵢⱼ", first, last, {1.});
		}
		if (P.HAS("Ixyfull"))
		{
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > first {kroneckerProduct(Bsub[loc].Id(),Bimp[loc].Scomp(SP,0)),
																	kroneckerProduct(Bsub[loc].Id(),Bimp[loc].Scomp(SM,0))};
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > Sm_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Sm_ranges[i] = kroneckerProduct(Bsub[i].Id(),Bimp[i].Scomp(SM,0));}
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > Sp_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = kroneckerProduct(Bsub[i].Id(),Bimp[i].Scomp(SP,0));}
			vector<vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > > last {Sm_ranges,Sp_ranges};
			push_full("Ixyfull", "Ixyᵢⱼ", first, last, {0.5,0.5});
		}
		if (P.HAS("Izfull"))
		{
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > first {kroneckerProduct(Bsub[loc].Id(),Bimp[loc].Scomp(SZ,0))};
			vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > Sz_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Sz_ranges[i] = kroneckerProduct(Bsub[i].Id(),Bimp[i].Scomp(SZ,0));}
			vector<vector<SiteOperatorQ<Symmetry,Eigen::MatrixXd> > > last {Sz_ranges};
			push_full("Izfull", "Izᵢⱼ", first, last, {1.});
		}
		if (P.HAS("Jzfull") or P.HAS("Jxyfull") or P.HAS("Izfull") or P.HAS("Ixyfull")) {continue;}
		

        // Nearest-neighbour terms: J
        
    //     param2d Jpara = P.fill_array2d<double>("Jpara", "Jpara_array", {orbitals, next_orbitals}, loc%Lcell);
    //     if((Jpara.a != 0.).any())
    //     {
    //         labellist[loc].push_back(Jpara.label);
    //         if(loc < N_sites-1 or boundary == BC::INFINITE)
    //         {
    //             for(int alpha=0; alpha<orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<next_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jpara.a(alpha,beta);
    //                     std::vector<OperatorType> ops(2);
    //                     ops[0] = OperatorType::outerprod(Bsub[loc].Sdag(alpha), Bimp[loc].Id(), {3});
    //                     ops[1] = OperatorType::outerprod(Bsub[(loc+1)%N_sites].S(beta), Bimp[(loc+1)%N_sites].Id(), {3});
    //                     pushlist.push_back(std::make_tuple(loc, ops, lambda));
    //                 }
    //             }
    //         }
    //     }

        
    //     // Next-nearest-neighbour terms: J
        
    //     param2d Jprime = P.fill_array2d<double>("Jprime", "Jprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
    //     if((Jprime.a != 0.).any())
    //     {
    //         labellist[loc].push_back(Jprime.label);
    //         if((N_sites > 1 and loc < N_sites-2) or boundary == BC::INFINITE)
    //         {
    //             for(int alpha=0; alpha<orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<nextn_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jprime.a(alpha,beta);
    //                     std::vector<OperatorType> ops(3);
    //                     ops[0] = OperatorType::outerprod(Bsub[loc].Sdag(alpha), Bimp[loc].Id(), {3});
    //                     ops[1] = OperatorType::outerprod(Bsub[(loc+1)%N_sites].Id(), Bimp[(loc+1)%N_sites].Id(), {1});
    //                     ops[2] = OperatorType::outerprod(Bsub[(loc+2)%N_sites].S(beta), Bimp[(loc+2)%N_sites].Id(), {3});
    //                     pushlist.push_back(std::make_tuple(loc, ops, lambda));
    //                 }
    //             }
    //         }
    //     }
    // }
    
        
    // if(false) //(boundary == BC::PERIODIC)
    // {
    //     std::size_t last_orbitals = Bsub[N_sites-1].orbitals();
    //     std::size_t first_orbitals = Bsub[0].orbitals();
    //     std::size_t previous_orbitals = Bsub[(2*N_sites-2)%N_sites].orbitals();
    //     std::size_t next_orbitals = Bsub[1%N_sites].orbitals();
        
    //     if(N_sites == 1)
    //     {
    //         param2d Jpara = P.fill_array2d<double>("Jpara", "Jpara_array", {first_orbitals,first_orbitals}, 0);
    //         if((Jpara.a != 0.).any())
    //         {
    //             labellist[0].push_back(Jpara.label);
    //             for(int alpha=0; alpha<last_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<first_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jpara.a(alpha,beta);
    //                     std::vector<OperatorType> ops(1);
    //                     ops[0] = OperatorType::outerprod(OperatorType::prod(Bsub[0].Sdag(alpha),Bsub[0].S(beta),{1}),Bimp[0].Id(),{1});
    //                     pushlist.push_back(std::make_tuple(0ul, ops, lambda));
    //                 }
    //             }
    //         }
            
    //         param2d Jprime = P.fill_array2d<double>("Jprime", "Jprime_array", {first_orbitals, first_orbitals}, 0%Lcell);
    //         if((Jprime.a != 0.).any())
    //         {
    //             labellist[0].push_back(Jprime.label);
    //             for(int alpha=0; alpha<first_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<first_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jprime.a(alpha,beta);
    //                     std::vector<OperatorType> ops(1);
    //                     ops[0] = OperatorType::outerprod(OperatorType::prod(Bsub[0].Sdag(alpha),Bsub[0].S(beta),{1}),Bimp[0].Id(),{1});
    //                     pushlist.push_back(std::make_tuple(0ul, ops, lambda));
    //                 }
    //             }
    //         }
    //     }
    //     else if(N_sites == 2)
    //     {
    //         param2d Jpara = P.fill_array2d<double>("Jpara", "Jpara_array", {last_orbitals,first_orbitals}, 1%Lcell);
    //         if((Jpara.a != 0.).any())
    //         {
    //             labellist[1].push_back(Jpara.label);
    //             for(int alpha=0; alpha<last_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<first_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jpara.a(alpha,beta);
    //                     std::vector<OperatorType> ops(2);
    //                     ops[0] = OperatorType::outerprod(Bsub[0].S(beta), Bimp[0].Id(), {3});
    //                     ops[1] = OperatorType::outerprod(Bsub[1].Sdag(alpha), Bimp[1].Id(), {3});
    //                     pushlist.push_back(std::make_tuple(0ul, ops, lambda));
    //                 }
    //             }
    //         }
            
    //         param2d Jprime_prev_to_first = P.fill_array2d<double>("Jprime", "Jprime_array", {first_orbitals, first_orbitals}, 0%Lcell);
    //         if((Jprime_prev_to_first.a != 0.).any())
    //         {
    //             labellist[0].push_back(Jprime_prev_to_first.label);
    //             for(int alpha=0; alpha<first_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<first_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jprime_prev_to_first.a(alpha,beta);
    //                     std::vector<OperatorType> ops(1);
    //                     ops[0] = OperatorType::outerprod(OperatorType::prod(Bsub[0].Sdag(alpha),Bsub[0].S(beta),{1}),Bimp[0].Id(),{1});
    //                     pushlist.push_back(std::make_tuple(0ul, ops, lambda));
    //                 }
    //             }
    //         }
            
    //         param2d Jprime_last_to_next = P.fill_array2d<double>("Jprime", "Jprime_array", {last_orbitals, last_orbitals}, 1%Lcell);
    //         if((Jprime_last_to_next.a != 0.).any())
    //         {
    //             labellist[1].push_back(Jprime_last_to_next.label);
    //             for(int alpha=0; alpha<last_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<last_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jprime_last_to_next.a(alpha,beta);
    //                     std::vector<OperatorType> ops(1);
    //                     ops[0] = OperatorType::outerprod(OperatorType::prod(Bsub[1].Sdag(alpha),Bsub[1].S(beta),{1}),Bimp[1].Id(),{1});
    //                     pushlist.push_back(std::make_tuple(1ul, ops, lambda));
    //                 }
    //             }
    //         }
    //     }
    //     else
    //     {
    //         param2d Jpara = P.fill_array2d<double>("Jpara", "Jpara_array", {last_orbitals,first_orbitals}, (N_sites-1)%Lcell);
    //         if((Jpara.a != 0.).any())
    //         {
    //             labellist[N_sites-1].push_back(Jpara.label);
    //             for(int alpha=0; alpha<last_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<first_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jpara.a(alpha,beta);
    //                     std::vector<OperatorType> ops(N_sites);
    //                     ops[0] = OperatorType::outerprod(Bsub[0].S(beta), Bimp[0].Id(), {3});
    //                     for(std::size_t loc=1; loc<N_sites-1; ++loc)
    //                     {
    //                         ops[loc] = OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1});
    //                     }
    //                     ops[N_sites-1] = OperatorType::outerprod(Bsub[N_sites-1].Sdag(alpha), Bimp[N_sites-1].Id(), {3});
    //                     pushlist.push_back(std::make_tuple(0ul, ops, lambda));
    //                 }
    //             }
    //         }
            
    //         param2d Jprime_prev_to_first = P.fill_array2d<double>("Jprime", "Jprime_array", {previous_orbitals, first_orbitals}, (N_sites-2)%Lcell);
    //         if((Jprime_prev_to_first.a != 0.).any())
    //         {
    //             labellist[N_sites-2].push_back(Jprime_prev_to_first.label);
    //             for(int alpha=0; alpha<previous_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<first_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jprime_prev_to_first.a(alpha,beta);
    //                     std::vector<OperatorType> ops(N_sites-1);
    //                     ops[0] = OperatorType::outerprod(Bsub[0].S(beta), Bimp[0].Id(), {3});
    //                     for(std::size_t loc=1; loc<N_sites-2; ++loc)
    //                     {
    //                         ops[loc] = OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1});
    //                     }
    //                     ops[N_sites-2] = OperatorType::outerprod(Bsub[N_sites-2].Sdag(alpha), Bimp[N_sites-2].Id(), {3});
    //                     pushlist.push_back(std::make_tuple(0ul, ops, lambda));
    //                 }
    //             }
    //         }
            
    //         param2d Jprime_last_to_next = P.fill_array2d<double>("Jprime", "Jprime_array", {last_orbitals, next_orbitals}, (N_sites-1)%Lcell);
    //         if((Jprime_last_to_next.a != 0.).any())
    //         {
    //             labellist[N_sites-1].push_back(Jprime_last_to_next.label);
    //             for(int alpha=0; alpha<last_orbitals; ++alpha)
    //             {
    //                 for(int beta=0; beta<next_orbitals; ++beta)
    //                 {
    //                     double lambda = std::sqrt(3.)*Jprime_last_to_next.a(alpha,beta);
    //                     std::vector<OperatorType> ops(N_sites-1);
    //                     ops[0] = OperatorType::outerprod(Bsub[1].S(beta), Bimp[1].Id(), {3});
    //                     for(std::size_t loc=2; loc<N_sites-1; ++loc)
    //                     {
    //                         ops[loc-1] = OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1});
    //                     }
    //                     ops[N_sites-2] = OperatorType::outerprod(Bsub[N_sites-1].Sdag(alpha), Bimp[N_sites-1].Id(), {3});
    //                     pushlist.push_back(std::make_tuple(1ul, ops, lambda));
    //                 }
    //             }
    //         }
		//       }
    }
}

// bool KondoNecklaceU1::validate(qType qnum)
// {
//     auto add = [](std::set<std::size_t>& left, std::set<std::size_t>& right) -> void
//     {
//         if(left.size() == 0)
//         {
//             left = right;
//         }
//         else
//         {
//             std::set<std::size_t> temp;
//             for(auto l : left) for(auto r : right)
//             {
//                 std::size_t min = std::abs((static_cast<int>(l))-(static_cast<int>(r)))+1;
//                 std::size_t max = l+r-1;
//                 for(std::size_t i=min; i<=max; i+=2ul) temp.insert(i);
//             }
//             left = temp;
//         }
//     };
//     std::vector<std::set<std::size_t>> local(N_sites);
//     std::vector<std::set<std::size_t>> reachable(N_sites);
//     for(std::size_t loc=0; loc<N_sites; ++loc)
//     {
//         std::set<std::size_t> subspins;
//         if(Bsub[loc].orbitals()%2 == 0) subspins.insert(1ul);
//         for(std::size_t i = Bsub[loc].get_D(); i<=Bsub[loc].orbitals()*(Bsub[loc].get_D()-1)+1; i+=2ul) subspins.insert(i);
//         add(local[loc], subspins);
//         std::set<std::size_t> impspins;
//         if(Bimp[loc].orbitals()%2 == 0) impspins.insert(1ul);
//         for(std::size_t i = Bimp[loc].get_D(); i<=Bimp[loc].orbitals()*(Bimp[loc].get_D()-1)+1; i+=2ul) impspins.insert(i);
//         add(local[loc], impspins);
//     }
//     reachable[0] = local[0];
//     for(std::size_t loc=1; loc<N_sites; ++loc)
//     {
//         reachable[loc] = local[loc];
//         add(reachable[loc], reachable[loc-1]);
//     }

//     auto it = find(reachable[N_sites-1].begin(), reachable[N_sites-1].end(), qnum[0]);
//     return it!=reachable[N_sites-1].end();
// }
    
// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// Stot()
// {
//     Mpo<Symmetry> Mout(this->N_sites, {3}, "S_tot", false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//         for(std::size_t alpha=0; alpha<Bsub[loc].orbitals(); ++alpha)
//         {
//             Mout.push(loc, {OperatorType::outerprod(Bsub[loc].S(alpha), Bimp[loc].Id(), {3}).plain<double>()});
//             Mout.push(loc, {OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].S(alpha), {3}).plain<double>()});
//         }
//     }
//     Mout.finalize();
//     return Mout;
// }
    
// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// Simp(std::size_t locx, std::size_t locy)
// {
//     assert(locx<this->N_sites);
//     assert(locy<Bimp[locx].orbitals());
//     std::stringstream ss;
//     ss << "S_imp(" << locx;
//     if(Bimp[locx].orbitals() > 1)
//     {
//         ss << "," << locy;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {3}, ss.str(), false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     Mout.push(locx, {OperatorType::outerprod(Bsub[locx].Id(), Bimp[locx].S(locy), {3}).plain<double>()});
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// Ssub(std::size_t locx, std::size_t locy)
// {
//     assert(locx<this->N_sites);
//     assert(locy<Bsub[locx].orbitals());
//     std::stringstream ss;
//     ss << "S_sub(" << locx;
//     if(Bsub[locx].orbitals() > 1)
//     {
//         ss << "," << locy;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {3}, ss.str(), false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     Mout.push(locx, {OperatorType::outerprod(Bsub[locx].S(locy), Bimp[locx].Id(), {3}).plain<double>()});
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// Simpdag(std::size_t locx, std::size_t locy)
// {
//     assert(locx<this->N_sites);
//     assert(locy<Bimp[locx].orbitals());
//     std::stringstream ss;
//     ss << "S_imp†(" << locx;
//     if(Bimp[locx].orbitals() > 1)
//     {
//         ss << "," << locy;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {3}, ss.str(), false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     Mout.push(locx, {OperatorType::outerprod(Bsub[locx].Id(), Bimp[locx].Sdag(locy), {3}).plain<double>()});
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// Ssubdag(std::size_t locx, std::size_t locy)
// {
//     assert(locx<this->N_sites);
//     assert(locy<Bsub[locx].orbitals());
//     std::stringstream ss;
//     ss << "S_sub†(" << locx;
//     if(Bsub[locx].orbitals() > 1)
//     {
//         ss << "," << locy;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {3}, ss.str(), false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     Mout.push(locx, {OperatorType::outerprod(Bsub[locx].Sdag(locy), Bimp[locx].Id(), {3}).plain<double>()});
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// SimpdagSimp(std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
// {
//     assert(locx1<this->N_sites);
//     assert(locy1<Bimp[locx1].orbitals());
//     assert(locx2<this->N_sites);
//     assert(locy2<Bimp[locx2].orbitals());
//     std::stringstream ss;
//     ss << "S_imp†(" << locx1;
//     if(Bimp[locx1].orbitals() > 1)
//     {
//         ss << "," << locy1;
//     }
//     ss << ")*S_imp(" << locx2;
//     if(Bimp[locx2].orbitals() > 1)
//     {
//         ss << "," << locy2;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {1}, ss.str(), true, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     std::vector<SiteOperator<Symmetry,double>> ops;
//     if(locx1 == locx2)
//     {
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Id(), OperatorType::prod(Bimp[locx1].Sdag(locy1), Bimp[locx1].S(locy2), {1}), {1}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else if(locx1 < locx2)
//     {
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Id(), Bimp[locx1].Sdag(locy1), {3}).plain<double>());
//         for(std::size_t loc=locx1+1; loc<locx2; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].Id(), Bimp[locx2].S(locy2), {3}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else
//     {
//         std::stringstream ss2;
//         ss2 << "S_imp(" << locx2;
//         if(Bimp[locx2].orbitals() > 1)
//         {
//             ss2 << "," << locy2;
//         }
//         ss2 << ")*S_imp†(" << locx1;
//         if(Bimp[locx1].orbitals() > 1)
//         {
//             ss2 << "," << locy1;
//         }
//         ss2 << ")";
//         Mout.set_name(ss2.str());
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].Id(), Bimp[locx2].S(locy2), {3}).plain<double>());
//         for(std::size_t loc=locx2+1; loc<locx1; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Id(), Bimp[locx1].Sdag(locy1), {3}).plain<double>());
//         Mout.push(locx2, ops, {1});
//     }
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// SsubdagSsub(std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
// {
//     assert(locx1<this->N_sites);
//     assert(locy1<Bsub[locx1].orbitals());
//     assert(locx2<this->N_sites);
//     assert(locy2<Bsub[locx2].orbitals());
//     std::stringstream ss;
//     ss << "S_sub†(" << locx1;
//     if(Bimp[locx1].orbitals() > 1)
//     {
//         ss << "," << locy1;
//     }
//     ss << ")*S_sub(" << locx2;
//     if(Bimp[locx2].orbitals() > 1)
//     {
//         ss << "," << locy2;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {1}, ss.str(), true, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     std::vector<SiteOperator<Symmetry,double>> ops;
//     if(locx1 == locx2)
//     {
//         ops.push_back(OperatorType::outerprod(OperatorType::prod(Bsub[locx1].Sdag(locy1),Bsub[locx1].S(locy2),{1}), Bimp[locx1].Id(), {1}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else if(locx1 < locx2)
//     {
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Sdag(locy1), Bimp[locx1].Id(), {3}).plain<double>());
//         for(std::size_t loc=locx1+1; loc<locx2; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].S(locy2), Bimp[locx2].Id(), {3}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else
//     {
//         std::stringstream ss2;
//         ss2 << "S_sub(" << locx2;
//         if(Bimp[locx2].orbitals() > 1)
//         {
//             ss2 << "," << locy2;
//         }
//         ss2 << ")*S_sub†(" << locx1;
//         if(Bimp[locx1].orbitals() > 1)
//         {
//             ss2 << "," << locy1;
//         }
//         ss2 << ")";
//         Mout.set_name(ss2.str());
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].S(locy2), Bimp[locx2].Id(), {3}).plain<double>());
//         for(std::size_t loc=locx2+1; loc<locx1; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Sdag(locy1), Bimp[locx1].Id(), {3}).plain<double>());
//         Mout.push(locx2, ops, {1});
//     }
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// SsubdagSimp(std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
// {
//     assert(locx1<this->N_sites);
//     assert(locy1<Bsub[locx1].orbitals());
//     assert(locx2<this->N_sites);
//     assert(locy2<Bimp[locx2].orbitals());
//     std::stringstream ss;
//     ss << "S_sub†(" << locx1;
//     if(Bimp[locx1].orbitals() > 1)
//     {
//         ss << "," << locy1;
//     }
//     ss << ")*S_imp(" << locx2;
//     if(Bimp[locx2].orbitals() > 1)
//     {
//         ss << "," << locy2;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {1}, ss.str(), false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     std::vector<SiteOperator<Symmetry,double>> ops;
//     if(locx1 == locx2)
//     {
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Sdag(locy1), Bimp[locx1].S(locy2), {1}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else if(locx1 < locx2)
//     {
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Sdag(locy1), Bimp[locx1].Id(), {3}).plain<double>());
//         for(std::size_t loc=locx1+1; loc<locx2; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].Id(), Bimp[locx2].S(locy2), {3}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else
//     {
//         std::stringstream ss2;
//         ss2 << "S_imp(" << locx2;
//         if(Bimp[locx2].orbitals() > 1)
//         {
//             ss2 << "," << locy2;
//         }
//         ss2 << ")*S_sub†(" << locx1;
//         if(Bimp[locx1].orbitals() > 1)
//         {
//             ss2 << "," << locy1;
//         }
//         ss2 << ")";
//         Mout.set_name(ss2.str());
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].Id(), Bimp[locx2].S(locy2), {3}).plain<double>());
//         for(std::size_t loc=locx2+1; loc<locx1; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Sdag(locy1), Bimp[locx1].Id(), {3}).plain<double>());
//         Mout.push(locx2, ops, {1});
//     }
//     Mout.finalize();
//     return Mout;
// }

// Mpo<Sym::SU2<Sym::SpinSU2>> KondoNecklaceU1::
// SimpdagSsub(std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
// {
//     assert(locx1<this->N_sites);
//     assert(locy1<Bimp[locx1].orbitals());
//     assert(locx2<this->N_sites);
//     assert(locy2<Bsub[locx2].orbitals());
//     std::stringstream ss;
//     ss << "S_imp†(" << locx1;
//     if(Bimp[locx1].orbitals() > 1)
//     {
//         ss << "," << locy1;
//     }
//     ss << ")*S_sub(" << locx2;
//     if(Bimp[locx2].orbitals() > 1)
//     {
//         ss << "," << locy2;
//     }
//     ss << ")";
//     Mpo<Symmetry> Mout(N_sites, {1}, ss.str(), false, false, BC::OPEN, DMRG::VERBOSITY::OPTION::SILENT);
//     for(std::size_t loc=0; loc<this->N_sites; ++loc)
//     {
//         Mout.set_qPhys(loc, (Bsub[loc].get_basis().combine(Bimp[loc].get_basis())).qloc());
//     }
//     std::vector<SiteOperator<Symmetry,double>> ops;
//     if(locx1 == locx2)
//     {
//         std::stringstream ss2;
//         ss2 << "S_sub(" << locx2;
//         if(Bimp[locx2].orbitals() > 1)
//         {
//             ss2 << "," << locy2;
//         }
//         ss2 << ")*S_imp†(" << locx1;
//         if(Bimp[locx1].orbitals() > 1)
//         {
//             ss2 << "," << locy1;
//         }
//         ss2 << ")";
//         Mout.set_name(ss2.str());
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].S(locy2), Bimp[locx1].S(locy1), {1}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else if(locx1 < locx2)
//     {
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Id(), Bimp[locx1].Sdag(locy1), {3}).plain<double>());
//         for(std::size_t loc=locx1+1; loc<locx2; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].S(locy2), Bimp[locx2].Id(), {3}).plain<double>());
//         Mout.push(locx1, ops, {1});
//     }
//     else
//     {
//         std::stringstream ss2;
//         ss2 << "S_sub(" << locx2;
//         if(Bimp[locx2].orbitals() > 1)
//         {
//             ss2 << "," << locy2;
//         }
//         ss2 << ")*S_imp†(" << locx1;
//         if(Bimp[locx1].orbitals() > 1)
//         {
//             ss2 << "," << locy1;
//         }
//         ss2 << ")";
//         Mout.set_name(ss2.str());
//         ops.push_back(OperatorType::outerprod(Bsub[locx2].S(locy2), Bimp[locx2].Id(), {3}).plain<double>());
//         for(std::size_t loc=locx2+1; loc<locx1; ++loc)
//         {
//             ops.push_back(OperatorType::outerprod(Bsub[loc].Id(), Bimp[loc].Id(), {1}).plain<double>());
//         }
//         ops.push_back(OperatorType::outerprod(Bsub[locx1].Id(), Bimp[locx1].Sdag(locy1), {3}).plain<double>());
//         Mout.push(locx2, ops, {1});
//     }
//     Mout.finalize();
//     return Mout;
// }

} // end namespace VMPS



#endif
