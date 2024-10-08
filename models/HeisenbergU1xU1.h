#ifndef DOUBLEHEISENBERGMODELU1
#define DOUBLEHEISENBERGMODELU1

#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
#include "bases/SpinBase.h"
#include "bases/FermionBase.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergObservables.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

class HeisenbergU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::SpinU1> > ,double>,
					    public HeisenbergObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::SpinU1> > >,
					    public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::SpinU1> > Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1xU1)
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HeisenbergU1xU1() : Mpo(){};
	
	HeisenbergU1xU1(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HeisenbergObservables(this->N_sites,params,HeisenbergU1::defaults),
	 ParamReturner(HeisenbergU1xU1::sweep_defaults)
	{
		ParamHandler P(params,HeisenbergU1xU1::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	HeisenbergU1xU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);

	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param pushlist : All the local operators for the Mpo will be pushed into \p pushlist.
	 * \param labellist : All the labels for the Mpo will be put into \p labellist. Mpo::generate_label will produce a nice label from the data in labellist.
	 * \describe_boundary 
	*/
//	template<typename Symmetry_> 
//	static void set_operators (const std::vector<SpinBase<Symmetry_,0ul>> &B0, const std::vector<SpinBase<Symmetry_,1ul>> &B1, 
//	                           const ParamHandler &P,
//	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
//	                           const BC boundary=BC::OPEN);
	
	static qarray<2> singlet (int N=0) {return qarray<2>{0,0};};
	static constexpr MODEL_FAMILY FAMILY = HEISENBERG;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

const std::map<string,std::any> HeisenbergU1xU1::defaults = 
{
	{"J",0.}, {"Jprime",0.}, {"Jrung",0.},
	{"Jxy",0.}, {"Jxyprime",0.}, {"Jxyrung",0.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	{"R",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"Bz",0.}, {"Kz",0.},
	{"mu",0.}, {"nu",0.}, // couple to Sz_i-1/2 and Sz_i+1/2
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}, {"mfactor",1}
};

const map<string,any> HeisenbergU1xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1e-11}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 2ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",1ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",6ul},
	{"Dinit",3ul}, {"Qinit",20ul}, {"Dlimit",500ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HeisenbergU1xU1::
HeisenbergU1xU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HeisenbergObservables(L,params,HeisenbergU1xU1::defaults),
 ParamReturner(HeisenbergU1xU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		N_phys += P.get<size_t>("Ly",l%Lcell);
//		setLocBasis((B0[l].get_basis().combine(B1[l].get_basis())).qloc(),l);
//	}
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis().qloc(),l);
	}
	
	this->set_name("HeisenbergSystemU1xBathU1");
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	//set_operators(B0, B1, P, pushlist, labellist, boundary);
	HeisenbergU1::set_operators(B, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

//template<typename Symmetry_>
//void HeisenbergU1xU1::
//set_operators (const std::vector<SpinBase<Symmetry_,0ul>> &B0, const std::vector<SpinBase<Symmetry_,1ul>> &B1, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
//{
//	std::size_t Lcell = P.size();
//	std::size_t N_sites = B0.size();
//	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
//	
//	for (std::size_t loc=0; loc<N_sites; ++loc)
//	{
//		stringstream ss;
//		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
//		labellist[loc].push_back(ss.str());
//		
//		auto push_full = [&N_sites, &loc, &B0, &B1, &P, &pushlist, &labellist, &boundary]
//			(string xxxFull, string label,
//			 const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > &first,
//			 const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > &last,
//			 vector<double> factor) -> void
//		{
//			ArrayXXd Full = P.get<Eigen::ArrayXXd>(xxxFull);
//			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
//			
//			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
//			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
//			
//			for (size_t j=0; j<first.size(); j++)
//			for (size_t h=0; h<R[loc].size(); ++h)
//			{
//				size_t range = R[loc][h].first;
//				double value = R[loc][h].second;
//				
//				if (range != 0)
//				{
//					vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > ops(range+1);
//					ops[0] = first[j];
//					for (size_t i=1; i<range; ++i)
//					{
//						ops[i] = OperatorType::outerprod(B0[(loc+i)%N_sites].Id(), B1[(loc+i)%N_sites].Id());
//					}
//					ops[range] = last[j][(loc+range)%N_sites];
//					pushlist.push_back(std::make_tuple(loc, ops, factor[j] * value));
//				}
//			}
//			
//			stringstream ss;
//			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
//			labellist[loc].push_back(ss.str());
//		};
//		
//		if (P.HAS("Kfull"))
//		{
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first
//			{
//				OperatorType::outerprod(B0[loc].Sp(0), B1[loc].Sm(0)),
//				OperatorType::outerprod(B0[loc].Sm(0), B1[loc].Sp(0)),
////				OperatorType::outerprod(B0[loc].Sz(0), B1[loc].Sz(0))
//			};
//			
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > S1_ranges(N_sites);
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > S2_ranges(N_sites);
////			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > S3_ranges(N_sites);
//			for (size_t i=0; i<N_sites; i++)
//			{
//				S1_ranges[i] = OperatorType::outerprod(B0[i].Sm(0), B1[i].Sp(0));
//				S2_ranges[i] = OperatorType::outerprod(B0[i].Sp(0), B1[i].Sm(0));
////				S3_ranges[i] = OperatorType::outerprod(B0[i].Sz(0), B1[i].Sz(0));
//			}
//			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {S1_ranges, S2_ranges};
//			push_full("Kfull", "Kᵢⱼ", first, last, {0.5,0.5});
//		}
//		
////		if (P.HAS("Kzfull"))
////		{
////			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first
////			{
////				OperatorType::outerprod(B0[loc].Sz(0), B1[loc].Sz(0)),
////			};
////			
////			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > S_ranges(N_sites);
////			for (size_t i=0; i<N_sites; i++)
////			{
////				S1_ranges[i] = OperatorType::outerprod(B0[i].Sz(0), B1[i].Sz(0));
////			}
////			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {S1_ranges, S2_ranges};
////			push_full("Kfull", "Kᵢⱼ", first, last, {1.});
////		}
//		
//		if (P.HAS("J1full"))
//		{
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first
//			{
//				OperatorType::outerprod(B0[loc].Sp(0), B1[loc].Id()),
//				OperatorType::outerprod(B0[loc].Sm(0), B1[loc].Id()),
//				OperatorType::outerprod(B0[loc].Sz(0), B1[loc].Id())
//			};
//			
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sp_ranges(N_sites);
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sm_ranges(N_sites);
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sz_ranges(N_sites);
//			for (size_t i=0; i<N_sites; i++)
//			{
//				Sp_ranges[i] = OperatorType::outerprod(B0[i].Sp(0), B1[i].Id());
//				Sm_ranges[i] = OperatorType::outerprod(B0[i].Sm(0), B1[i].Id());
//				Sz_ranges[i] = OperatorType::outerprod(B0[i].Sz(0), B1[i].Id());
//			}
//			
//			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sm_ranges, Sp_ranges, Sz_ranges};
//			
//			push_full("J1full", "J1ᵢⱼ", first, last, {0.5,0.5,1.0});
//		}
//		
//		if (P.HAS("J2full"))
//		{
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first
//			{
//				OperatorType::outerprod(B0[loc].Id(), B1[loc].Sp(0)),
//				OperatorType::outerprod(B0[loc].Id(), B1[loc].Sm(0)),
//				OperatorType::outerprod(B0[loc].Id(), B1[loc].Sz(0))
//			};
//			
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sp_ranges(N_sites);
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sm_ranges(N_sites);
//			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sz_ranges(N_sites);
//			for (size_t i=0; i<N_sites; i++)
//			{
//				Sp_ranges[i] = OperatorType::outerprod(B0[i].Id(), B1[i].Sp(0));
//				Sm_ranges[i] = OperatorType::outerprod(B0[i].Id(), B1[i].Sm(0));
//				Sz_ranges[i] = OperatorType::outerprod(B0[i].Id(), B1[i].Sz(0));
//			}
//			
//			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sm_ranges, Sp_ranges, Sz_ranges};
//			
//			push_full("J2full", "J2ᵢⱼ", first, last, {0.5,0.5,1.0});
//		}
//	}
//}

} //end namespace VMPS
#endif
