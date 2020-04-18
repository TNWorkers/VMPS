#ifndef KONDOMODEL_SU2XSU2_H_
#define KONDOMODEL_SU2XSU2_H_

#include "bases/SpinBase.h"
#include "bases/FermionBase.h"
#include "Mpo.h"
#include "models/KondoObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

class KondoSU2xSU2 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,double>,
					 public KondoObservables<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >,
					 public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	MAKE_TYPEDEFS(KondoSU2xSU2)
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	KondoSU2xSU2 (): Mpo(), KondoObservables(), ParamReturner(KondoSU2xSU2::sweep_defaults) {};
	KondoSU2xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{1,1};};
	static constexpr MODEL_FAMILY FAMILY = KONDO;
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param pushlist : All the local operators for the Mpo will be pushed into \p pushlist.
	 * \param labellist : All the labels for the Mpo will be put into \p labellist. Mpo::generate_label will produce a nice label from the data in labellist.
	 * \describe_boundary 
	*/
	template<typename Symmetry_> 
	static void set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
	static const map<string,any> sweep_defaults;	
};

const std::map<string,std::any> KondoSU2xSU2::defaults =
{
	{"t",1.}, {"tRung",0.},
	{"J",1.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}, {"LyF",1ul}
};

const map<string,any> KondoSU2xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",16ul}, {"eps_svd",1e-7},
	{"Dincr_abs",5ul}, {"Dincr_per",2ul}, {"Dincr_rel",1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",10ul},
	{"Dinit",5ul}, {"Qinit",6ul}, {"Dlimit",200ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KondoSU2xSU2::
KondoSU2xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 KondoObservables(L,params,KondoSU2xSU2::defaults),
 ParamReturner(KondoSU2xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("LyF",l%Lcell);		
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);				
	}

	this->set_name("Kondo");

	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
    set_operators(B, F, P, pushlist, labellist, boundary);
    
    this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));

	this->precalc_TwoSiteData();
}

template<typename Symmetry_> 
void KondoSU2xSU2::
set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P,
			   PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}

	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;

		auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
		auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));
		auto Glp2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp2)));
		
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[lp1].orbitals();
		std::size_t Fnextn_orbitals = F[lp2].orbitals();
		
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[lp1].orbitals();
		std::size_t Bnextn_orbitals = B[lp2].orbitals();
		
		stringstream Slabel;
		Slabel << "S=" << print_frac_nice(frac(B[loc].get_D()-1,2));
		labellist[loc].push_back(Slabel.str());

		auto push_full = [&N_sites, &loc, &B, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
																						 const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > &first,
																						 const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > &last,
																						 vector<double> factor, bool FERMIONIC) -> void
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
					vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						if (FERMIONIC) {ops[i] = kroneckerProduct(B[(loc+i)%N_sites].Id(), F[(loc+i)%N_sites].sign());}
						else {ops[i] = kroneckerProduct(B[(loc+i)%N_sites].Id(), F[(loc+i)%N_sites].Id());}
					}
					ops[range] = last[j][(loc+range)%N_sites];
					pushlist.push_back(std::make_tuple(loc, ops, factor[j] * value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};
		
		if (P.HAS("tFull"))
		{
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_sign_local = kroneckerProduct(B[loc].Id(),(F[loc].cdag(Gloc,0) * F[loc].sign()));
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > c_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				c_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].c(Gi,0));
			}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {cdag_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {c_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.) * std::sqrt(2.)}, PROP::FERMIONIC);			
		}

		if (P.HAS("Vfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {kroneckerProduct(B[loc].Id(),F[loc].Tdag(0))};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > T_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {T_ranges[i] = kroneckerProduct(B[i].Id(),F[loc].T(0));}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {T_ranges};
			push_full("Vfull", "Vᵢⱼ", first, last, {std::sqrt(3.)}, PROP::BOSONIC);			
		}

		// local terms
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung","t","tPerp",Forbitals,loc%Lcell,P.get<bool>("CYLINDER"));
		labellist[loc].push_back(tPerp.label);
		
		// V⟂
		param2d Vperp = P.fill_array2d<double>("Vrung","V","Vperp",Forbitals,loc%Lcell,P.get<bool>("CYLINDER"));
		labellist[loc].push_back(Vperp.label);
		
		// Hubbard U
		param1d U = P.fill_array1d<double>("U","Uorb",Forbitals,loc%Lcell);
		labellist[loc].push_back(U.label);
		
		if (F[loc].dim() > 1)
		{
			OperatorType KondoHamiltonian({1,1}, B[loc].get_basis().combine(F[loc].get_basis()));
			
			ArrayXXd Jperp    = B[loc].ZeroHopping();
			ArrayXXd Jperpsub = F[loc].ZeroHopping();
			
			//set Hubbard part of Kondo Hamiltonian
			KondoHamiltonian = kroneckerProduct(B[loc].Id(), F[loc].HubbardHamiltonian(U.a,tPerp.a,Vperp.a,Jperpsub));
			
			//set Heisenberg part of Hamiltonian
		//	KondoHamiltonian += OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jperp), F[loc].Id(), {1,1});
			
			// Kondo-J
			param1d J = P.fill_array1d<double>("J","Jorb",Forbitals,loc%Lcell);
			labellist[loc].push_back(J.label);
			
			//set interaction part of Hamiltonian.
			for (int alfa=0; alfa<Forbitals; ++alfa)
			{
				assert(Borbitals == Forbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
				KondoHamiltonian += J(alfa) * sqrt(3.) * SiteOperatorQ<Symmetry_,Eigen::MatrixXd>::outerprod(B[loc].Sdag(alfa), F[loc].S(alfa), {1,1});
			}
			
			pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(KondoHamiltonian), 1.));
		}
		
		// NN terms
		if (!P.HAS("tFull"))
		{
		    auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{Forbitals,F[lp1].orbitals()}},loc%Lcell);
		    labellist[loc].push_back(tlabel);
		
		    if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
			    {
					auto cdag_sign_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(Gloc,alfa) * F[loc].sign());
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_loc,kroneckerProduct(B[lp1].Id(),F[lp1].c(Glp1,beta))),-std::sqrt(2.)*std::sqrt(2.)*tPara(alfa,beta)));
			    }
			}
		}
		
		if (!P.HAS("Vfull"))
		{
		    auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{Forbitals,F[lp1].orbitals()}},loc%Lcell);
		    labellist[loc].push_back(Vlabel);

		    if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
			    {
					auto Tdag_loc = kroneckerProduct(B[loc].Id(), F[loc].Tdag(alfa));
					auto T_lp1    = kroneckerProduct(B[lp1].Id(), F[lp1].T   (beta));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Tdag_loc,T_lp1),std::sqrt(3.)*Vpara(alfa,beta)));
			    }
			}
		}
	}
}

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// make_local (KONDO_SUBSYSTEM SUBSYS, 
//             string name, 
//             size_t locx, size_t locy, 
//             const OperatorType &Op, 
//             double factor, bool FERMIONIC, bool HERMITIAN) const
// {
// 	assert(locx<F.size() and locy<F[locx].dim());
// 	assert(SUBSYS != IMPSUB);
// 	stringstream ss;
// 	ss << name << "(" << locx << "," << locy << ")";
	
// 	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
// 	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	OperatorType OpExt;
// 	vector<SiteOperator<Symmetry,MatrixType::Scalar> > SignExt(locx);
	
// 	if (SUBSYS == SUB)
// 	{
// 		OpExt   = OperatorType::outerprod(B[locx].Id(), Op, Op.Q());
// 		for (size_t l=0; l<locx; ++l)
// 		{
// 			SignExt[l] = OperatorType::outerprod(B[l].Id(), F[l].sign(), Symmetry::qvacuum()).plain<double>();
// 		}
// 	}
// 	else if (SUBSYS == IMP)
// 	{
// 		assert(!FERMIONIC and "Impurity cannot be fermionic!");
// 		OpExt = OperatorType::outerprod(Op, F[locx].Id(), Op.Q());
// 	}
	
// 	Mout.set_locality(locx);
// 	Mout.set_localOperator(OpExt.plain<double>());
	
// 	(FERMIONIC)? Mout.setLocal(locx, (factor * OpExt).plain<double>(), SignExt)
// 	           : Mout.setLocal(locx, (factor * OpExt).plain<double>());
	
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,complex<double> > KondoSU2xSU2::
// make_FourierYSum (string name, const vector<OperatorType> &Ops, 
//                   double factor, bool HERMITIAN, const vector<complex<double> > &phases) const
// {
// 	stringstream ss;
// 	ss << name << "_ky(";
// 	for (int l=0; l<phases.size(); ++l)
// 	{
// 		ss << phases[l];
// 		if (l!=phases.size()-1) {ss << ",";}
// 		else                    {ss << ")";}
// 	}
	
// 	// all Ops[l].Q() must match
// 	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,complex<double> > Mout(N_sites, Ops[0].Q(), ss.str(), HERMITIAN);
// 	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	vector<complex<double> > phases_x_factor = phases;
// 	for (int l=0; l<phases.size(); ++l)
// 	{
// 		phases_x_factor[l] = phases[l] * factor;
// 	}
	
// 	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
// 	for (int l=0; l<OpsPlain.size(); ++l)
// 	{
// 		OpsPlain[l] = Ops[l].plain<double>().cast<complex<double> >();
// 	}
	
// 	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// nh (size_t locx, size_t locy)
// {
// 	return make_local(SUB, "nh", locx,locy, F[locx].nh(locy), 1., false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// ns (size_t locx, size_t locy)
// {
// 	return make_local(SUB, "ns", locx,locy, F[locx].ns(locy), 1., false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// make_corr (KONDO_SUBSYSTEM SUBSYS,
//            string name1, string name2, 
//            size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//            const OperatorType &Op1, const OperatorType &Op2, 
//            qarray<Symmetry::Nq> Qtot, 
//            double factor,
//            bool BOTH_HERMITIAN) const
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << name1 << "(" << locx1 << "," << locy1 << ")" << name2 << "(" << locx2 << "," << locy2 << ")";
	
// 	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
// 	Mpo<Symmetry> Mout(N_sites, Qtot, ss.str(), HERMITIAN);
// 	for(size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	OperatorType Op1Ext;
// 	OperatorType Op2Ext;
	
// 	if (SUBSYS == SUB)
// 	{
// 		Op1Ext = OperatorType::outerprod(B[locx1].Id(), Op1, Op1.Q());
// 		Op2Ext = OperatorType::outerprod(B[locx2].Id(), Op2, Op2.Q());
// 	}
// 	else if (SUBSYS == IMP)
// 	{
// 		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), Op1.Q());
// 		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), Op2.Q());
// 	}
// 	else if (SUBSYS == IMPSUB)
// 	{
// 		Op2Ext = OperatorType::outerprod(Op1, F[locx1].Id(), Op1.Q());
// 		Op1Ext = OperatorType::outerprod(B[locx2].Id(), Op2, Op2.Q());
// 	}
	
// 	if (locx1 == locx2)
// 	{
// 		auto LocProd = OperatorType::prod(Op1Ext, Op2Ext, Qtot);
// 		Mout.setLocal(locx1, factor * LocProd.plain<double>());
// 	}
// 	else
// 	{
// 		Mout.setLocal({locx1, locx2}, {factor * Op1Ext.plain<double>(), Op2Ext.plain<double>()});
// 	}
	
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// SsubSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr (SUB, "Ssub","Ssub", locx1,locx2,locy1,locy2, F[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// SimpSimp (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr (IMP, "Simp","Simp", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),B[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// SimpSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr (IMPSUB, "Simp","Ssub", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// TsubTsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr (SUB, "Tsub","Tsub", locx1,locx2,locy1,locy2, F[locx1].Tdag(locy1),F[locx2].T(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
// cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
// 	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	auto cdag  = OperatorType::outerprod(B[locx1].Id(), F[locx1].cdag(locy1),{2,2});
// 	auto c     = OperatorType::outerprod(B[locx2].Id(), F[locx2].c(locy2),   {2,2});
// 	auto sign1 = OperatorType::outerprod(B[locx2].Id(), F[locx1].sign(),     {1,1});
// 	auto sign2 = OperatorType::outerprod(B[locx2].Id(), F[locx2].sign(),     {1,1});
	
// 	vector<SiteOperator<Symmetry,MatrixType::Scalar> > signs;
// 	for (size_t l=min(locx1,locx2)+1; l<max(locx1,locx2); l++)
// 	{
// 		signs.push_back(OperatorType::outerprod(B[l].Id(), F[l].sign(), {1,1}).plain<double>());
// 	}
	
// 	if (locx1 == locx2)
// 	{
// 		Mout.setLocal(locx1, sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
// 	}
// 	else if(locx1<locx2)
// 	{
// 		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(cdag, sign1, {2,2}).plain<double>(), c.plain<double>()}, signs);
// 	}
// 	else if(locx1>locx2)
// 	{
// 		Mout.setLocal({locx2, locx1}, {sqrt(2.) * OperatorType::prod(c, sign2, {2,2}).plain<double>(), cdag.plain<double>()}, signs);
// 	}
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,complex<double> > KondoSU2xSU2::
// Simp_ky (vector<complex<double> > phases) const
// {
// 	vector<OperatorType> Ops(N_sites);
// 	for (size_t l=0; l<N_sites; ++l)
// 	{
// 	  Ops[l] = OperatorType::outerprod(B[l].S(0), F[l].Id(), {3,1});	  
// 	}
// 	return make_FourierYSum("S", Ops, 1., false, phases);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,complex<double> > KondoSU2xSU2::
// Simpdag_ky (vector<complex<double> > phases, double factor) const
// {
// 	vector<OperatorType> Ops(N_sites);
// 	for (size_t l=0; l<N_sites; ++l)
// 	{
// 	  Ops[l] = OperatorType::outerprod(B[l].Sdag(0), F[l].Id(), {3,1});	  
// 	}
// 	return make_FourierYSum("S†", Ops, factor, false, phases);
// }

//HamiltonianTermsXd<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > KondoSU2xSU2::
//set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
//{
//	HamiltonianTermsXd<Symmetry> Terms;
//	
//	frac S = frac(B[loc].get_D()-1,2);
//	stringstream Slabel;
//	Slabel << "S=" << print_frac_nice(S);
//	Terms.info.push_back(Slabel.str());
//	
//	auto save_label = [&Terms] (string label)
//	{
//		if (label!="") {Terms.info.push_back(label);}
//	};
//	
//	size_t lp1 = (loc+1)%F.size();
//	
//	// NN terms
//	
//	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(tlabel);
//	
//	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(Vlabel);
//	
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	for (int j=0; j<F[lp1].orbitals(); ++j)
//	{
//		if (tPara(i,j) != 0.)
//		{
////			auto Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id(), F[loc].cdag(i), {2,2}),
////			                               OperatorType::outerprod(B[loc].Id(), F[loc].sign() , {1,1}),
////			                               {2,2});
////			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.)*sqrt(2.),
////			                                 Otmp.plain<double>(),
////			                                 OperatorType::outerprod(B[loc].Id(), F[loc].c(i), {2,2}).plain<double>()));
//		}
//	}
//	
//	// local terms
//	
//	// t⟂
//	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(tPerplabel);
//	
//	// V⟂
//	auto [Vrung,Vperp,Vperplabel] = P.fill_array2d<double>("Vrung","V","Vperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(Vperplabel);
//	
//	// J⟂
//	auto [Jrung,Jperp,Jperplabel] = P.fill_array2d<double>("Jrung","J","Jperp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(Jperplabel);
//	
//	// Hubbard U
//	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
//	save_label(Ulabel);
//	
//	OperatorType KondoHamiltonian({1,0}, B[loc].get_structured_basis().combine(F[loc].get_basis()));
//	
//	//set Hubbard part of Kondo Hamiltonian
//	KondoHamiltonian = OperatorType::outerprod(B[loc].Id(), F[loc].HubbardHamiltonian(Uorb,tPerp,Vperp,Jperp), {1,1});
//	
//	//set Heisenberg part of Hamiltonian
////	KondoHamiltonian += OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jperp), F[loc].Id(), {1,1});
//	
//	// Kondo-J
//	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
//	save_label(Jlabel);
//	
//	//set interaction part of Hamiltonian.
//	
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	{
//		KondoHamiltonian += Jorb(i)*sqrt(3.) * OperatorType::outerprod(B[loc].Sdag(i), F[loc].S(i), {1,1});
//	}
//	
//	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
//	
//	Terms.name = "Kondo SU(2)⊗SU(2)";
//	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
//	
//	return Terms;
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//Simp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
//{
//	assert(locx < this->N_sites);
//	std::stringstream ss;
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
//	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
//	
//	auto Sop = OperatorType::outerprod(B[locx].Scomp(Sa,locy).structured(), F[locx].Id(), {1});
//	
//	Mout.setLocal(locx, Sop.plain<double>());
//	return Mout;
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy) const
//{
//	assert(locx < this->N_sites);
//	std::stringstream ss;
//	
//	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
//	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
//	
//	auto Sop = OperatorType::outerprod(B[locx].Id().structured(), F[locx].Scomp(Sa,locy), {1});
//	
//	Mout.setLocal(locx, Sop.plain<double>());
//	return Mout;
//}

///*Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::*/
///*SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const*/
///*{*/
///*	assert(locx1 < this->N_sites and locx2 < this->N_sites);*/
///*	std::stringstream ss;*/
///*	*/
///*	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());*/
///*	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }*/
///*	*/
///*	auto Sop1 = OperatorType::outerprod(B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Id(), {1});*/
///*	auto Sop2 = OperatorType::outerprod(B[locx1].Id().structured(), F[locx2].Scomp(SOP2,locy2), {1});*/
///*	*/
///*	Mout.setLocal({locx1,locx2}, {Sop1.plain<double>(),Sop2.plain<double>()});*/
///*	return Mout;*/
///*}*/

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, 
//           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//           const OperatorType &Op1, const OperatorType &Op2, 
//           bool BOTH_HERMITIAN) const
//{
//	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << name1 << "(" << locx1 << "," << locy1 << ")"
//	   << name2 << "(" << locx2 << "," << locy2 << ")";
//	
//	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
//	
//	OperatorType Op1Ext;
//	OperatorType Op2Ext;
//	
//	Mpo<Symmetry> Mout(F.size(), Symmetry::qvacuum(), ss.str(), HERMITIAN);
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l);}
//	
//	if (SUBSYS == SUB)
//	{
//		Op1Ext = OperatorType::outerprod(B[locx1].Id().structured(), Op1, {1});
//		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
//	}
//	else if (SUBSYS == IMP)
//	{
//		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
//		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), {1});
//	}
//	else if (SUBSYS == IMPSUB and locx1 != locx2)
//	{
//		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
//		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
//	}
//	else if (SUBSYS == IMPSUB and locx1 == locx2)
//	{
//		OperatorType OpExt = OperatorType::outerprod(Op1, Op2, {1});
//		
//		Mout.setLocal(locx1, OpExt.plain<double>());
//		return Mout;
//	}
//	
//	Mout.setLocal({locx1,locx2}, {Op1Ext.plain<double>(),Op2Ext.plain<double>()});
//	return Mout;
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	stringstream ss1; ss1 << SOP1 << "imp";
//	stringstream ss2; ss2 << SOP2 << "sub";
//	
//	return make_corr(IMPSUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Scomp(SOP2,locy2));
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	stringstream ss1; ss1 << SOP1 << "imp";
//	stringstream ss2; ss2 << SOP2 << "imp";
//	
//	return make_corr(IMP, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), B[locx2].Scomp(SOP2,locy2).structured());
//}

//Mpo<Sym::SU2<Sym::ChargeSU2> > KondoSU2xSU2::
//SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	stringstream ss1; ss1 << SOP1 << "sub";
//	stringstream ss2; ss2 << SOP2 << "sub";
//	
//	return make_corr(SUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
//}

} //end namespace VMPS

#endif
