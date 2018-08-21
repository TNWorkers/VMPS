#ifndef PARAMRETURNER
#define PARAMRETURNER

#include "ParamHandler.h"
#include "DmrgTypedefs.h"

class ParamReturner
{
public:
	
	ParamReturner() : TRIVIALLY_CONSTRUCTED(true) {}
	
	ParamReturner (const std::map<string,std::any> &defaults_input)
	:defaults(defaults_input)
	{}
	
	///@{
	/**Push params for DMRG algorithms via these functions to an instance of DmrgSolver.*/
	DMRG::CONTROL::DYN  get_DynParam  (const vector<Param> &params={}) const;
	DMRG::CONTROL::GLOB get_GlobParam (const vector<Param> &params={}) const;
	///@}
	
private:
	
	bool TRIVIALLY_CONSTRUCTED = false;
	std::map<string,std::any> defaults;
};

DMRG::CONTROL::GLOB ParamReturner::
get_GlobParam (const vector<Param> &params) const
{
	DMRG::CONTROL::GLOB out;
	if(TRIVIALLY_CONSTRUCTED) {return out;} //Return defaults from DmrgTypedefs
	
	ParamHandler P(params,defaults);
	out.min_halfsweeps = P.get<size_t>("min_halfsweeps");
	out.max_halfsweeps = P.get<size_t>("max_halfsweeps");
	out.Dinit          = P.get<size_t>("Dinit");
	out.Qinit          = P.get<size_t>("Qinit");
	out.Dlimit         = P.get<size_t>("Dlimit");
	out.tol_eigval     = P.get<double>("tol_eigval");
	out.tol_state      = P.get<double>("tol_state");
	out.savePeriod     = P.get<size_t>("savePeriod");
	out.CONVTEST       = P.get<DMRG::CONVTEST::OPTION>("CONVTEST");
	out.CALC_S_ON_EXIT = P.get<bool>("CALC_S_ON_EXIT");
	return out;
}

DMRG::CONTROL::DYN ParamReturner::
get_DynParam (const vector<Param> &params) const
{
//	ParamHandler P(params,Heisenberg::sweep_defaults);
//	DMRG::CONTROL::DYN out;
//	out.max_alpha_rsvd = [&P] (size_t i) {return P.get<double>("max_alpha");};
//	out.min_alpha_rsvd = [&P] (size_t i) {return P.get<double>("min_alpha");};
//	out.eps_svd        = [&P] (size_t i) {return P.get<double>("eps_svd");};
//	out.Dincr_abs      = [&P] (size_t i) {return P.get<size_t>("Dincr_abs");};
//	out.Dincr_per      = [&P] (size_t i) {return P.get<size_t>("Dincr_per");};
//	out.Dincr_rel      = [&P] (size_t i) {return P.get<double>("Dincr_rel");};
//	out.min_Nsv        = [&P] (size_t i) {return P.get<size_t>("min_Nsv");};
//	out.max_Nrich      = [&P] (size_t i) {return P.get<int>   ("max_Nrich");};
//	return out;
	
	DMRG::CONTROL::DYN out;
	if(TRIVIALLY_CONSTRUCTED) {return out;} //Return defaults from DmrgTypedefs
	
	ParamHandler P(params,defaults);
	
	double tmp1        = P.get<double>("max_alpha");
	out.max_alpha_rsvd = [tmp1] (size_t i) {return tmp1;};
	tmp1               = P.get<double>("min_alpha");
	out.min_alpha_rsvd = [tmp1] (size_t i) {return tmp1;};
	tmp1               = P.get<double>("eps_svd");
	out.eps_svd        = [tmp1] (size_t i) {return tmp1;};
	size_t tmp2        = P.get<size_t>("Dincr_abs");
	out.Dincr_abs      = [tmp2] (size_t i) {return tmp2;};
	tmp2               = P.get<size_t>("Dincr_per");
	out.Dincr_per      = [tmp2] (size_t i) {return tmp2;};
	tmp1               = P.get<double>("Dincr_rel");
	out.Dincr_rel      = [tmp1] (size_t i) {return tmp1;};
	tmp2               = P.get<size_t>("min_Nsv");
	out.min_Nsv        = [tmp2] (size_t i) {return tmp2;};
	int tmp3           = P.get<int>("max_Nrich");
	out.max_Nrich      = [tmp3] (size_t i) {return tmp3;};
	return out;
}

#endif
