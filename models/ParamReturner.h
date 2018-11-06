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
	DMRG::CONTROL::DYN  get_DmrgDynParam  (const vector<Param> &params={}) const;
	DMRG::CONTROL::GLOB get_DmrgGlobParam (const vector<Param> &params={}) const;
	///@}

	///@{
	/**Push params for DMRG algorithms via these functions to an instance of DmrgSolver.*/
	VUMPS::CONTROL::DYN  get_VumpsDynParam  (const vector<Param> &params={}) const;
	VUMPS::CONTROL::GLOB get_VumpsGlobParam (const vector<Param> &params={}) const;
	///@}

private:
	
	bool TRIVIALLY_CONSTRUCTED = false;
	std::map<string,std::any> defaults;
};

DMRG::CONTROL::GLOB ParamReturner::
get_DmrgGlobParam (const vector<Param> &params) const
{
	DMRG::CONTROL::GLOB out;
	if (TRIVIALLY_CONSTRUCTED) {return out;} //Return defaults from DmrgTypedefs
	
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
get_DmrgDynParam (const vector<Param> &params) const
{
	DMRG::CONTROL::DYN out;
	if (TRIVIALLY_CONSTRUCTED) {return out;} //Return defaults from DmrgTypedefs
	
	ParamHandler P(params,defaults);
	
	double tmp1        = P.get<double>("max_alpha");
	size_t lim         = P.get<size_t>("lim_alpha");
	out.max_alpha_rsvd = [tmp1,lim] (size_t i) {return (i<lim)? tmp1:0.;};
	
	tmp1               = P.get<double>("min_alpha");
	out.min_alpha_rsvd = [tmp1,lim] (size_t i) {return (i<lim)? tmp1:0.;};
	
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
	
//	out.iteration = P.get<function<DMRG::ITERATION::OPTION(size_t)> >("iteration");
	
	return out;
}

VUMPS::CONTROL::GLOB ParamReturner::
get_VumpsGlobParam (const vector<Param> &params) const
{
	VUMPS::CONTROL::GLOB out; return out; //For now return the default values in every case. Change this if the parameters are in the models.
	if (TRIVIALLY_CONSTRUCTED) {return out;} //Return defaults from VumpsTypedefs

	ParamHandler P(params,defaults);
	out.min_iterations = P.get<size_t>("min_iterations");
	out.max_iterations = P.get<size_t>("max_iterations");
	out.Dinit          = P.get<size_t>("Dinit");
	out.Qinit          = P.get<size_t>("Qinit");
	out.Dlimit         = P.get<size_t>("Dlimit");
	out.tol_eigval     = P.get<double>("tol_eigval");
	out.tol_var        = P.get<double>("tol_var");
	out.tol_state      = P.get<double>("tol_state");
	out.savePeriod     = P.get<size_t>("savePeriod");
	return out;
}

VUMPS::CONTROL::DYN ParamReturner::
get_VumpsDynParam (const vector<Param> &params) const
{
	VUMPS::CONTROL::DYN out; return out; //For now return the default values in every case. Change this if the parameters are in the models.
	if (TRIVIALLY_CONSTRUCTED) {return out;} //Return defaults from VumpsTypedefs
	
	ParamHandler P(params,defaults);
	
	size_t tmp1        = P.get<double>("max_deltaD");
	size_t lim         = P.get<size_t>("lim_deltaD");
	out.max_deltaD     = [tmp1,lim] (size_t i) {return (i<lim)? tmp1:0.;};
		
	size_t tmp2        = P.get<size_t>("Dincr_abs");
	out.Dincr_abs      = [tmp2] (size_t i) {return tmp2;};
	
	tmp2               = P.get<size_t>("Dincr_per");
	out.Dincr_per      = [tmp2] (size_t i) {return tmp2;};
	
	tmp1               = P.get<double>("Dincr_rel");
	out.Dincr_rel      = [tmp1] (size_t i) {return tmp1;};	
	
	return out;
}

#endif
