#ifndef VUMPSTYPEDEFS
#define VUMPSTYPEDEFS

/**Gauge of the UMPS tensor: \p L (left gauge), \p R (right gauge), or \p C (no gauge).*/
struct GAUGE
{
	enum OPTION {L=0, R=1, C=2};
};

std::ostream& operator<< (std::ostream& s, GAUGE::OPTION g)
{
	if      (g==GAUGE::OPTION::L) {s << "L";}
	else if (g==GAUGE::OPTION::R) {s << "R";}
	else if (g==GAUGE::OPTION::C) {s << "C";}
	return s;
}

struct UMPS_ALG
{
	enum OPTION {PARALLEL=0, SEQUENTIAL=1, H2SITE=2, IDMRG=3, DYNAMIC=4};
};

std::ostream& operator<< (std::ostream& s, UMPS_ALG::OPTION a)
{
	if      (a==UMPS_ALG::OPTION::PARALLEL)   {s << "parallel";}
	else if (a==UMPS_ALG::OPTION::SEQUENTIAL) {s << "sequential";}
	else if (a==UMPS_ALG::OPTION::H2SITE)     {s << "h2site";}
	else if (a==UMPS_ALG::OPTION::IDMRG)      {s << "IDMRG";}
	else if (a==UMPS_ALG::OPTION::DYNAMIC)    {s << "dynamic(par/seq)";}
	return s;
}

/**Namespace imitation for various enums.*/
struct VUMPS
{
	struct TWOSITE_A
	{
		enum OPTION
		{
			ALxAC=0,
			ACxAR=1,
			ALxCxAR=2
		};
	};
	/**Default configuration values for the VUMPS solver.*/
	struct CONTROL
	{
		struct DEFAULT
		{
			//GLOB DEFAULTS
			constexpr static size_t min_iterations             = 1;
			constexpr static size_t max_iterations             = 1000;
			constexpr static size_t min_iter_without_expansion = 10;
			constexpr static size_t max_iter_without_expansion = 100;
			constexpr static double tol_eigval                 = 1e-13;
			constexpr static double tol_state                  = 1e-7;
			constexpr static double tol_var                    = 1e-13;
			constexpr static size_t Minit                      = 10;
			constexpr static size_t Mlimit                     = 2000;
			constexpr static size_t Qinit                      = 4;
			constexpr static size_t savePeriod                 = 0;
			constexpr static size_t truncatePeriod             = std::numeric_limits<size_t>::max();
			constexpr static bool   INIT_TO_HALF_INTEGER_QN    = false;
			constexpr static char   saveName[]                 = "UmpsBackup";
			constexpr static bool FULLMMAX_FILENAME            = true;
			constexpr static bool CALC_S_ON_EXIT               = true;
			constexpr static bool Niter_before_save            = 100;
			
			//DYN DEFAULTS
			static size_t max_deltaM          (size_t i) {return (i<1800)? 100ul:0ul;} // Maximum expansion by 100 and turn off expansion completely after 1800 iterations
			static size_t Mincr_abs           (size_t i) {return 50ul;} // increase M by at least Mincr_abs
			static double Mincr_rel           (size_t i) {return 1.04;} // increase M by at least 4%
			static void   doSomething         (size_t i) {return;}
			static UMPS_ALG::OPTION iteration (size_t i) {return UMPS_ALG::PARALLEL;}
			
			//LANCZOS DEFAULTS
			constexpr static ::LANCZOS::REORTHO::OPTION REORTHO           = LANCZOS::REORTHO::FULL;
			constexpr static double eps_eigval                            = 1.e-14;
			constexpr static double eps_coeff                             = 1.e-14;
			constexpr static size_t dimK                                  = 200ul;
		};
		
		struct GLOB
		{
			size_t min_iterations             = CONTROL::DEFAULT::min_iterations;
			size_t max_iterations             = CONTROL::DEFAULT::max_iterations;
			size_t min_iter_without_expansion = CONTROL::DEFAULT::min_iter_without_expansion;
			size_t max_iter_without_expansion = CONTROL::DEFAULT::max_iter_without_expansion;
			double tol_eigval                 = CONTROL::DEFAULT::tol_eigval;
			double tol_var                    = CONTROL::DEFAULT::tol_var;
			double tol_state                  = CONTROL::DEFAULT::tol_state;
			size_t Minit                      = CONTROL::DEFAULT::Minit;
			size_t Mlimit                     = CONTROL::DEFAULT::Mlimit;
			size_t Qinit                      = CONTROL::DEFAULT::Qinit;
			size_t savePeriod                 = CONTROL::DEFAULT::savePeriod;
			size_t truncatePeriod             = CONTROL::DEFAULT::truncatePeriod;
			bool   INIT_TO_HALF_INTEGER_QN    = CONTROL::DEFAULT::INIT_TO_HALF_INTEGER_QN;
			std::string saveName              = std::string(CONTROL::DEFAULT::saveName);
			bool FULLMMAX_FILENAME            = CONTROL::DEFAULT::FULLMMAX_FILENAME;
			bool CALC_S_ON_EXIT               = CONTROL::DEFAULT::CALC_S_ON_EXIT;
			int Niter_before_save             = CONTROL::DEFAULT::Niter_before_save;
		};
		
		struct DYN
		{
			function<size_t(size_t)> max_deltaM          = CONTROL::DEFAULT::max_deltaM;
			function<size_t(size_t)> Mincr_abs           = CONTROL::DEFAULT::Mincr_abs;
			function<double(size_t)> Mincr_rel           = CONTROL::DEFAULT::Mincr_rel;
			// function<double(size_t)> eps_svd             = CONTROL::DEFAULT::eps_svd;
			function<void(size_t)>   doSomething         = CONTROL::DEFAULT::doSomething;
			function<size_t(size_t)> iteration           = CONTROL::DEFAULT::iteration;
		};
		
		struct LOC
		{
			::LANCZOS::REORTHO::OPTION REORTHO   = CONTROL::DEFAULT::REORTHO;
			double eps_eigval                    = CONTROL::DEFAULT::eps_eigval;
			double eps_coeff                     = CONTROL::DEFAULT::eps_coeff;
			size_t dimK                          = CONTROL::DEFAULT::dimK;
		};
	};
};

std::ostream& operator<< (std::ostream& s, VUMPS::TWOSITE_A::OPTION a)
{
	if      (a==VUMPS::TWOSITE_A::ALxAC)   {s << "ALxAC";}
	else if (a==VUMPS::TWOSITE_A::ACxAR)   {s << "ACxAR";}
	else if (a==VUMPS::TWOSITE_A::ALxCxAR) {s << "ALxCxAR";}
	return s;
}

#endif
