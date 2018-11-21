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
			ALxAC,
			ACxAR,
			ALxCxAR
		};
	};
	/**Default configuration values for the VUMPS solver.*/
	struct CONTROL
	{
		struct DEFAULT
		{
			//GLOB DEFAULTS
			constexpr static size_t min_iterations             = 1;
			constexpr static size_t max_iterations             = 300;
			constexpr static size_t min_iter_without_expansion = 10;
			constexpr static size_t max_iter_without_expansion = 50;
			constexpr static double tol_eigval                 = 1e-8;
			constexpr static double tol_state                  = 1e-7;
			constexpr static double tol_var                    = 1e-7;
			constexpr static size_t Dinit                      = 4;
			constexpr static size_t Dlimit                     = 500;
			constexpr static size_t Qinit                      = 4;
			constexpr static size_t savePeriod                 = 0;

			//DYN DEFAULTS
			static size_t max_deltaD          (size_t i) {return (i<200)? 100ul:0ul;} // Maximum expansion by 100 and turn off expansion completely after 200 iterations
			static size_t Dincr_abs           (size_t i) {return 2ul;} // increase D by at least Dincr_abs
			static double Dincr_rel           (size_t i) {return 1.04;} // increase D by at least 4%
			static size_t Dincr_per           (size_t i) {return 10ul;} // increase D every 10 iterations
			static void   doSomething         (size_t i) {return;}
			static UMPS_ALG::OPTION iteration (size_t i) {return UMPS_ALG::PARALLEL;}
			
			//LANCZOS DEFAULTS
			constexpr static ::LANCZOS::REORTHO::OPTION REORTHO           = LANCZOS::REORTHO::FULL;
			constexpr static double eps_eigval                            = 1.e-7;
			constexpr static double eps_coeff                             = 1.e-4;
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
			size_t Dinit                      = CONTROL::DEFAULT::Dinit;
			size_t Dlimit                     = CONTROL::DEFAULT::Dlimit;
			size_t Qinit                      = CONTROL::DEFAULT::Qinit;
			size_t savePeriod                 = CONTROL::DEFAULT::savePeriod;
		};
		
		struct DYN
		{
			function<size_t(size_t)> max_deltaD          = CONTROL::DEFAULT::max_deltaD;
			function<size_t(size_t)> Dincr_abs           = CONTROL::DEFAULT::Dincr_abs;
			function<double(size_t)> Dincr_rel           = CONTROL::DEFAULT::Dincr_rel;
			function<size_t(size_t)> Dincr_per           = CONTROL::DEFAULT::Dincr_per;
			// function<double(size_t)> eps_svd             = CONTROL::DEFAULT::eps_svd;
			function<void(size_t)>   doSomething         = CONTROL::DEFAULT::doSomething;
			function<size_t(size_t)> iteration           = CONTROL::DEFAULT::iteration;
		};
		
		struct LANCZOS
		{
			::LANCZOS::REORTHO::OPTION REORTHO   = CONTROL::DEFAULT::REORTHO;
			double eps_eigval                    = CONTROL::DEFAULT::eps_eigval;
			double eps_coeff                     = CONTROL::DEFAULT::eps_coeff;
			size_t dimK                          = CONTROL::DEFAULT::dimK;
		};
	};
};

#endif
