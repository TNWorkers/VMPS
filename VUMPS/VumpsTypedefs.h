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
	/**Default configuration values for the VUMPS solver.*/
	struct CONTROL
	{
		struct DEFAULT
		{
			//GLOB DEFAULTS
			constexpr static size_t min_iterations = 6;
			constexpr static size_t max_iterations = 20;
			constexpr static double tol_eigval     = 1e-6;
			constexpr static double tol_var        = 1e-5;
			constexpr static size_t Dinit          = 20;
			constexpr static size_t Dlimit         = 500;
			constexpr static size_t Qinit          = 20;
			constexpr static size_t savePeriod     = 0;

			//DYN DEFAULTS
			static size_t deltaD (size_t i)
				{
					size_t out;
					if      (i<=80)            { out=5ul; }
					else if (i> 80 and i<=120) { out=0ul; }
					else if (i>120 and i<=160) { out=5ul; }
					else if (i>160 and i<=200) { out=0ul; }
					else if (i>200 and i<=240) { out=4ul; }
					else if (i>240 and i<=280) { out=0ul; }
					else if (i>280 and i<=300) { out=2ul; }
					else if (i>300)            { out=10ul; }    
					return out;
				}
			static double errLimit_for_flucts (size_t i) {return 1.e-1;}
			static double eps_svd        (size_t i)      {return 1.e-7;}
			static void   doSomething    (size_t i)      {return;}
			
			//LANCZOS DEFAULTS
			constexpr static ::LANCZOS::REORTHO::OPTION REORTHO           = LANCZOS::REORTHO::FULL;
			constexpr static double eps_eigval                            = 1.e-7;
			constexpr static double eps_coeff                             = 1.e-4;
			constexpr static size_t dimK                                  = 200ul;
		};
		
		struct GLOB
		{
			size_t min_iterations           = CONTROL::DEFAULT::min_iterations;
			size_t max_iterations           = CONTROL::DEFAULT::max_iterations;
			double tol_eigval               = CONTROL::DEFAULT::tol_eigval;
			double tol_cvar                 = CONTROL::DEFAULT::tol_var;
			size_t Dinit                    = CONTROL::DEFAULT::Dinit;
			size_t Dlimit                   = CONTROL::DEFAULT::Dlimit;
			size_t Qinit                    = CONTROL::DEFAULT::Qinit;
			size_t savePeriod               = CONTROL::DEFAULT::savePeriod;
		};
		
		struct DYN
		{
			function<size_t(size_t)> deltaD              = CONTROL::DEFAULT::deltaD;
			function<double(size_t)> errLimit_for_flucts = CONTROL::DEFAULT::errLimit_for_flucts;
			function<double(size_t)> eps_svd             = CONTROL::DEFAULT::eps_svd;
			function<void(size_t)>   doSomething         = CONTROL::DEFAULT::doSomething;
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
