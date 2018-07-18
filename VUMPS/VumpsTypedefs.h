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

#endif
