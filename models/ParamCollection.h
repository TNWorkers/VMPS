#ifndef PARAMCOLLECTION
#define PARAMCOLLECTION

#include "ParamHandler.h"

void push_back_KondoUnpacked (vector<Param> &params, size_t L, double J, double t, size_t D, bool START_WITH_SPIN=true)
{
	int SPIN_PARITY = (START_WITH_SPIN==true)? 0:1;
	for (size_t l=0; l<2*L; ++l)
	{
		// spin site
		if (l%2 == SPIN_PARITY)
		{
			params.push_back({"D",D,l});
			params.push_back({"LyF",0ul,l});
			params.push_back({"Iprev",0.,l});
			(START_WITH_SPIN==true)? params.push_back({"Inext",J,l}) : params.push_back({"Inext",0.,l});
			params.push_back({"tPrime",0.,l});
		}
		// fermionic site
		else
		{
			params.push_back({"D",1ul,l});
			params.push_back({"LyF",1ul,l});
			params.push_back({"Inext",0.,l});
			(START_WITH_SPIN==false)? params.push_back({"Iprev",J,l}) : params.push_back({"Iprev",0.,l});
			params.push_back({"tPrime",t,l});
		}
	}
}

#endif
