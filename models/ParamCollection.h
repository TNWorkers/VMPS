#ifndef PARAMCOLLECTION
#define PARAMCOLLECTION

#include "ParamHandler.h"

void push_back_KondoUnpacked (vector<Param> &params, size_t L, double J, double t, size_t D)
{
	for (size_t l=0; l<2*L; ++l)
	{
		if (l%2 == 0)
		{
			params.push_back({"D",D,l});
			params.push_back({"LyF",0ul,l});
			params.push_back({"Inext",J,l});
			params.push_back({"tPrime",0.,l});
		}
		else
		{
			params.push_back({"D",1ul,l});
			params.push_back({"LyF",1ul,l});
			params.push_back({"Inext",0.,l});
			params.push_back({"tPrime",t,l});
		}
	}
}

#endif
