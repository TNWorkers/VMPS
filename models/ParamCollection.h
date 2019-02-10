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
			if (START_WITH_SPIN==true)
			{
				params.push_back({"Inext",J,l});
				params.push_back({"Iprev",0.,l});
			}
			else
			{
				params.push_back({"Inext",0.,l});
				params.push_back({"Iprev",J,l});
			}
			params.push_back({"tPrime",0.,l});
		}
		// fermionic site
		else
		{
			params.push_back({"D",1ul,l});
			params.push_back({"LyF",1ul,l});
			params.push_back({"tPrime",t,l});
			params.push_back({"Inext",0.,l});
			params.push_back({"Iprev",0.,l});
		}
	}
}

Eigen::ArrayXXd coupling2d (double coupl_x, double coupl_y, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
{
	Eigen::ArrayXXd out(Lx*Ly,Lx*Ly); out.setZero();
	for (int ix=0; ix<Lx; ix++)
	for (int iy=0; iy<Ly; iy++)
	for (int jx=0; jx<Lx; jx++)
	for (int jy=0; jy<Ly; jy++)
	{
		if (abs(ix-jx) == 1 and (iy==jy))
		{
			out(ix*Ly+iy, jx*Ly+jy) += coupl_x;
		}
		else if (abs(iy-jy) == 1 and (ix==jx))
		{
			out(ix*Ly+iy, jx*Ly+jy) += coupl_y;
		}
		else if (abs(iy-jy) == Ly-1 and (ix==jx) and PERIODIC_Y == true)
		{
			out(ix*Ly+iy, jx*Ly+jy) += coupl_y;
		}
	}
	return out;
}

Eigen::ArrayXXd coupling2d (double coupl, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
{
	return coupling2d (coupl,coupl,Lx,Ly,PERIODIC_Y);
}
#endif
