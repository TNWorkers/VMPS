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

//Eigen::ArrayXXd coupling2d (double coupl_x, double coupl_y, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	Eigen::ArrayXXd out(Lx*Ly,Lx*Ly); out.setZero();
//	for (int ix=0; ix<Lx; ix++)
//	for (int iy=0; iy<Ly; iy++)
//	for (int jx=0; jx<Lx; jx++)
//	for (int jy=0; jy<Ly; jy++)
//	{
//		if (abs(ix-jx) == 1 and (iy==jy))
//		{
//			out(ix*Ly+iy, jx*Ly+jy) += coupl_x;
//		}
//		else if (abs(iy-jy) == 1 and (ix==jx))
//		{
//			out(ix*Ly+iy, jx*Ly+jy) += coupl_y;
//		}
//		else if (abs(iy-jy) == Ly-1 and (ix==jx) and PERIODIC_Y == true)
//		{
//			out(ix*Ly+iy, jx*Ly+jy) += coupl_y;
//		}
//	}
//	return out;
//}

//Eigen::ArrayXXd coupling2d (double coupl, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	return coupling2d (coupl,coupl,Lx,Ly,PERIODIC_Y);
//}

//Eigen::ArrayXXd coupling2d_snake (double coupl_x, double coupl_y, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	Eigen::ArrayXXd out(Lx*Ly,Lx*Ly); out.setZero();
//	
//	// Mirrors the y coordinate to create a snake.
//	auto mirror = [&Ly] (int iy) -> int
//	{
//		vector<int> v(Ly);
//		iota (begin(v),end(v),0);
//		reverse(v.begin(),v.end());
//		return v[iy];
//	};
//	
//	for (int ix=0; ix<Lx; ix++)
//	for (int iy=0; iy<Ly; iy++)
//	for (int jx=0; jx<Lx; jx++)
//	for (int jy=0; jy<Ly; jy++)
//	{
//		// mirror even y only
//		int iy_ = (ix%2==0)? iy : mirror(iy);
//		int jy_ = (jx%2==0)? jy : mirror(jy);
//		int index_i = iy_+Ly*ix;
//		int index_j = jy_+Ly*jx;
//		
////		if (jx==0 and jy==0)
////		{
////			cout << ix << ", " << iy << " -> index=" << index_i << endl;
////		}
//		
//		if (abs(ix-jx) == 1 and (iy==jy))
//		{
//			out(index_i,index_j) += coupl_x;
//		}
//		else if (abs(iy-jy) == 1 and (ix==jx))
//		{
//			out(index_i,index_j) += coupl_y;
//		}
//		else if (abs(iy-jy) == Ly-1 and (ix==jx) and PERIODIC_Y == true)
//		{
//			out(index_i,index_j) += coupl_y;
//		}
//	}
//	return out;
//}

//Eigen::ArrayXXd coupling2d_snake (double coupl, size_t Lx, size_t Ly, bool PERIODIC_Y=false)
//{
//	return coupling2d_snake (coupl,coupl,Lx,Ly,PERIODIC_Y);
//}

//// standard deviation for a hopping matrix
//double sigma_hop (const Eigen::ArrayXXd &thop)
//{
//	double res = 0.;
//	
//	Eigen::ArrayXd x(thop.rows()); x = 0;
//	
//	for (int i=0; i<thop.rows(); ++i)
//	for (int j=0; j<thop.cols(); ++j)
//	{
//		if (thop(i,j) != 0.) x(i) += abs(j-i);
//	}
//	
//	double avg = x.sum()/x.rows();
//	double var = ((x-avg)*(x-avg)).sum();
//	
//	return sqrt(var);
//}

#endif
