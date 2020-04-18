#ifndef PARAMCOLLECTION
#define PARAMCOLLECTION

#include "ParamHandler.h"

// Simple generation of periodic boundary conditions with NN and NNN coupling
ArrayXXd create_1D_PBC (size_t L, double lambda1=1., double lambda2=0.)
{
	ArrayXXd res(L,L); res.setZero();
	
	res.matrix().diagonal<1>().setConstant(lambda1);
	res.matrix().diagonal<-1>().setConstant(lambda1);
	res(0,L-1) = lambda1;
	res(L-1,0) = lambda1;
	
	res.matrix().diagonal<2>().setConstant(lambda2);
	res.matrix().diagonal<-2>().setConstant(lambda2);
	res(0,L-2) = lambda2;
	res(L-2,0) = lambda2;
	res(1,L-1) = lambda2;
	res(L-1,1) = lambda2;
	
	return res;
}

ArrayXXd create_1D_OBC (size_t L, double lambda1=1., double lambda2=0.)
{
	ArrayXXd res(L,L); res.setZero();
	
	res.matrix().diagonal<1>().setConstant(lambda1);
	res.matrix().diagonal<-1>().setConstant(lambda1);
	res(0,L-1) = lambda1;
	res(L-1,0) = lambda1;
	
	res.matrix().diagonal<2>().setConstant(lambda2);
	res.matrix().diagonal<-2>().setConstant(lambda2);
	
	return res;
}

// reference: PRB 93, 165406 (2016), Appendix C
ArrayXXd hopping_fullerene (int L=60, double t=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L== 60)
	{
		res(0,4) = t;
		res(3,4) = t;
		res(2,3) = t;
		res(1,2) = t;
		res(0,1) = t;
		
		res(0,8) = t;
		res(4,5) = t;
		res(3,17) = t;
		res(2,14) = t;
		res(1,11) = t;
		
		res(7,8) = t;
		res(6,7) = t;
		res(5,6) = t;
		res(5,19) = t;
		res(18,19) = t;
		res(17,18) = t;
		res(16,17) = t;
		res(15,16) = t;
		res(14,15) = t;
		res(13,14) = t;
		res(12,13) = t;
		res(11,12) = t;
		res(10,11) = t;
		res(9,10) = t;
		res(8,9) = t;
		
		res(7,22) = t;
		res(6,21) = t;
		res(19,20) = t;
		res(18,29) = t;
		res(16,28) = t;
		res(15,27) = t;
		res(13,26) = t;
		res(12,25) = t;
		res(10,24) = t;
		res(9,23) = t;
		
		res(22,33) = t;
		res(32,33) = t;
		res(21,32) = t;
		res(20,21) = t;
		res(20,31) = t;
		res(30,31) = t;
		res(29,30) = t;
		res(28,29) = t;
		res(28,39) = t;
		res(38,39) = t;
		res(27,38) = t;
		res(26,27) = t;
		res(26,37) = t;
		res(36,37) = t;
		res(25,36) = t;
		res(24,25) = t;
		res(24,35) = t;
		res(34,35) = t;
		res(23,34) = t;
		res(22,23) = t;
		
		res(33,46) = t;
		res(32,44) = t;
		res(31,43) = t;
		res(30,41) = t;
		res(39,40) = t;
		res(38,53) = t;
		res(37,52) = t;
		res(36,50) = t;
		res(35,49) = t;
		res(34,47) = t;
		
		res(45,46) = t;
		res(44,45) = t;
		res(43,44) = t;
		res(42,43) = t;
		res(41,42) = t;
		res(40,41) = t;
		res(40,54) = t;
		res(53,54) = t;
		res(52,53) = t;
		res(51,52) = t;
		res(50,51) = t;
		res(49,50) = t;
		res(48,49) = t;
		res(47,48) = t;
		res(46,47) = t;
		
		res(45,57) = t;
		res(42,56) = t;
		res(54,55) = t;
		res(51,59) = t;
		res(48,58) = t;
		res(56,57) = t;
		res(55,56) = t;
		res(55,59) = t;
		res(58,59) = t;
		res(57,58) = t;
	}
	// reference: https://www.qmul.ac.uk/sbcs/iupac/fullerene2/311.html
	// also in: Phys. Rev. B 72, 064453 (2005)
	else if (L == 20)
	{
//		res(11,12) = t;
//		res(12,13) = t;
//		res(13,14) = t;
//		res(5,14) = t;
//		res(5,6) = t;
//		res(6,7) = t;
//		res(7,8) = t;
//		res(8,9) = t;
//		res(9,10) = t;
//		res(10,11) = t;
//		
//		res(0,1) = t;
//		res(1,2) = t;
//		res(2,3) = t;
//		res(3,4) = t;
//		res(0,4) = t;
//		
//		res(2,11) = t;
//		res(3,13) = t;
//		res(4,5) = t;
//		res(0,7) = t;
//		res(1,9) = t;
//		res(2,11) = t;
//		
//		res(15,16) = t;
//		res(16,17) = t;
//		res(17,18) = t;
//		res(18,19) = t;
//		res(15,19) = t;
//		
//		res(12,19) = t;
//		res(14,15) = t;
//		res(6,16) = t;
//		res(8,17) = t;
//		res(10,18) = t;
		
		// better numbering (inwards spiral):
		
		res(0,1) = t;
		res(1,2) = t;
		res(2,3) = t;
		res(3,4) = t;
		res(0,4) = t;
		
		res(0,7) = t;
		res(1,9) = t;
		res(2,11) = t;
		res(3,13) = t;
		res(4,5) = t;
		
		res(5,6) = t;
		res(6,7) = t;
		res(7,8) = t;
		res(8,9) = t;
		res(9,10) = t;
		res(10,11) = t;
		res(11,12) = t;
		res(12,13) = t;
		res(13,14) = t;
		res(5,14) = t;
		
		res(6,16) = t;
		res(8,17) = t;
		res(10,18) = t;
		res(12,19) = t;
		res(14,15) = t;
		
		res(15,16) = t;
		res(16,17) = t;
		res(17,18) = t;
		res(18,19) = t;
		res(15,19) = t;
	}
	// reference: Phys. Rev. B 72, 064453 (2005)
	else if (L == 12)
	{
//		res(0,1) = t;
//		res(1,2) = t;
//		res(0,2) = t;
//		
//		res(1,5) = t;
//		res(2,5) = t;
//		res(2,6) = t;
//		res(2,3) = t;
//		res(0,3) = t;
//		res(0,7) = t;
//		res(0,4) = t;
//		res(1,4) = t;
//		res(1,8) = t;
//		
//		res(4,8) = t;
//		res(5,8) = t;
//		res(5,6) = t;
//		res(3,6) = t;
//		res(3,7) = t;
//		res(4,7) = t;
//		
//		res(8,10) = t;
//		res(5,10) = t;
//		res(6,10) = t;
//		res(6,11) = t;
//		res(3,11) = t;
//		res(7,11) = t;
//		res(7,9) = t;
//		res(4,9) = t;
//		res(9,8) = t;
//		
//		res(9,10) = t;
//		res(10,11) = t;
//		res(9,11) = t;
		
		// better numbering (inwards spiral):
		
		res(0,1) = t;
		res(1,2) = t;
		res(0,2) = t;
		
		res(0,3) = t;
		res(0,4) = t;
		res(0,5) = t;
		
		res(1,5) = t;
		res(1,6) = t;
		res(1,7) = t;
		
		res(2,3) = t;
		res(2,7) = t;
		res(2,8) = t;
		
		res(3,4) = t;
		res(4,5) = t;
		res(5,6) = t;
		res(6,7) = t;
		res(7,8) = t;
		res(3,8) = t;
		
		res(3,9) = t;
		res(4,9) = t;
		res(4,10) = t;
		res(5,10) = t;
		res(6,10) = t;
		res(6,11) = t;
		res(7,11) = t;
		res(8,11) = t;
		res(8,9) = t;
		
		res(9,10) = t;
		res(10,11) = t;
		res(9,11) = t;
	}
	
	res += res.transpose().eval();
	
	return res;
}

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
