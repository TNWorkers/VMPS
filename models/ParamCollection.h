#ifndef PARAMCOLLECTION
#define PARAMCOLLECTION

#include <set>

#include <Eigen/Dense>
using namespace Eigen;

#include "ParamHandler.h"
#include "CuthillMcKeeCompressor.h" // from ALGS
#include <boost/rational.hpp>
#include "termcolor.hpp"
#include "DmrgTypedefs.h" // for SUB_LATTICE

ArrayXXd create_1D_OBC (size_t L, double lambda1=1., double lambda2=0.)
{
	ArrayXXd res(L,L); res.setZero();
	
	res.matrix().diagonal<1>().setConstant(lambda1);
	res.matrix().diagonal<-1>().setConstant(lambda1);
	
	res.matrix().diagonal<2>().setConstant(lambda2);
	res.matrix().diagonal<-2>().setConstant(lambda2);
	
	return res;
}

// Simple generation of periodic boundary conditions with NN and NNN coupling
// If COMPRESSED=true, the ring is flattened so that adjacent sites are as close together as possible
ArrayXXd create_1D_PBC (size_t L, double lambda1=1., double lambda2=0., bool COMPRESSED=false)
{
	ArrayXXd res(L,L);
	res.setZero();
	
	res = create_1D_OBC(L,lambda1,lambda2);
	
	res(0,L-1) = lambda1;
	res(L-1,0) = lambda1;
	
	res(0,L-2) = lambda2;
	res(L-2,0) = lambda2;
	res(1,L-1) = lambda2;
	res(L-1,1) = lambda2;
	
	if (COMPRESSED and lambda2 == 0.)
	{
		res.setZero();
		
		res(0,1) = lambda1; res(1,0) = lambda1;
		res(L-2,L-1) = lambda1; res(L-1,L-2) = lambda1;
		for (size_t l=0; l<L-2; l++)
		{
			res(l,l+2) = lambda1;
			res(l+2,l) = lambda1;
		}
	}
	else if (COMPRESSED and lambda2 != 0.)
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	return res;
}

ArrayXXd extend_to_thermal (const ArrayXXd &tFull, double factor)
{
	int L = tFull.rows();
	ArrayXXd res(2*L,2*L); res.setZero();
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		if (abs(tFull(i,j)) > 1e-10)
		{
			res(2*i,2*j) = tFull(i,j);
			res(2*i+1,2*j+1) = factor*tFull(i,j);
		}
	}
	return res;
}

ArrayXXd create_1D_AB (size_t L, double lambda1A=1., double lambda1B=1., double lambda2A=0., double lambda2B=0.)
{
	ArrayXXd res(L,L);
	res.setZero();
	
	for (int i=0; i<L; i+=2)
	{
		if (i+1<L) res(i,  i+1) = lambda1A;
		if (i+2<L) res(i+1,i+2) = lambda1B;
	}
	
	for (int i=0; i<L; i+=2)
	{
		if (i+2<L) res(i  ,i+2) = lambda2A;
		if (i+3<L) res(i+1,i+3) = lambda2B;
	}
	
	res += res.transpose().eval();
	
	return res;
}

ArrayXXd create_1D_PBC_AB (size_t L, double lambda1A=1., double lambda1B=1., double lambda2A=0., double lambda2B=0., bool COMPRESSED=true)
{
	ArrayXXd res(L,L);
	res.setZero();
	
	for (int i=0; i<L; i+=2)
	{
		res(i,  (i+1)%L) = lambda1A;
		if (i+1<L) res(i+1,(i+2)%L) = lambda1B;
	}
	
	for (int i=0; i<L; i+=2)
	{
		res(i,  (i+2)%L) = lambda2A;
		if (i+1<L) res(i+1,(i+3)%L) = lambda2B;
	}
	
	res += res.transpose().eval();
	
//	cout << "res=" << endl << res << endl;
	
	if (COMPRESSED)
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	
	return res;
}

ArrayXXd hopping_square (int Lx, int Ly, bool PBCx=false, bool PBCy=true, double lambda=1.)
{
	ArrayXXd res(Lx*Ly,Lx*Ly); res.setZero();
	// vertical
	for (int x=0; x<Lx; ++x)
	{
		int i0 = Ly*x;
		for (int i=i0; i<=i0+Ly-2; ++i)
		{
			res(i,i+1) = lambda;
		}
		// y-periodic part for all x:
		if (PBCy) res(i0,i0+Ly-1) = lambda;
	}
	// horizontal
	for (int y=0; y<Ly; ++y)
	{
		for (int x=0; x<Lx-1; ++x)
		{
			int i = Ly*x+y;
			res(i,i+Ly) = lambda;
		}
	}
	// x-periodic part for all y:
	if (PBCx)
	{
		for (int i=0; i<Ly; ++i)
		{
			res(i,Ly*Lx-Ly+i) = lambda;
		}
	}
	res += res.transpose().eval();
	return res;
}

// split into batches of Ly for A, and Ly/2 for B (depleted sites)
void split_kagomeYC_AB (int Lx, int Ly, std::vector<std::vector<int>> &A, std::vector<std::vector<int>> &B)
{
	int Nevn = Lx/2;
	int Nodd = Lx/2;
	int L = Ly*Nevn+Ly/2*Nodd;
	
	vector<int> input(L);
	for (int j=0; j<L; ++j) input[j] = j;
	
	int i = 0;
	
	while (i+Ly <= L)
	{
		std::vector<int> batchA;
		std::vector<int> batchB;
		
		for (int j=0; j<Ly; j++)
		{
			batchA.push_back(input[i+j]);
		}
		
		A.push_back(batchA);
		i += Ly;
		
		if (i + Ly/2 <= L)
		{
			for (int j=0; j<Ly/2; j++)
			{
				batchB.push_back(input[i+j]);
			}
			B.push_back(batchB);
			i += Ly/2;
		}
	}
}

// split into batches of Ly/2 for A (depleted sites), and Ly for B
void split_kagomeYC_BAB (int Lx, int Ly, std::vector<std::vector<int>> &A, std::vector<std::vector<int>> &B)
{
	int Nevn = Lx/2+1;
	int Nodd = Lx/2;
	int L = Ly/2*Nevn+Ly*Nodd;
	
	std::vector<int> input(L);
	for (int j=0; j<L; ++j) input[j] = j;
	
	int i = 0;
	
	while (i + Ly/2 <= L)
	{
		std::vector<int> batchA;
		
		for (int j=0; j<Ly/2; j++)
		{
			batchA.push_back(input[i+j]);
		}
		
		A.push_back(batchA);
		i += Ly/2;
		
		if (i + Ly <= L)
		{
			std::vector<int> batchB;
			
			for (int j=0; j<Ly; j++)
			{
				batchB.push_back(input[i+j]);
			}
			
			B.push_back(batchB);
			i += Ly;
		}
	}
}

int find_x_kagomeYC (int index, int L, const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B)
{
	//int Nevn = Lx/2;
	//int Nodd = Lx/2;
	//int L = Ly*Nevn+Ly/2*Nodd;
	
	MatrixXi res(L,3);
	
	bool EVN = false;
	bool ODD = false;
	int i0 = -1;
	
	for (int i=0; i<A.size(); ++i)
	{
		auto it = find(A[i].begin(), A[i].end(), index);
		if (it != A[i].end())
		{
			EVN = true;
			i0 = i;
			break;
		}
	}
	
	if (!EVN)
	{
		for (int i=0; i<B.size(); ++i)
		{
			auto it = find(B[i].begin(), B[i].end(), index);
			if (it != B[i].end())
			{
				ODD = true;
				i0 = i;
				break;
			}
		}
	}
	
	assert(EVN or ODD);
	
	return (EVN)? 2*i0 : 2*i0+1;
}

ArrayXXd hopping_kagomeYC_AB (int Lx, int Ly, bool PBCx=false, bool PBCy=true, double lambda=1.)
{
	assert(PBCx==false and PBCy==true);
	
	int Nevn = Lx/2;
	int Nodd = Lx/2;
	int L = Ly*Nevn+Ly/2*Nodd;
	
	ArrayXXd res(L,L); res.setZero();
	
	vector<vector<int>> i_evn;
	vector<vector<int>> i_odd;
	
	split_kagomeYC_BAB(Lx, Ly, i_evn, i_odd);
	
//	lout << "EVN:" << endl;
//	
//	for (int i=0; i<i_evn.size(); ++i)
//	{
//		for (int j=0; j<i_evn[i].size(); ++j)
//		{
//			lout << i_evn[i][j] << endl;
//		}
//		lout << "----" << endl;
//	}
//	
//	lout << "ODD:" << endl;
//	
//	for (int i=0; i<i_odd.size(); ++i)
//	{
//		for (int j=0; j<i_odd[i].size(); ++j)
//		{
//			lout << i_odd[i][j] << endl;
//		}
//		lout << "----" << endl;
//	}
	
	// vertical
	for (int x=0; x<i_evn.size(); ++x)
	for (int i=0; i<i_evn[x].size(); ++i)
	{
		int k = i_evn[x][i];
		int l = i_evn[x][(i+1)%Ly];
		//lout << "vertical evn bond: " << k << ", " << l << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// horizontal, even x
	for (int x=0; x<i_evn.size(); ++x)
	for (int i=1; i<i_evn[x].size(); i+=2)
	{
		int k = i_evn[x][i];
		int l = i_odd[x][(i-1)/2];
		//lout << "horizontal evn bond: " << k << ", " << l << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// horizontal, odd x
	for (int x=0; x<i_odd.size()-1; ++x)
	for (int i=0; i<i_odd[x].size(); i+=1)
	{
		int k = i_odd[x][i];
		int l = i_evn[x+1][2*i+1];
		//lout << "horizontal odd bond: " << k << ", " << l << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// diagonal, even x
	for (int x=0; x<i_evn.size(); ++x)
	for (int i=0; i<i_evn[x].size(); i+=2)
	{
		int k = i_evn[x][i];
		int l = i_odd[x][i/2];
		//lout << "diagonal evn bond: " << k << ", " << l << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// diagonal, odd x
	for (int x=0; x<i_odd.size()-1; ++x)
	for (int i=0; i<i_odd[x].size(); ++i)
	{
		int k = i_odd[x][i];
		int l = i_evn[x+1][(2*i+2)%Ly];
		//lout << "diagonal odd bond: " << k << ", " << l << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	res += res.transpose().eval();
	
	//lout << res << endl;
	
	return res;
}

ArrayXXd hopping_kagomeYC_BAB (int Lx, int Ly, bool PBCx=false, bool PBCy=true, double lambda=1., bool VERBOSE=false)
{
	assert(PBCx==false and PBCy==true);
	
	int Nevn = Lx/2+1;
	int Nodd = Lx/2;
	int L = Ly/2*Nevn+Ly*Nodd;
	
	ArrayXXd res(L,L); res.setZero();
	
	vector<vector<int>> i_evn;
	vector<vector<int>> i_odd;
	
	split_kagomeYC_BAB(Lx, Ly, i_evn, i_odd);
	
	if (VERBOSE)
	{
		lout << "EVN:" << endl;
		for (int i=0; i<i_evn.size(); ++i)
		{
			for (int j=0; j<i_evn[i].size(); ++j)
			{
				lout << i_evn[i][j] << endl;
			}
			lout << "----" << endl;
		}
		
		lout << "ODD:" << endl;
		for (int i=0; i<i_odd.size(); ++i)
		{
			for (int j=0; j<i_odd[i].size(); ++j)
			{
				lout << i_odd[i][j] << endl;
			}
			lout << "----" << endl;
		}
	}
	
	// vertical, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=0; i<i_odd[x].size(); ++i)
	{
		int k = i_odd[x][i];
		int l = i_odd[x][(i+1)%Ly];
		if (VERBOSE) lout << "vertical odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// horizontal, evn x
	for (int x=0; x<i_evn.size()-1; ++x)
	for (int i=0; i<i_evn[x].size(); i+=1)
	{
		int k = i_evn[x][i];
		int l = i_odd[x][2*i+1];
		if (VERBOSE) lout << "horizontal evn bond: " << min(k,l) << ", " << max(k,l) << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// horizontal, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=1; i<i_odd[x].size(); i+=2)
	{
		int k = i_odd[x][i];
		int l = i_evn[x+1][(i-1)/2];
		if (VERBOSE) lout << "horizontal odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// diagonal, evn x
	for (int x=0; x<i_evn.size()-1; ++x)
	for (int i=0; i<i_evn[x].size(); ++i)
	{
		int k = i_evn[x][i];
		int l = i_odd[x][(2*i+2)%Ly];
		if (VERBOSE) lout << "diagonal evn bond: " << min(k,l) << ", " << max(k,l) << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	// diagonal, odd x
	for (int x=0; x<i_odd.size(); ++x)
	for (int i=0; i<i_odd[x].size(); i+=2)
	{
		int k = i_odd[x][i];
		int l = i_evn[x+1][i/2];
		if (VERBOSE) lout << "diagonal odd bond: " << min(k,l) << ", " << max(k,l) << endl;
		res(min(k,l),max(k,l)) = lambda;
	}
	
	res += res.transpose().eval();
	
	//lout << res << endl;
	
	return res;
}

void split_kagomeXC (int Lx, int Ly, std::vector<std::vector<int>> &A, std::vector<std::vector<int>> &B, std::vector<std::vector<int>> &C, std::vector<std::vector<int>> &D)
{
	int N = Ly/4;
	int L = N*Lx + N*(Lx/2+1) + N*Lx + N*Lx/2;
	
	std::vector<int> input(L);
	for (int j=0; j<L; ++j) input[j] = j;
	
	int currentIndex = 0;
    while(currentIndex < L)
    {
        // Batch of Lx into A
        vector<int> subA;
        for(int i=currentIndex; i<currentIndex+Lx && i<L; i++) {
            subA.push_back(input[i]);
        }
        A.push_back(subA);
        currentIndex += Lx;
        if(currentIndex >= L) break;

        // Batch of Lx/2+1 into B
        vector<int> subB;
        for(int i=currentIndex; i<currentIndex+Lx/2+1 && i<L; i++) {
            subB.push_back(input[i]);
        }
        B.push_back(subB);
        currentIndex += Lx/2+1;
        if(currentIndex >= L) break;

        // Another batch of Lx into C
        vector<int> subC;
        for(int i=currentIndex; i<currentIndex+Lx && i<L; i++) {
            subC.push_back(input[i]);
        }
        C.push_back(subC);
        currentIndex += Lx;
        if(currentIndex >= L) break;

        // Batch of Lx/2 into D
        vector<int> subD;
        for(int i=currentIndex; i<currentIndex+Lx/2 && i<L; i++) {
            subD.push_back(input[i]);
        }
        D.push_back(subD);
        currentIndex += Lx/2;
    }
    
//    for (int i=0; i<Ly/4; ++i)
//	{
//		lout << "A:" << endl;
//		for (int j=0; j<iA[i].size(); ++j)
//		{
//			lout << iA[i][j] << ", ";
//		}
//		lout << endl;
//		lout << "B:" << endl;
//		for (int j=0; j<iB[i].size(); ++j)
//		{
//			lout << iB[i][j] << ", ";
//		}
//		lout << endl;
//		lout << "C:" << endl;
//		for (int j=0; j<iC[i].size(); ++j)
//		{
//			lout << iC[i][j] << ", ";
//		}
//		lout << endl;
//		lout << "D:" << endl;
//		for (int j=0; j<iD[i].size(); ++j)
//		{
//			lout << iD[i][j] << ", ";
//		}
//		lout << endl;
//	}
}

ArrayXXd hopping_kagomeXC (int Lx, int Ly, bool PBCx=false, bool PBCy=true, double lambda=1., bool VERBOSE=false)
{
	int N = Ly/4;
	int L = N*Lx + N*(Lx/2+1) + N*Lx + N*Lx/2;
	
	ArrayXXd res(L,L); res.setZero();
	
	vector<vector<int>> iA, iB, iC, iD;
	split_kagomeXC(Lx, Ly, iA, iB, iC, iD);
	
	for (int y=0; y<N; ++y)
	{
		if (VERBOSE) lout << "y=" << y << endl;
		
		// horizontal bonds
		for (int i=0; i<iA[y].size()-1; ++i)
		{
			int k = iA[y][i];
			int l = iA[y][i+1];
			res(min(k,l),max(k,l)) = lambda;
			if (VERBOSE) lout << "A-A: " << min(k,l) << ", " << max(k,l) << endl;
		}
		for (int i=0; i<iC[y].size()-1; ++i)
		{
			int k = iC[y][i];
			int l = iC[y][i+1];
			res(min(k,l),max(k,l)) = lambda;
			if (VERBOSE) lout << "C-C: " << min(k,l) << ", " << max(k,l) << endl;
		}
		
		// vertical bonds
		for (int i=0; i<iA[y].size(); ++i)
		{
			if (i==0)
			{
				int k = iA[y][0];
				int l = iB[y][0];
				res(min(k,l),max(k,l)) = lambda;
				if (VERBOSE) lout << "A-B first: " << min(k,l) << ", " << max(k,l) << endl;
			}
			else if (i==Lx-1)
			{
				int k = iA[y][Lx-1];
				int l = iB[y][Lx/2];
				res(min(k,l),max(k,l)) = lambda;
				if (VERBOSE) lout << "A-B last: " << min(k,l) << ", " << max(k,l) << endl;
			}
			else
			{
				if (i%2==0)
				{
					int k = iA[y][i];
					int l = iB[y][i/2];
					res(min(k,l),max(k,l)) = lambda;
					if (VERBOSE) lout << "A-B: " << min(k,l) << ", " << max(k,l) << endl;
				}
				else
				{
					int k = iA[y][i];
					int l = iB[y][(i+1)/2];
					res(min(k,l),max(k,l)) = lambda;
					if (VERBOSE) lout << "A-B: " << min(k,l) << ", " << max(k,l) << endl;
				}
			}
		}
		
		for (int i=0; i<iB[y].size(); ++i)
		{
			if (i==0)
			{
				int k = iB[y][0];
				int l = iC[y][0];
				res(min(k,l),max(k,l)) = lambda;
				if (VERBOSE) lout << "B-C first: " << min(k,l) << ", " << max(k,l) << endl;
			}
			else if (i==Lx/2)
			{
				int k = iB[y][Lx/2];
				int l = iC[y][Lx-1];
				res(min(k,l),max(k,l)) = lambda;
				if (VERBOSE) lout << "B-C last: " << min(k,l) << ", " << max(k,l) << endl;
			}
			else
			{
				if (i%2==1)
				{
					int k = iB[y][i];
					int l = iC[y][2*i];
					res(min(k,l),max(k,l)) = lambda;
					if (VERBOSE) lout << "B-C: " << min(k,l) << ", " << max(k,l) << endl;
					
					k = iB[y][i];
					l = iC[y][2*i-1];
					res(min(k,l),max(k,l)) = lambda;
					if (VERBOSE) lout << "B-C: " << min(k,l) << ", " << max(k,l) << endl;
				}
				else
				{
					int k = iB[y][i];
					int l = iC[y][2*i];
					res(min(k,l),max(k,l)) = lambda;
					if (VERBOSE) lout << "B-C: " << min(k,l) << ", " << max(k,l) << endl;
					
					k = iB[y][i];
					l = iC[y][2*i-1];
					res(min(k,l),max(k,l)) = lambda;
					if (VERBOSE) lout << "B-C: " << min(k,l) << ", " << max(k,l) << endl;
				}
			}
		}
		
		for (int i=0; i<iD[y].size(); ++i)
		{
			int k = iC[y][2*i];
			int l = iD[y][i];
			res(min(k,l),max(k,l)) = lambda;
			if (VERBOSE) lout << "C-D: " << min(k,l) << ", " << max(k,l) << endl;
			
			k = iC[y][2*i+1];
			l = iD[y][i];
			res(min(k,l),max(k,l)) = lambda;
			if (VERBOSE) lout << "C-D: " << min(k,l) << ", " << max(k,l) << endl;
		}
		
		// includes periodic y-bonds
		for (int i=0; i<iD[y].size(); ++i)
		{
			int k = iA[(y+1)%iA.size()][2*i];
			int l = iD[y][i];
			res(min(k,l),max(k,l)) = lambda;
			if (VERBOSE) lout << "D-A: " << min(k,l) << ", " << max(k,l) << endl;
			
			k = iA[(y+1)%iA.size()][2*i+1];
			l = iD[y][i];
			res(min(k,l),max(k,l)) = lambda;
			if (VERBOSE) lout << "D-A: " << min(k,l) << ", " << max(k,l) << endl;
		}
	}
	
	res += res.transpose().eval();
	return res;
}

ArrayXXd hopping_triangularYC (int Lx, int Ly, bool PBCx=false, bool PBCy=true, double lambda=1.)
{
	ArrayXXd res = hopping_square(Lx,Ly,PBCx,PBCy,lambda);
	res.matrix().triangularView<Eigen::Upper>().setZero();
	// diagonal
	for (int x=0; x<Lx-1; ++x)
	{
		for (int y=1; y<Ly; ++y)
		{
			int i = y+Ly*x;
			res(i,i+Ly-1) = lambda;
		}
		// y-periodic part for all x:
		if (PBCy) res(Ly*x,Ly*x+2*Ly-1) = lambda;
	}
	// x-periodic part for all y:
	if (PBCx)
	{
		for (int i=0; i<Ly-1; ++i)
		{
			res(i,Ly*Lx-Ly+i+1) = lambda;
		}
		res(Ly-1,Ly*Lx-Ly) = lambda;
	}
	res += res.transpose().eval();
	return res;
}

void add_triangle (int i, int j, int k, ArrayXXd &target, double lambda=1.)
{
	assert(i<j and j<k);
	target(i,j) = lambda;
	target(j,k) = lambda;
	target(i,k) = lambda;
}

// Leung, Elser PRB 47, 9 (1992)
ArrayXXd hopping_kagome36d (double lambda=1.)
{
	ArrayXXd res(36,36);
	res.setZero();
	
	add_triangle(0,3,4,res,lambda);
	add_triangle(1,5,6,res,lambda);
	add_triangle(2,3,7,res,lambda);
	add_triangle(4,5,8,res,lambda);
	add_triangle(6,9,21,res,lambda);
	add_triangle(7,11,12,res,lambda);
	add_triangle(8,13,14,res,lambda);
	add_triangle(9,15,16,res,lambda);
	add_triangle(10,11,17,res,lambda);
	add_triangle(12,13,18,res,lambda);
	add_triangle(14,15,19,res,lambda);
	add_triangle(16,20,31,res,lambda);
	add_triangle(17,21,22,res,lambda);
	add_triangle(18,23,24,res,lambda);
	add_triangle(19,25,26,res,lambda);
	add_triangle(2,20,27,res,lambda);
	add_triangle(22,23,28,res,lambda);
	add_triangle(24,25,29,res,lambda);
	add_triangle(26,27,30,res,lambda);
	add_triangle(28,31,32,res,lambda);
	add_triangle(29,33,35,res,lambda);
	add_triangle(10,30,35,res,lambda);
	add_triangle(0,32,33,res,lambda);
	add_triangle(1,34,35,res,lambda);
	
	res += res.transpose().eval();
	
	auto res_ = compress_CuthillMcKee(res,true);
	res = res_;
	
	return res;
}

ArrayXXd triangularFlake (int L, double lambda=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L == 28 or L == 36 or L == 45 or L == 55 or L == 66)
	{
		add_triangle(0,1,2,res,lambda);
		
		add_triangle(1,3,4,res,lambda);
		add_triangle(1,2,4,res,lambda);
		add_triangle(2,4,5,res,lambda);
		
		add_triangle(3,6,7,res,lambda);
		add_triangle(3,4,7,res,lambda);
		add_triangle(4,7,8,res,lambda);
		add_triangle(4,5,8,res,lambda);
		add_triangle(5,8,9,res,lambda);
		
		add_triangle(6,10,11,res,lambda);
		add_triangle(6,7,11,res,lambda);
		add_triangle(7,11,12,res,lambda);
		add_triangle(7,8,12,res,lambda);
		add_triangle(8,12,13,res,lambda);
		add_triangle(8,9,13,res,lambda);
		add_triangle(9,13,14,res,lambda);
		
		add_triangle(10,15,16,res,lambda);
		add_triangle(10,11,16,res,lambda);
		add_triangle(11,16,17,res,lambda);
		add_triangle(11,12,17,res,lambda);
		add_triangle(12,17,18,res,lambda);
		add_triangle(12,13,18,res,lambda);
		add_triangle(13,18,19,res,lambda);
		add_triangle(13,14,19,res,lambda);
		add_triangle(14,19,20,res,lambda);
		
		add_triangle(15,21,22,res,lambda);
		add_triangle(15,16,22,res,lambda);
		add_triangle(16,22,23,res,lambda);
		add_triangle(16,17,23,res,lambda);
		add_triangle(17,23,24,res,lambda);
		add_triangle(17,18,24,res,lambda);
		add_triangle(18,24,25,res,lambda);
		add_triangle(18,19,25,res,lambda);
		add_triangle(19,25,26,res,lambda);
		add_triangle(19,20,26,res,lambda);
		add_triangle(20,26,27,res,lambda);
		
		if (L >= 36)
		{
			add_triangle(21,28,29,res,lambda);
			add_triangle(21,22,29,res,lambda);
			add_triangle(22,29,30,res,lambda);
			add_triangle(22,23,30,res,lambda);
			add_triangle(23,30,31,res,lambda);
			add_triangle(23,24,31,res,lambda);
			add_triangle(24,31,32,res,lambda);
			add_triangle(24,25,32,res,lambda);
			add_triangle(25,32,33,res,lambda);
			add_triangle(25,26,33,res,lambda);
			add_triangle(26,33,34,res,lambda);
			add_triangle(26,27,34,res,lambda);
			add_triangle(27,34,35,res,lambda);
		}
		
		if (L >= 45)
		{
			add_triangle(28,36,37,res,lambda);
			add_triangle(29,37,38,res,lambda);
			add_triangle(30,38,39,res,lambda);
			add_triangle(31,39,40,res,lambda);
			add_triangle(32,40,41,res,lambda);
			add_triangle(33,41,42,res,lambda);
			add_triangle(34,42,43,res,lambda);
			add_triangle(35,43,44,res,lambda);
			
			add_triangle(28,29,37,res,lambda);
			add_triangle(29,30,38,res,lambda);
			add_triangle(30,31,39,res,lambda);
			add_triangle(31,32,40,res,lambda);
			add_triangle(32,33,41,res,lambda);
			add_triangle(33,34,42,res,lambda);
			add_triangle(34,35,43,res,lambda);
			
			if (L >= 55)
			{
				add_triangle(36,45,46,res,lambda);
				add_triangle(37,38,47,res,lambda);
				add_triangle(38,39,48,res,lambda);
				add_triangle(39,40,49,res,lambda);
				add_triangle(40,41,50,res,lambda);
				add_triangle(41,42,51,res,lambda);
				add_triangle(42,43,52,res,lambda);
				add_triangle(43,44,53,res,lambda);
				
				add_triangle(36,45,46,res,lambda);
				add_triangle(37,46,47,res,lambda);
				add_triangle(28,36,37,res,lambda);
				add_triangle(38,47,48,res,lambda);
				add_triangle(39,48,49,res,lambda);
				add_triangle(40,49,50,res,lambda);
				add_triangle(41,50,51,res,lambda);
				add_triangle(42,51,52,res,lambda);
				add_triangle(43,52,53,res,lambda);
				add_triangle(44,53,54,res,lambda);
				
				if (L >= 66)
				{
					add_triangle(45,46,56,res,lambda);
					add_triangle(46,47,57,res,lambda);
					add_triangle(47,48,58,res,lambda);
					add_triangle(48,49,59,res,lambda);
					add_triangle(49,50,60,res,lambda);
					add_triangle(50,51,61,res,lambda);
					add_triangle(51,52,62,res,lambda);
					add_triangle(52,53,63,res,lambda);
					add_triangle(53,54,64,res,lambda);
					
					add_triangle(45,55,56,res,lambda);
					add_triangle(46,56,57,res,lambda);
					add_triangle(47,57,58,res,lambda);
					add_triangle(48,58,59,res,lambda);
					add_triangle(49,59,60,res,lambda);
					add_triangle(50,60,61,res,lambda);
					add_triangle(51,61,62,res,lambda);
					add_triangle(52,62,63,res,lambda);
					add_triangle(53,63,64,res,lambda);
					add_triangle(54,64,65,res,lambda);
				}
			}
		}
	}
	
	res += res.transpose().eval();
	
	auto res_ = compress_CuthillMcKee(res,true);
	res = res_;
	
	return res;
}

ArrayXXd hexagonalFlake (int L, double lambda=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L == 48)
	{
		add_triangle(0,4,5,res,lambda);
		add_triangle(0,1,5,res,lambda);
		add_triangle(1,5,6,res,lambda);
		add_triangle(1,2,6,res,lambda);
		add_triangle(2,6,7,res,lambda);
		add_triangle(2,3,7,res,lambda);
		add_triangle(3,7,8,res,lambda);
		
		add_triangle(4,9,10,res,lambda);
		add_triangle(4,5,10,res,lambda);
		add_triangle(5,10,11,res,lambda);
		add_triangle(5,6,11,res,lambda);
		add_triangle(6,11,12,res,lambda);
		add_triangle(6,7,12,res,lambda);
		add_triangle(7,12,13,res,lambda);
		add_triangle(7,8,13,res,lambda);
		add_triangle(8,13,14,res,lambda);
		
		add_triangle(9,15,16,res,lambda);
		add_triangle(9,10,16,res,lambda);
		add_triangle(10,16,17,res,lambda);
		add_triangle(10,11,17,res,lambda);
		add_triangle(11,17,18,res,lambda);
		add_triangle(11,12,18,res,lambda);
		add_triangle(12,18,19,res,lambda);
		add_triangle(12,13,19,res,lambda);
		add_triangle(13,19,20,res,lambda);
		add_triangle(13,14,20,res,lambda);
		add_triangle(14,20,21,res,lambda);
		
		add_triangle(15,16,23,res,lambda);
		add_triangle(16,17,24,res,lambda);
		add_triangle(17,18,25,res,lambda);
		add_triangle(18,19,26,res,lambda);
		add_triangle(19,20,27,res,lambda);
		add_triangle(20,21,28,res,lambda);
		add_triangle(15,22,23,res,lambda);
		add_triangle(16,23,24,res,lambda);
		add_triangle(17,24,25,res,lambda);
		add_triangle(18,25,26,res,lambda);
		add_triangle(19,26,27,res,lambda);
		add_triangle(20,27,28,res,lambda);
		add_triangle(21,28,29,res,lambda);
		
		add_triangle(22,23,30,res,lambda);
		add_triangle(23,24,31,res,lambda);
		add_triangle(24,25,32,res,lambda);
		add_triangle(25,26,33,res,lambda);
		add_triangle(26,27,34,res,lambda);
		add_triangle(27,28,35,res,lambda);
		add_triangle(28,29,36,res,lambda);
		add_triangle(23,30,31,res,lambda);
		add_triangle(24,31,32,res,lambda);
		add_triangle(25,32,33,res,lambda);
		add_triangle(26,33,34,res,lambda);
		add_triangle(27,34,35,res,lambda);
		add_triangle(28,35,36,res,lambda);
		
		add_triangle(30,31,37,res,lambda);
		add_triangle(31,32,38,res,lambda);
		add_triangle(32,33,39,res,lambda);
		add_triangle(33,34,40,res,lambda);
		add_triangle(34,35,41,res,lambda);
		add_triangle(35,36,42,res,lambda);
		add_triangle(31,37,38,res,lambda);
		add_triangle(32,38,39,res,lambda);
		add_triangle(33,39,40,res,lambda);
		add_triangle(34,40,41,res,lambda);
		add_triangle(35,41,42,res,lambda);
		
		add_triangle(37,38,43,res,lambda);
		add_triangle(38,39,44,res,lambda);
		add_triangle(39,40,45,res,lambda);
		add_triangle(40,41,46,res,lambda);
		add_triangle(41,42,47,res,lambda);
		add_triangle(38,43,44,res,lambda);
		add_triangle(39,44,45,res,lambda);
		add_triangle(40,45,46,res,lambda);
		add_triangle(41,46,47,res,lambda);
	}
	
	res += res.transpose().eval();
	
	auto res_ = compress_CuthillMcKee(res,true);
	res = res_;
	
	return res;
}

ArrayXXd hopping_Archimedean (string vertex_conf, int VARIANT=0, double lambda1=1., double lambda2=1.)
{
	ArrayXXd res;
	
	if (vertex_conf == "3.5.3.5") // icosidodecahedron
	{
		int L=30;
		res.resize(L,L); res.setZero();
		
		if (VARIANT==1 or VARIANT==0) // my naive counting
		{
			res(0,1) = lambda1;
			res(1,2) = lambda1;
			res(2,3) = lambda1;
			res(3,4) = lambda1;
			res(0,4) = lambda1;
			
			res(0,5) = lambda1;
			res(0,6) = lambda1;
			res(1,6) = lambda1;
			res(1,7) = lambda1;
			res(2,7) = lambda1;
			res(2,8) = lambda1;
			res(3,8) = lambda1;
			res(3,9) = lambda1;
			res(4,5) = lambda1;
			res(4,9) = lambda1;
			
			res(5,11) = lambda1;
			res(5,12) = lambda1;
			res(6,13) = lambda1;
			res(6,14) = lambda1;
			res(7,15) = lambda1;
			res(7,16) = lambda1;
			res(8,17) = lambda1;
			res(8,18) = lambda1;
			res(9,10) = lambda1;
			res(9,19) = lambda1;
			
			res(12,13) = lambda1;
			res(13,14) = lambda1;
			res(14,15) = lambda1;
			res(15,16) = lambda1;
			res(16,17) = lambda1;
			res(17,18) = lambda1;
			res(18,19) = lambda1;
			res(10,19) = lambda1;
			res(10,11) = lambda1;
			res(11,12) = lambda1;
			
			res(12,22) = lambda1;
			res(13,22) = lambda1;
			res(14,23) = lambda1;
			res(15,23) = lambda1;
			res(16,24) = lambda1;
			res(17,24) = lambda1;
			res(18,20) = lambda1;
			res(19,20) = lambda1;
			res(10,21) = lambda1;
			res(11,21) = lambda1;
			
			res(20,28) = lambda1;
			res(20,29) = lambda1;
			res(21,27) = lambda1;
			res(21,28) = lambda1;
			res(22,26) = lambda1;
			res(22,27) = lambda1;
			res(23,25) = lambda1;
			res(23,26) = lambda1;
			res(24,25) = lambda1;
			res(24,29) = lambda1;
			
			res(25,26) = lambda1;
			res(26,27) = lambda1;
			res(27,28) = lambda1;
			res(28,29) = lambda1;
			res(25,29) = lambda1;
		}
		else if (VARIANT==2) // According to Exler, Schnack (2003)
		{
			res(1-1,3-1) = lambda1;
			res(1-1,4-1) = lambda1;
			res(1-1,28-1) = lambda1;
			res(1-1,29-1) = lambda1;
			
			res(2-1,5-1) = lambda1;
			res(2-1,8-1) = lambda1;
			res(2-1,26-1) = lambda1;
			res(2-1,29-1) = lambda1;
			
			res(3-1,4-1) = lambda1;
			res(3-1,6-1) = lambda1;
			res(3-1,30-1) = lambda1;
			
			res(4-1,5-1) = lambda1;
			res(4-1,7-1) = lambda1;
			
			res(5-1,7-1) = lambda1;
			res(5-1,8-1) = lambda1;
			
			res(6-1,30-1) = lambda1;
			res(6-1,9-1) = lambda1;
			res(6-1,12-1) = lambda1;
			
			res(7-1,9-1) = lambda1;
			res(7-1,10-1) = lambda1;
			
			res(8-1,11-1) = lambda1;
			res(8-1,14-1) = lambda1;
			
			res(9-1,10-1) = lambda1;
			res(9-1,12-1) = lambda1;
			
			res(10-1,11-1) = lambda1;
			res(10-1,13-1) = lambda1;
			
			res(11-1,13-1) = lambda1;
			res(11-1,14-1) = lambda1;
			
			res(12-1,15-1) = lambda1;
			res(12-1,18-1) = lambda1;
			
			res(13-1,15-1) = lambda1;
			res(13-1,16-1) = lambda1;
			
			res(14-1,17-1) = lambda1;
			res(14-1,20-1) = lambda1;
			
			res(15-1,16-1) = lambda1;
			res(15-1,18-1) = lambda1;
			
			res(16-1,17-1) = lambda1;
			res(16-1,19-1) = lambda1;
			
			res(17-1,19-1) = lambda1;
			res(17-1,20-1) = lambda1;
			
			res(18-1,21-1) = lambda1;
			res(18-1,24-1) = lambda1;
			
			res(19-1,21-1) = lambda1;
			res(19-1,22-1) = lambda1;
			
			res(20-1,23-1) = lambda1;
			res(20-1,26-1) = lambda1;
			
			res(21-1,22-1) = lambda1;
			res(21-1,24-1) = lambda1;
			
			res(22-1,23-1) = lambda1;
			res(22-1,25-1) = lambda1;
			
			res(23-1,25-1) = lambda1;
			res(23-1,26-1) = lambda1;
			
			res(24-1,27-1) = lambda1;
			res(24-1,30-1) = lambda1;
			
			res(25-1,27-1) = lambda1;
			res(25-1,28-1) = lambda1;
			
			res(26-1,29-1) = lambda1;
			
			res(27-1,28-1) = lambda1;
			res(27-1,30-1) = lambda1;
			
			res(28-1,29-1) = lambda1;
		}
	}
	else if (vertex_conf == "3^4.5") // snub dodecahedron
	{
		int L=60;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=58; ++i) res(i,i+1) = lambda1;
		
		res( 0, 4) = lambda1;
		res( 0, 6) = lambda1;
		res( 0, 7) = lambda1;
		res( 0, 8) = lambda1;
		
		res( 1, 8) = lambda1;
		res( 1, 9) = lambda1;
		res( 1,10) = lambda1;
		
		res( 2,10) = lambda1;
		res( 2,11) = lambda1;
		res( 2,12) = lambda1;
		
		res( 3,12) = lambda1;
		res( 3,13) = lambda1;
		res( 3,14) = lambda1;
		
		res( 4, 6) = lambda1;
		res( 4,14) = lambda1;
		
		res( 5,14) = lambda1;
		res( 5,15) = lambda1;
		res( 5,16) = lambda1;
		
		res( 6,18) = lambda1;
		
		res( 7,18) = lambda1;
		res( 7,19) = lambda1;
		
		res( 8,21) = lambda1;
		
		res( 9,21) = lambda1;
		res( 9,22) = lambda1;
		
		res(10,24) = lambda1;
		
		res(11,24) = lambda1;
		res(11,25) = lambda1;
		
		res(12,27) = lambda1;
		
		res(13,27) = lambda1;
		res(13,28) = lambda1;
		
		res(15,29) = lambda1;
		res(15,30) = lambda1;
		
		res(16,30) = lambda1;
		res(16,31) = lambda1;
		
		res(17,31) = lambda1;
		res(17,32) = lambda1;
		res(17,33) = lambda1;
		
		res(18,33) = lambda1;
		
		res(19,33) = lambda1;
		res(19,34) = lambda1;
		
		res(20,34) = lambda1;
		res(20,35) = lambda1;
		res(20,36) = lambda1;
		
		res(21,36) = lambda1;
		
		res(22,36) = lambda1;
		res(22,37) = lambda1;
		
		res(23,37) = lambda1;
		res(23,38) = lambda1;
		res(23,39) = lambda1;
		
		res(24,39) = lambda1;
		
		res(25,39) = lambda1;
		res(25,40) = lambda1;
		
		res(26,40) = lambda1;
		res(26,41) = lambda1;
		res(26,42) = lambda1;
		
		res(27,42) = lambda1;
		
		res(28,42) = lambda1;
		res(28,43) = lambda1;
		
		res(29,43) = lambda1;
		res(29,44) = lambda1;
		
		res(30,44) = lambda1;
		
		res(31,46) = lambda1;
		
		res(32,46) = lambda1;
		res(32,47) = lambda1;
		
		res(34,48) = lambda1;
		
		res(35,48) = lambda1;
		res(35,49) = lambda1;
		
		res(37,50) = lambda1;
		
		res(38,50) = lambda1;
		res(38,51) = lambda1;
		
		res(40,52) = lambda1;
		
		res(41,52) = lambda1;
		res(41,53) = lambda1;
		
		res(43,54) = lambda1;
		
		res(44,54) = lambda1;
		
		res(45,54) = lambda1;
		res(45,55) = lambda1;
		res(45,56) = lambda1;
		
		res(46,56) = lambda1;
		
		res(47,56) = lambda1;
		res(47,57) = lambda1;
		
		res(48,57) = lambda1;
		
		res(49,57) = lambda1;
		res(49,58) = lambda1;
		
		res(50,58) = lambda1;
		
		res(51,58) = lambda1;
		res(51,59) = lambda1;
		
		res(52,59) = lambda1;
		
		res(53,55) = lambda1;
		res(53,59) = lambda1;
		
		res(55,59) = lambda1;
	}
	else if (vertex_conf == "4.6^2") // truncated octahedron
	{
		int L=24;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=4; ++i) res(i,i+1) = lambda1;
		res(0,5) = lambda1;
		
		res(5,6) = lambda1;
		res(0,9) = lambda1;
		res(1,10) = lambda1;
		res(2,13) = lambda1;
		res(3,14) = lambda1;
		res(4,17) = lambda1;
		
		for (int i=6; i<=16; ++i) res(i,i+1) = lambda1;
		res(6,17) = lambda1;
		
		res(7,19) = lambda1;
		res(8,20) = lambda1;
		res(11,21) = lambda1;
		res(12,22) = lambda1;
		res(15,23) = lambda1;
		res(16,18) = lambda1;
		
		for (int i=18; i<=22; ++i) res(i,i+1) = lambda1;
		res(18,23) = lambda1;
	}
	else if (vertex_conf == "3.8^2") // truncated cube
	{
		int L=24;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=6; ++i) res(i,i+1) = lambda1;
		res(0,7) = lambda1;
		
		res(0,8) = lambda1;
		res(7,8) = lambda1;
		res(1,9) = lambda1;
		res(2,9) = lambda1;
		res(3,10) = lambda1;
		res(4,10) = lambda1;
		res(5,11) = lambda1;
		res(6,11) = lambda1;
		
		res(8,13) = lambda1;
		res(9,14) = lambda1;
		res(10,15) = lambda1;
		res(11,12) = lambda1;
		
		res(12,17) = lambda1;
		res(12,18) = lambda1;
		res(13,19) = lambda1;
		res(13,20) = lambda1;
		res(14,21) = lambda1;
		res(14,22) = lambda1;
		res(15,16) = lambda1;
		res(15,23) = lambda1;
		
		for (int i=15; i<=22; ++i) res(i,i+1) = lambda1;
		res(15,23) = lambda1;
		
		res(19,20) = lambda1;
		res(16,23) = lambda1;
	}
	else if (vertex_conf == "3.4.3.4") // cuboctahedron
	{
		int L=12;
		res.resize(L,L); res.setZero();
		
		for (int i=0; i<=2; ++i) res(i,i+1) = lambda1;
		res(0,3) = lambda1;
		
		res(0,4) = lambda1;
		res(3,4) = lambda1;
		res(0,5) = lambda1;
		res(1,5) = lambda1;
		res(1,6) = lambda1;
		res(2,6) = lambda1;
		res(2,7) = lambda1;
		res(3,7) = lambda1;
		
		res(4,8) = lambda1;
		res(7,8) = lambda1;
		res(4,9) = lambda1;
		res(5,9) = lambda1;
		res(5,10) = lambda1;
		res(6,10) = lambda1;
		res(6,11) = lambda1;
		res(7,11) = lambda1;
		
		for (int i=8; i<=10; ++i) res(i,i+1) = lambda1;
		res(8,11) = lambda1;
	}
	else if (vertex_conf == "3.6^2") // truncated tetrahedron = C12
	{
		int L=12;
		res.resize(L,L); res.setZero();
		
		// A
		res(0,1) = lambda1;
		res(0,3) = lambda1;
		res(1,3) = lambda1;
		
		// B
		res(2,6) = lambda1;
		res(2,5) = lambda1;
		res(5,6) = lambda1;
		
		// C
		res(4,9) = lambda1;
		res(4,8) = lambda1;
		res(8,9) = lambda1;
		
		// D
		res(7,10) = lambda1;
		res(7,11) = lambda1;
		res(10,11) = lambda1;
		
		res(3,7) = lambda2; // A-D
		res(6,10) = lambda2; // B-D
		res(9,11) = lambda2; // C-D <--
		
		res(0,2) = lambda2; // A-B  <--
		res(1,4) = lambda2; // A-C
		res(5,8) = lambda2; // B-C
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0 and vertex_conf != "3.6^2" and vertex_conf != "3.4.3.4")
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	
	return res;
}

ArrayXXd hopping_fullerene_C40Td (int VARIANT=0, double lambda1=1., double lambda2=1.)
{
	int L = 40;
	ArrayXXd res(L,L); res.setZero();
	
	res( 0 , 2 ) = lambda1;
	res( 0 , 26 ) = lambda1;
	res( 0 , 1 ) = lambda1;
	res( 1 , 9 ) = lambda1;
	res( 1 , 28 ) = lambda1;
	res( 2 , 3 ) = lambda1;
	res( 2 , 5 ) = lambda1;
	res( 3 , 4 ) = lambda1;
	res( 3 , 15 ) = lambda1;
	res( 4 , 7 ) = lambda1;
	res( 4 , 16 ) = lambda1;
	res( 5 , 6 ) = lambda1;
	res( 5 , 9 ) = lambda1;
	res( 6 , 7 ) = lambda1;
	res( 6 , 10 ) = lambda1;
	res( 7 , 18 ) = lambda1;
	res( 8 , 9 ) = lambda1;
	res( 8 , 12 ) = lambda1;
	res( 8 , 11 ) = lambda1;
	res( 10 , 11 ) = lambda1;
	res( 10 , 37 ) = lambda1;
	res( 11 , 38 ) = lambda1;
	res( 12 , 39 ) = lambda1;
	res( 12 , 29 ) = lambda1;
	res( 13 , 15 ) = lambda1;
	res( 13 , 19 ) = lambda1;
	res( 13 , 14 ) = lambda1;
	res( 14 , 16 ) = lambda1;
	res( 14 , 23 ) = lambda1;
	res( 15 , 26 ) = lambda1;
	res( 16 , 17 ) = lambda1;
	res( 17 , 18 ) = lambda1;
	res( 17 , 21 ) = lambda1;
	res( 18 , 37 ) = lambda1;
	res( 19 , 20 ) = lambda1;
	res( 19 , 24 ) = lambda1;
	res( 20 , 23 ) = lambda1;
	res( 20 , 31 ) = lambda1;
	res( 21 , 22 ) = lambda1;
	res( 21 , 35 ) = lambda1;
	res( 22 , 23 ) = lambda1;
	res( 22 , 32 ) = lambda1;
	res( 24 , 25 ) = lambda1;
	res( 24 , 27 ) = lambda1;
	res( 25 , 26 ) = lambda1;
	res( 25 , 28 ) = lambda1;
	res( 27 , 30 ) = lambda1;
	res( 27 , 31 ) = lambda1;
	res( 28 , 29 ) = lambda1;
	res( 29 , 30 ) = lambda1;
	res( 30 , 34 ) = lambda1;
	res( 31 , 32 ) = lambda1;
	res( 32 , 33 ) = lambda1;
	res( 33 , 34 ) = lambda1;
	res( 33 , 36 ) = lambda1;
	res( 34 , 39 ) = lambda1;
	res( 35 , 37 ) = lambda1;
	res( 35 , 36 ) = lambda1;
	res( 36 , 38 ) = lambda1;
	res( 38 , 39 ) = lambda1;
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	
	return res;
}

// reference: PRB 93, 165406 (2016), Appendix C
ArrayXXd hopping_fullerene (int L=60, int VARIANT=0, double lambda1=1., double lambda2=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	// lambda1: pentagon bond, lambda2: hexagon bond
	if (L == 60)
	{
		if (VARIANT==1) // inwards spiral
		{
			for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
			res(0,4) = lambda1;
			
			res(0,8) = lambda2; //1
			res(1,11) = lambda2; //2
			res(2,14) = lambda2; //3
			res(3,17) = lambda2; //4
			res(4,5) = lambda2; //5
			
			for (int i=5; i<=18; ++i) res(i,i+1) = lambda1;
				res(6,7) = lambda2; //6
				res(9,10) = lambda2; //7
				res(12,13) = lambda2; //8
				res(15,16) = lambda2; //9
				res(18,19) = lambda2; //10
			res(5,19) = lambda1;
			
			res(7,24) = lambda1;
			res(9,25) = lambda1;
			res(10,28) = lambda1;
			res(12,29) = lambda1;
			res(13,32) = lambda1;
			res(15,33) = lambda1;
			res(16,36) = lambda1;
			res(18,37) = lambda1;
			res(19,20) = lambda1;
			res(6,21) = lambda1;
			
			for (int i=20; i<=38; ++i) res(i,i+1) = lambda1;
				res(21,22) = lambda2; //11
				res(23,24) = lambda2; //12
				res(25,26) = lambda2; //13
				res(27,28) = lambda2; //14
				res(29,30) = lambda2; //15
				res(31,32) = lambda2; //16
				res(33,34) = lambda2; //17
				res(35,36) = lambda2; //18
				res(37,38) = lambda2; //19
			res(20,39) = lambda2; //20
			
			res(26,44) = lambda1;
			res(27,46) = lambda1;
			res(30,47) = lambda1;
			res(31,49) = lambda1;
			res(34,50) = lambda1;
			res(35,52) = lambda1;
			res(38,53) = lambda1;
			res(39,40) = lambda1;
			res(22,41) = lambda1;
			res(23,43) = lambda1;
			
			for (int i=40; i<=54; ++i) res(i,i+1) = lambda1;
				res(40,41) = lambda2; //21
				res(43,44) = lambda2; //22
				res(46,47) = lambda2; //23
				res(49,50) = lambda2; //24
				res(52,53) = lambda2; //25
			res(40,54) = lambda1;
			
			res(45,57) = lambda2; //26
			res(48,58) = lambda2; //27
			res(51,59) = lambda2; //28
			res(54,55) = lambda2; //29
			res(42,56) = lambda2; //30
			
			for (int i=55; i<=58; ++i) res(i,i+1) = lambda1;
			res(55,59) = lambda1;
		}
		// reference: https://www.qmul.ac.uk/sbcs/iupac/fullerene2/311.html
		// also in: Phys. Rev. B 72, 064453 (2005)
		else if (VARIANT==2 or VARIANT==0) // Kivelson
		{
			res(0,4) = lambda1;
			res(3,4) = lambda1;
			res(2,3) = lambda1;
			res(1,2) = lambda1;
			res(0,1) = lambda1;
			
			res(0,8) = lambda1;
			res(4,5) = lambda1;
			res(3,17) = lambda1;
			res(2,14) = lambda1;
			res(1,11) = lambda1;
			
			res(7,8) = lambda1;
			res(6,7) = lambda1;
			res(5,6) = lambda1;
			res(5,19) = lambda1;
			res(18,19) = lambda1;
			res(17,18) = lambda1;
			res(16,17) = lambda1;
			res(15,16) = lambda1;
			res(14,15) = lambda1;
			res(13,14) = lambda1;
			res(12,13) = lambda1;
			res(11,12) = lambda1;
			res(10,11) = lambda1;
			res(9,10) = lambda1;
			res(8,9) = lambda1;
			
			res(7,22) = lambda1;
			res(6,21) = lambda1;
			res(19,20) = lambda1;
			res(18,29) = lambda1;
			res(16,28) = lambda1;
			res(15,27) = lambda1;
			res(13,26) = lambda1;
			res(12,25) = lambda1;
			res(10,24) = lambda1;
			res(9,23) = lambda1;
			
			res(22,33) = lambda1;
			res(32,33) = lambda1;
			res(21,32) = lambda1;
			res(20,21) = lambda1;
			res(20,31) = lambda1;
			res(30,31) = lambda1;
			res(29,30) = lambda1;
			res(28,29) = lambda1;
			res(28,39) = lambda1;
			res(38,39) = lambda1;
			res(27,38) = lambda1;
			res(26,27) = lambda1;
			res(26,37) = lambda1;
			res(36,37) = lambda1;
			res(25,36) = lambda1;
			res(24,25) = lambda1;
			res(24,35) = lambda1;
			res(34,35) = lambda1;
			res(23,34) = lambda1;
			res(22,23) = lambda1;
			
			res(33,46) = lambda1;
			res(32,44) = lambda1;
			res(31,43) = lambda1;
			res(30,41) = lambda1;
			res(39,40) = lambda1;
			res(38,53) = lambda1;
			res(37,52) = lambda1;
			res(36,50) = lambda1;
			res(35,49) = lambda1;
			res(34,47) = lambda1;
			
			res(45,46) = lambda1;
			res(44,45) = lambda1;
			res(43,44) = lambda1;
			res(42,43) = lambda1;
			res(41,42) = lambda1;
			res(40,41) = lambda1;
			res(40,54) = lambda1;
			res(53,54) = lambda1;
			res(52,53) = lambda1;
			res(51,52) = lambda1;
			res(50,51) = lambda1;
			res(49,50) = lambda1;
			res(48,49) = lambda1;
			res(47,48) = lambda1;
			res(46,47) = lambda1;
			
			res(45,57) = lambda1;
			res(42,56) = lambda1;
			res(54,55) = lambda1;
			res(51,59) = lambda1;
			res(48,58) = lambda1;
			res(56,57) = lambda1;
			res(55,56) = lambda1;
			res(55,59) = lambda1;
			res(58,59) = lambda1;
			res(57,58) = lambda1;
		}
	}
	else if (L == 20)
	{
//		res(11,12) = lambda1;
//		res(12,13) = lambda1;
//		res(13,14) = lambda1;
//		res(5,14) = lambda1;
//		res(5,6) = lambda1;
//		res(6,7) = lambda1;
//		res(7,8) = lambda1;
//		res(8,9) = lambda1;
//		res(9,10) = lambda1;
//		res(10,11) = lambda1;
//		
//		res(0,1) = lambda1;
//		res(1,2) = lambda1;
//		res(2,3) = lambda1;
//		res(3,4) = lambda1;
//		res(0,4) = lambda1;
//		
//		res(2,11) = lambda1;
//		res(3,13) = lambda1;
//		res(4,5) = lambda1;
//		res(0,7) = lambda1;
//		res(1,9) = lambda1;
//		res(2,11) = lambda1;
//		
//		res(15,16) = lambda1;
//		res(16,17) = lambda1;
//		res(17,18) = lambda1;
//		res(18,19) = lambda1;
//		res(15,19) = lambda1;
//		
//		res(12,19) = lambda1;
//		res(14,15) = lambda1;
//		res(6,16) = lambda1;
//		res(8,17) = lambda1;
//		res(10,18) = lambda1;
		
		// better numbering (inwards spiral):
		
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(3,4) = lambda1;
		res(0,4) = lambda1;
		
		res(0,7) = lambda1;
		res(1,9) = lambda1;
		res(2,11) = lambda1;
		res(3,13) = lambda1;
		res(4,5) = lambda1;
		
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(7,8) = lambda1;
		res(8,9) = lambda1;
		res(9,10) = lambda1;
		res(10,11) = lambda1;
		res(11,12) = lambda1;
		res(12,13) = lambda1;
		res(13,14) = lambda1;
		res(5,14) = lambda1;
		
		res(6,16) = lambda1;
		res(8,17) = lambda1;
		res(10,18) = lambda1;
		res(12,19) = lambda1;
		res(14,15) = lambda1;
		
		res(15,16) = lambda1;
		res(16,17) = lambda1;
		res(17,18) = lambda1;
		res(18,19) = lambda1;
		res(15,19) = lambda1;
	}
	else if (L==40) // symmetry D_5d(I)
	{
		for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
		res(0,4) = lambda1;
		
		res(0,8) = lambda1;
		res(1,11) = lambda1;
		res(2,14) = lambda1;
		res(3,17) = lambda1;
		res(4,5) = lambda1;
		
		for (int i=5; i<=18; ++i) res(i,i+1) = lambda1;
		res(5,19) = lambda1;
		
		res(6,21) = lambda1;
		res(7,23) = lambda1;
		res(9,24) = lambda1;
		res(10,26) = lambda1;
		res(12,27) = lambda1;
		res(13,29) = lambda1;
		res(15,30) = lambda1;
		res(16,32) = lambda1;
		res(18,33) = lambda1;
		res(19,20) = lambda1;
		
		for (int i=20; i<=33; ++i) res(i,i+1) = lambda1;
		res(20,34) = lambda1;
		
		res(22,36) = lambda1;
		res(25,37) = lambda1;
		res(28,38) = lambda1;
		res(31,39) = lambda1;
		res(34,35) = lambda1;
		
		for (int i=35; i<=38; ++i) res(i,i+1) = lambda1;
		res(35,39) = lambda1;
	}
	else if (L==36) // symmetry D_6h // https://nanotube.msu.edu/fullerene/fullerene.php?C=36
	{
		for (int i=0; i<=4; ++i) res(i,i+1) = lambda1;
		res(0,5) = lambda1;
		
		res(0,8) = lambda1;
		res(1,10) = lambda1;
		res(2,12) = lambda1;
		res(3,14) = lambda1;
		res(4,16) = lambda1;
		res(5,6) = lambda1;
		
		for (int i=6; i<=16; ++i) res(i,i+1) = lambda1;
		res(6,17) = lambda1;
		
		res(7,20) = lambda1;
		res(9,22) = lambda1;
		res(11,24) = lambda1;
		res(13,26) = lambda1;
		res(15,28) = lambda1;
		res(17,18) = lambda1;
		
		for (int i=18; i<=28; ++i) res(i,i+1) = lambda1;
		res(18,29) = lambda1;
		
		res(19,31) = lambda1;
		res(21,32) = lambda1;
		res(23,33) = lambda1;
		res(25,34) = lambda1;
		res(27,35) = lambda1;
		res(29,30) = lambda1;
		
		for (int i=30; i<=34; ++i) res(i,i+1) = lambda1;
		res(30,35) = lambda1;
	}
	else if (L==30) // symmetry D5h // https://nanotube.msu.edu/fullerene/fullerene.php?C=30
	{
		for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
		res(0,4) = lambda1;
		
		res(0,7) = lambda1;
		res(1,9) = lambda1;
		res(2,11) = lambda1;
		res(3,13) = lambda1;
		res(4,5) = lambda1;
		
		for (int i=5; i<=13; ++i) res(i,i+1) = lambda1;
		res(5,14) = lambda1;
		
		res(6,17) = lambda1;
		res(8,19) = lambda1;
		res(10,21) = lambda1;
		res(12,23) = lambda1;
		res(14,15) = lambda1;
		
		for (int i=15; i<=23; ++i) res(i,i+1) = lambda1;
		res(15,24) = lambda1;
		
		res(16,26) = lambda1;
		res(18,27) = lambda1;
		res(20,28) = lambda1;
		res(22,29) = lambda1;
		res(24,25) = lambda1;
		
		for (int i=25; i<=28; ++i) res(i,i+1) = lambda1;
		res(25,29) = lambda1;
	}
	else if (L==28) // symmetry Td // https://nanotube.msu.edu/fullerene/fullerene.php?C=28
	{
		res( 0 , 4 ) = lambda1;
		res( 0 , 20 ) = lambda1;
		res( 0 , 3 ) = lambda1;
		res( 1 , 19 ) = lambda1;
		res( 1 , 17 ) = lambda1;
		res( 1 , 2 ) = lambda1;
		res( 2 , 7 ) = lambda1;
		res( 2 , 3 ) = lambda1;
		res( 3 , 5 ) = lambda1;
		res( 4 , 8 ) = lambda1;
		res( 4 , 6 ) = lambda1;
		res( 5 , 23 ) = lambda1;
		res( 5 , 6 ) = lambda1;
		res( 6 , 14 ) = lambda1;
		res( 7 , 23 ) = lambda1;
		res( 7 , 27 ) = lambda1;
		res( 8 , 11 ) = lambda1;
		res( 8 , 9 ) = lambda1;
		res( 9 , 20 ) = lambda1;
		res( 9 , 10 ) = lambda1;
		res( 10 , 16 ) = lambda1;
		res( 10 , 13 ) = lambda1;
		res( 11 , 14 ) = lambda1;
		res( 11 , 12 ) = lambda1;
		res( 12 , 22 ) = lambda1;
		res( 12 , 13 ) = lambda1;
		res( 13 , 18 ) = lambda1;
		res( 14 , 25 ) = lambda1;
		res( 15 , 18 ) = lambda1;
		res( 15 , 16 ) = lambda1;
		res( 15 , 17 ) = lambda1;
		res( 16 , 19 ) = lambda1;
		res( 17 , 27 ) = lambda1;
		res( 18 , 21 ) = lambda1;
		res( 19 , 20 ) = lambda1;
		res( 21 , 22 ) = lambda1;
		res( 21 , 26 ) = lambda1;
		res( 22 , 25 ) = lambda1;
		res( 23 , 24 ) = lambda1;
		res( 24 , 25 ) = lambda1;
		res( 24 , 26 ) = lambda1;
		res( 26 , 27 ) = lambda1;
	}
	else if (L==26)
	{
		for (int i=0; i<=3; ++i) res(i,i+1) = lambda1;
		res(0,4) = lambda1;
		
		res(0,7) = lambda1;
		res(1,10) = lambda1;
		res(2,12) = lambda1;
		res(3,14) = lambda1;
		res(4,5) = lambda1;
		
		for (int i=5; i<=14; ++i) res(i,i+1) = lambda1;
		res(5,15) = lambda1;
		
		res(6,18) = lambda1;
		res(8,19) = lambda1;
		res(9,21) = lambda1;
		res(11,22) = lambda1;
		res(13,25) = lambda1;
		res(15,16) = lambda1;
		
		for (int i=16; i<=23; ++i) res(i,i+1) = lambda1;
		
		res(16,25) = lambda1;
		res(17,24) = lambda1;
		res(20,24) = lambda1;
		res(23,25) = lambda1;
	}
	else if (L==24)
	{
		for (int i=0; i<=4; ++i) res(i,i+1) = lambda1;
		res(0,5) = lambda1;
		
		res(0,8) = lambda1;
		res(1,10) = lambda1;
		res(2,12) = lambda1;
		res(3,14) = lambda1;
		res(4,16) = lambda1;
		res(5,6) = lambda1;
		
		for (int i=6; i<=16; ++i) res(i,i+1) = lambda1;
		res(6,17) = lambda1;
		
		res(17,18) = lambda1;
		res(7,19) = lambda1;
		res(9,20) = lambda1;
		res(11,21) = lambda1;
		res(13,22) = lambda1;
		res(15,23) = lambda1;
		
		for (int i=18; i<=22; ++i) res(i,i+1) = lambda1;
		res(18,23) = lambda1;
	}
	else if (L==12)
	{
		return hopping_Archimedean("3.6^2",VARIANT,lambda1,lambda2);
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		auto res_ = compress_CuthillMcKee(res,true);
		res = res_;
	}
	
	return res;
}

ArrayXXd hopping_Platonic (int L, int VARIANT=0, double lambda1=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L == 4) // tetrahedron
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		res(0,3) = lambda1;
		res(1,3) = lambda1;
		res(2,3) = lambda1;
	}
	if (L == 5) // bipyramid
	{
		res(0,1) = lambda1;
		res(0,2) = lambda1;
		res(0,3) = lambda1;
		res(1,2) = lambda1;
		res(1,3) = lambda1;
		res(2,3) = lambda1;
		res(1,4) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
	}
	else if (L == 6) // octahedron
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(0,3) = lambda1;
		
		res(0,4) = lambda1;
		res(1,4) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
		
		res(0,5) = lambda1;
		res(1,5) = lambda1;
		res(2,5) = lambda1;
		res(3,5) = lambda1;
	}
	else if (L == 8) // cube
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(0,3) = lambda1;
		
		res(4,5) = lambda1;
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(4,7) = lambda1;
		
		res(0,4) = lambda1;
		res(1,5) = lambda1;
		res(2,6) = lambda1;
		res(3,7) = lambda1;
	}
	// reference: Phys. Rev. B 72, 064453 (2005)
	else if (L == 12) // icosahedron
	{
//		res(0,1) = lambda1;
//		res(1,2) = lambda1;
//		res(0,2) = lambda1;
//		
//		res(1,5) = lambda1;
//		res(2,5) = lambda1;
//		res(2,6) = lambda1;
//		res(2,3) = lambda1;
//		res(0,3) = lambda1;
//		res(0,7) = lambda1;
//		res(0,4) = lambda1;
//		res(1,4) = lambda1;
//		res(1,8) = lambda1;
//		
//		res(4,8) = lambda1;
//		res(5,8) = lambda1;
//		res(5,6) = lambda1;
//		res(3,6) = lambda1;
//		res(3,7) = lambda1;
//		res(4,7) = lambda1;
//		
//		res(8,10) = lambda1;
//		res(5,10) = lambda1;
//		res(6,10) = lambda1;
//		res(6,11) = lambda1;
//		res(3,11) = lambda1;
//		res(7,11) = lambda1;
//		res(7,9) = lambda1;
//		res(4,9) = lambda1;
//		res(9,8) = lambda1;
//		
//		res(9,10) = lambda1;
//		res(10,11) = lambda1;
//		res(9,11) = lambda1;
		
		// better numbering (inwards spiral):
		
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		
		res(0,3) = lambda1;
		res(0,4) = lambda1;
		res(0,5) = lambda1;
		
		res(1,5) = lambda1;
		res(1,6) = lambda1;
		res(1,7) = lambda1;
		
		res(2,3) = lambda1;
		res(2,7) = lambda1;
		res(2,8) = lambda1;
		
		res(3,4) = lambda1;
		res(4,5) = lambda1;
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(7,8) = lambda1;
		res(3,8) = lambda1;
		
		res(3,9) = lambda1;
		res(4,9) = lambda1;
		res(4,10) = lambda1;
		res(5,10) = lambda1;
		res(6,10) = lambda1;
		res(6,11) = lambda1;
		res(7,11) = lambda1;
		res(8,11) = lambda1;
		res(8,9) = lambda1;
		
		res(9,10) = lambda1;
		res(10,11) = lambda1;
		res(9,11) = lambda1;
	}
	else if (L==20) // dodecahedron
	{
		res = hopping_fullerene(L, VARIANT, lambda1, lambda1);
	}
	else if (L==8) // cube
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(0,3) = lambda1;
		
		res(0,5) = lambda1;
		res(1,6) = lambda1;
		res(2,7) = lambda1;
		res(3,4) = lambda1;
		
		res(4,5) = lambda1;
		res(5,6) = lambda1;
		res(6,7) = lambda1;
		res(4,7) = lambda1;
	}
	
	res += res.transpose().eval();
	
	// not required for small Platonic solids
//	if (VARIANT==0)
//	{
//		auto res_ = compress_CuthillMcKee(res);
//		res = res_;
//	}
	
	return res;
}

ArrayXXd hopping_triangular (int L, int VARIANT=0, double lambda1=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L==3)
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
	}
	else if (L==4) // two edge-shared triangles
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(2,3) = lambda1;
		res(0,3) = lambda1;
		res(0,2) = lambda1;
	}
	else if (L==5)
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		
		res(2,3) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
	}
	else if (L==6)
	{
		res(0,1) = lambda1;
		res(1,2) = lambda1;
		res(0,2) = lambda1;
		
		res(2,3) = lambda1;
		res(2,4) = lambda1;
		res(3,4) = lambda1;
		
		res(1,4) = lambda1;
		res(1,5) = lambda1;
		res(4,5) = lambda1;
	}
	
	res += res.transpose().eval();
	return res;
}

void add_tetrahedron (int i, int j, int k, int l, vector<pair<size_t,size_t>> &target)
{
//	std::cout << "ijkl=" << i << ", " << j << ", " << k << ", " << l << std::endl;
//	
	(i<j)? target.push_back(pair<size_t,size_t>(i,j)) : target.push_back(pair<size_t,size_t>(j,i));
	(i<k)? target.push_back(pair<size_t,size_t>(i,k)) : target.push_back(pair<size_t,size_t>(k,i));
	(i<l)? target.push_back(pair<size_t,size_t>(i,l)) : target.push_back(pair<size_t,size_t>(l,i));
	(j<k)? target.push_back(pair<size_t,size_t>(j,k)) : target.push_back(pair<size_t,size_t>(k,j));
	(j<l)? target.push_back(pair<size_t,size_t>(j,l)) : target.push_back(pair<size_t,size_t>(l,j));
	(k<l)? target.push_back(pair<size_t,size_t>(k,l)) : target.push_back(pair<size_t,size_t>(l,k));
//	
//	for (int n=0; n<6; ++n)
//	{
//		std::cout << "tetrahedron: " << target[target.size()-6+n].first << ", " << target[target.size()-6+n].second << std::endl;
//	}
//	std::cout << std::endl;
}

void add_edge (int i, int j, vector<pair<size_t,size_t>> &target)
{
	(i<j)? target.push_back(pair<size_t,size_t>(i,j)) : target.push_back(pair<size_t,size_t>(j,i));
}

ArrayXXd hopping_sodaliteCage (int L=60, int VARIANT=0, double lambda1=1.)
{
	std::vector<std::pair<std::size_t, std::size_t>> edges;
	
	ArrayXXd res(L,L); res.setZero();
	
	if (L==60)
	{
		if (VARIANT==0)
		{
			// dmax=16 with Cuthill-McKee algorithm
			add_tetrahedron(0,1,3,4,edges); // dmax=4
			add_tetrahedron(0,2,5,6,edges); // dmax=6
			add_tetrahedron(5,13,14,15,edges); // dmax=10
			add_tetrahedron(14,20,27,28,edges); // dmax=14
			add_tetrahedron(8,19,20,21,edges); // dmax=13
			add_tetrahedron(3,7,8,9,edges); // dmax=6
			
			add_tetrahedron(9,11,22,23,edges); // dmax=14
			add_tetrahedron(4,10,11,12,edges); // dmax=8
			add_tetrahedron(6,16,17,18,edges); // dmax=12
			add_tetrahedron(15,18,29,30,edges); // dmax=15
			add_tetrahedron(28,35,42,43,edges); // dmax=15
			add_tetrahedron(21,33,34,35,edges); // dmax=14
			
			add_tetrahedron(34,37,49,50,edges); // dmax=16
			add_tetrahedron(23,36,37,38,edges); // dmax=15
			add_tetrahedron(12,24,25,26,edges); // dmax=14
			add_tetrahedron(17,25,31,32,edges); // dmax=15
			add_tetrahedron(30,44,45,46,edges); // dmax=16
			add_tetrahedron(43,45,53,54,edges); // dmax=11
			
			add_tetrahedron(54,56,58,59,edges); // dmax=5
			add_tetrahedron(50,52,57,58,edges); // dmax=8
			add_tetrahedron(38,41,51,52,edges); // dmax=14
			add_tetrahedron(26,39,40,41,edges); // dmax=15
			add_tetrahedron(32,40,47,48,edges); // dmax=16
			add_tetrahedron(46,48,55,56,edges); // dmax=10
		}
		else if (VARIANT==1)
		{
			// dmax=25 without optimization
			add_tetrahedron(0,5,6,13,edges); // dmax=13
			add_tetrahedron(0,1,7,14,edges); // dmax=14
			add_tetrahedron(1,2,8,15,edges); // dmax=14
			add_tetrahedron(2,3,9,16,edges); // dmax=14
			add_tetrahedron(3,4,10,17,edges); // dmax=14
			add_tetrahedron(4,5,11,12,edges); // dmax=8
			
			add_tetrahedron(12,20,21,33,edges); // dmax=21
			add_tetrahedron(13,21,22,34,edges); // dmax=21
			add_tetrahedron(14,24,25,37,edges); // dmax=23
			add_tetrahedron(15,25,26,38,edges); // dmax=23
			add_tetrahedron(16,28,29,41,edges); // dmax=25
			add_tetrahedron(17,18,29,30,edges); // dmax=13
			
			add_tetrahedron(18,19,31,43,edges); // dmax=25
			add_tetrahedron(19,20,32,44,edges); // dmax=25
			add_tetrahedron(22,23,35,45,edges); // dmax=23
			add_tetrahedron(23,24,36,46,edges); // dmax=23
			add_tetrahedron(26,27,39,47,edges); // dmax=21
			add_tetrahedron(27,28,40,42,edges); // dmax=15
			
			add_tetrahedron(42,57,58,59,edges); // dmax=17
			add_tetrahedron(43,55,56,57,edges); // dmax=14
			add_tetrahedron(44,53,54,55,edges); // dmax=11
			add_tetrahedron(45,51,52,53,edges); // dmax=8
			add_tetrahedron(46,49,50,51,edges); // dmax=5
			add_tetrahedron(47,48,49,59,edges); // dmax=12
		}
	}
	else if (L==50)
	{
		if (VARIANT==0)
		{
			add_tetrahedron(0,2,3,6,edges);
			add_tetrahedron(3,8,7,9,edges);
			add_tetrahedron(8,11,19,20,edges);
			add_tetrahedron(4,10,11,12,edges);
			add_tetrahedron(0,1,4,5,edges);
			
			add_tetrahedron(6,16,17,18,edges);
			add_tetrahedron(9,21,22,23,edges);
			add_tetrahedron(20,33,34,35,edges);
			add_tetrahedron(12,24,25,26,edges);
			add_tetrahedron(5,13,14,15,edges);
			
			add_tetrahedron(15,17,29,30,edges);
			add_tetrahedron(18,22,31,32,edges);
			add_tetrahedron(23,34,36,37,edges);
			add_tetrahedron(25,35,38,39,edges);
			add_tetrahedron(14,26,27,28,edges);
			
			add_tetrahedron(30,42,43,44,edges);
			add_tetrahedron(32,44,45,46,edges);
			add_tetrahedron(37,46,47,48,edges);
			add_tetrahedron(39,41,48,49,edges);
			add_tetrahedron(28,40,41,42,edges);
		}
		else if (VARIANT==1)
		{
			add_tetrahedron(0,1,6,12,edges);
			add_tetrahedron(1,2,7,13,edges);
			add_tetrahedron(2,3,8,14,edges);
			add_tetrahedron(3,4,9,10,edges);
			add_tetrahedron(0,4,5,11,edges);
			
			add_tetrahedron(12,21,31,32,edges);
			add_tetrahedron(13,23,33,34,edges);
			add_tetrahedron(14,15,25,26,edges);
			add_tetrahedron(10,17,27,28,edges);
			add_tetrahedron(11,19,29,30,edges);
			
			add_tetrahedron(20,30,31,38,edges);
			add_tetrahedron(22,32,33,39,edges);
			add_tetrahedron(24,25,34,35,edges);
			add_tetrahedron(16,26,27,36,edges);
			add_tetrahedron(18,28,29,37,edges);
			
			add_tetrahedron(38,44,45,49,edges);
			add_tetrahedron(39,40,45,46,edges);
			add_tetrahedron(35,41,46,47,edges);
			add_tetrahedron(36,42,47,48,edges);
			add_tetrahedron(37,43,48,49,edges);
		}
	}
	else if (L==20)
	{
		if (VARIANT==0)
		{
			add_tetrahedron(0,1,4,5,edges);
			add_tetrahedron(0,2,3,6,edges);
			add_tetrahedron(3,7,8,9,edges);
			add_tetrahedron(8,4,10,11,edges);
			
			add_tetrahedron(5,13,12,14,edges);
			add_tetrahedron(6,14,15,16,edges);
			add_tetrahedron(9,16,17,18,edges);
			add_tetrahedron(11,13,18,19,edges);
		}
		else if (VARIANT==1)
		{
			add_tetrahedron(0,3,4,12,edges);
			add_tetrahedron(0,1,5,13,edges);
			add_tetrahedron(1,2,6,14,edges);
			add_tetrahedron(2,3,7,15,edges);
			
			add_tetrahedron(4,8,9,16,edges);
			add_tetrahedron(5,9,10,17,edges);
			add_tetrahedron(6,10,11,18,edges);
			add_tetrahedron(7,8,11,19,edges);
		}
	}
	else if (L==16)
	{
		add_tetrahedron(1,2,3,4,edges);
		add_tetrahedron(4,6,7,8,edges);
		add_tetrahedron(8,9,10,11,edges);
		add_tetrahedron(3,9,14,15,edges);
		
		add_edge(0,1,edges);
		add_edge(5,6,edges);
		add_edge(11,12,edges);
		add_edge(13,14,edges);
	}
	else if (L==28)
	{
		add_tetrahedron(0,1,2,3,edges);
		add_tetrahedron(3,4,5,6,edges);
		add_tetrahedron(7,8,9,10,edges);
		add_tetrahedron(10,11,12,13,edges);
		add_tetrahedron(14,15,16,17,edges);
		add_tetrahedron(17,18,19,20,edges);
		add_tetrahedron(21,22,23,24,edges);
		add_tetrahedron(24,25,26,27,edges);
		
		add_edge(0,26,edges);
		add_edge(1,27,edges);
		add_edge(5,8,edges);
		add_edge(6,7,edges);
		add_edge(12,15,edges);
		add_edge(13,14,edges);
		add_edge(19,22,edges);
		add_edge(20,21,edges);
	}
	
	for (int e=0; e<edges.size(); ++e)
	{
		int i = edges[e].first;
		int j = edges[e].second;
		res(i,j) = lambda1;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		compress_CuthillMcKee(res,true);
	}
	
	return res;
}

ArrayXXd hopping_Mn32 (double lambda_cap=1., double lambda_corner=0., double lambda_edge=1., int VARIANT=0)
{
	std::vector<std::pair<std::size_t, std::size_t>> edges;
	
	ArrayXXd res(32,32); res.setZero();
	
	// outer circle:
	add_tetrahedron(0,1,2,3,edges);
	add_tetrahedron(4,5,6,7,edges);
	add_tetrahedron(8,9,10,11,edges);
	add_tetrahedron(12,13,14,15,edges);
	// inner circle:
	add_tetrahedron(16,17,18,19,edges);
	add_tetrahedron(20,21,22,23,edges);
	add_tetrahedron(24,25,26,27,edges);
	add_tetrahedron(28,29,30,31,edges);
	
	vector<int> caps = {3,7,11,15, 19,23,27,31};
	
	for (int e=0; e<edges.size(); ++e)
	{
		int i = edges[e].first;
		int j = edges[e].second;
		
		auto it_i = find(caps.begin(), caps.end(), i);
		auto it_j = find(caps.begin(), caps.end(), j);
		
		if (it_i!=caps.end() or it_j!=caps.end())
		{
			res(i,j) = lambda_cap;
		}
		else
		{
			res(i,j) = lambda_corner;
		}
	}
	
	edges.clear();
	
	// outer circle:
	edges.push_back(pair<size_t,size_t>(1,4));
	edges.push_back(pair<size_t,size_t>(5,8));
	edges.push_back(pair<size_t,size_t>(9,12));
	edges.push_back(pair<size_t,size_t>(0,13));
	// connection:
	edges.push_back(pair<size_t,size_t>(2,16));
	edges.push_back(pair<size_t,size_t>(6,21));
	edges.push_back(pair<size_t,size_t>(10,26));
	edges.push_back(pair<size_t,size_t>(14,29));
	// inner circle:
	edges.push_back(pair<size_t,size_t>(17,20));
	edges.push_back(pair<size_t,size_t>(22,24));
	edges.push_back(pair<size_t,size_t>(25,28));
	edges.push_back(pair<size_t,size_t>(18,30));
	
	for (int e=0; e<edges.size(); ++e)
	{
		int i = edges[e].first;
		int j = edges[e].second;
		res(i,j) = lambda_edge;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		compress_CuthillMcKee(res,true);
	}
	
	return res;
}

pair<ArrayXXd,vector<SUB_LATTICE> > hopping_PPV (int L, int VARIANT=0, double t0=1., double tsingle=1., double tdouble=1., string BC="")
{
	ArrayXXd res(L,L); res.setZero();
	assert(L==8);
	vector<SUB_LATTICE> G;
	
	if (BC == "INIT_HEX")
	{
		res(0,1) = t0;
		res(0,2) = t0;
		res(1,3) = t0;
		res(2,4) = t0;
		res(3,5) = t0;
		res(4,5) = t0;
		res(5,6) = tsingle; // tvinyl(single)
		res(6,7) = tdouble; // tvinyl(double)
		// intermolecular hopping is is tsingle
		
		// AABA|BBAB
		G.push_back(static_cast<SUB_LATTICE>(1)); //0
		G.push_back(static_cast<SUB_LATTICE>(-1)); //1
		G.push_back(static_cast<SUB_LATTICE>(-1)); //2
		G.push_back(static_cast<SUB_LATTICE>(1)); //3
		G.push_back(static_cast<SUB_LATTICE>(1)); //4
		G.push_back(static_cast<SUB_LATTICE>(-1)); //5
		G.push_back(static_cast<SUB_LATTICE>(1)); //6
		G.push_back(static_cast<SUB_LATTICE>(-1)); //7
	}
	else // PPV
	{
		res(0,1) = tsingle;
		res(1,2) = t0;
		res(1,3) = t0;
		res(2,4) = t0;
		res(3,5) = t0;
		res(4,6) = t0;
		res(5,6) = t0;
		res(6,7) = tsingle; // tvinyl(single)
		// intermolecular hopping is is tdouble
		
		G.push_back(static_cast<SUB_LATTICE>(1)); //0
		G.push_back(static_cast<SUB_LATTICE>(-1)); //1
		G.push_back(static_cast<SUB_LATTICE>(1)); //2
		G.push_back(static_cast<SUB_LATTICE>(1)); //3
		G.push_back(static_cast<SUB_LATTICE>(-1)); //4
		G.push_back(static_cast<SUB_LATTICE>(-1)); //5
		G.push_back(static_cast<SUB_LATTICE>(1)); //6
		G.push_back(static_cast<SUB_LATTICE>(-1)); //7
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		CuthillMcKeeCompressor CMK(res,true);
		CMK.apply_compression(res);
		
		vector<SUB_LATTICE> G_ = G;
		for (int i=0; i<L; ++i)
		{
			G[CMK.get_transform()[i]] = G_[i];
		}
	}
	
	pair<ArrayXXd,vector<SUB_LATTICE> > ret(res,G);
	return ret;
}

pair<ArrayXXd,vector<SUB_LATTICE> > hopping_triangulene (int L, int VARIANT=0, double lambda=1., double lambda2=1., string BC="")
{
	ArrayXXd res(L,L); res.setZero();
	assert(L==13 or L==8 or L==10 or L==22 or L==4); // 33, 46, 61 // 4: simplified model, 8: simplified, larger unit cell with different coupling
	// (6+)7+9+11+13
	
	vector<SUB_LATTICE> G;
	
	if (L==4)
	{
		res(0,2) = lambda;
		res(1,2) = lambda;
		res(2,3) = lambda;
		
		G.push_back(static_cast<SUB_LATTICE>(1)); //0
		G.push_back(static_cast<SUB_LATTICE>(1)); //1
		G.push_back(static_cast<SUB_LATTICE>(-1)); //2
		G.push_back(static_cast<SUB_LATTICE>(1)); //3
	}
	else if (L==8)
	{
		if (BC == "CUT_HEX")
		{
			res(0,2) = lambda;
			res(1,2) = lambda;
			res(2,3) = lambda;
			res(3,4) = lambda2; // tinter
			res(4,5) = lambda;
			res(5,6) = lambda;
			res(5,7) = lambda;
			// intermolecular hopping is thex (2x)
			
			// AABA|BBAB
			G.push_back(static_cast<SUB_LATTICE>(1)); //0
			G.push_back(static_cast<SUB_LATTICE>(1)); //1
			G.push_back(static_cast<SUB_LATTICE>(-1)); //2
			G.push_back(static_cast<SUB_LATTICE>(1)); //3
			G.push_back(static_cast<SUB_LATTICE>(-1)); //4
			G.push_back(static_cast<SUB_LATTICE>(1)); //5
			G.push_back(static_cast<SUB_LATTICE>(-1)); //6
			G.push_back(static_cast<SUB_LATTICE>(-1)); //7
		}
		else
		{
			res(0,1) = lambda;
			res(1,2) = lambda;
			res(1,3) = lambda;
			res(2,4) = lambda2; // thex
			res(3,5) = lambda2; // thex
			res(4,6) = lambda;
			res(5,6) = lambda;
			res(6,7) = lambda;
			// intermolecular hopping is tinter
			
			G.push_back(static_cast<SUB_LATTICE>(1)); //0
			G.push_back(static_cast<SUB_LATTICE>(-1)); //1
			G.push_back(static_cast<SUB_LATTICE>(1)); //2
			G.push_back(static_cast<SUB_LATTICE>(1)); //3
			G.push_back(static_cast<SUB_LATTICE>(-1)); //4
			G.push_back(static_cast<SUB_LATTICE>(-1)); //5
			G.push_back(static_cast<SUB_LATTICE>(1)); //6
			G.push_back(static_cast<SUB_LATTICE>(-1)); //7
		}
	}
	else if (L==10) //porphyrine coarse-grained
	{
		res(0,1) = lambda;
		
		res(1,2) = lambda;
		res(1,3) = lambda;
		
		res(2,4) = lambda;
		res(4,6) = lambda;
		
		res(3,5) = lambda;
		res(5,7) = lambda;
		
		res(6,8) = lambda;
		res(7,8) = lambda;
		
		res(8,9) = lambda;
		
		G.push_back(static_cast<SUB_LATTICE>(1)); //0
		G.push_back(static_cast<SUB_LATTICE>(-1)); //1
		G.push_back(static_cast<SUB_LATTICE>(1)); //2
		G.push_back(static_cast<SUB_LATTICE>(1)); //3
		
		G.push_back(static_cast<SUB_LATTICE>(-1)); //4
		G.push_back(static_cast<SUB_LATTICE>(-1)); //5
		
		G.push_back(static_cast<SUB_LATTICE>(1)); //6
		G.push_back(static_cast<SUB_LATTICE>(1)); //7
		G.push_back(static_cast<SUB_LATTICE>(-1)); //8
		G.push_back(static_cast<SUB_LATTICE>(1)); //9
	}
	else if (L==13)
	{
		for (int i=0; i<=10; ++i) res(i,i+1) = lambda;
		res(0,11) = lambda;
		
		res(6,12) = lambda;
		res(2,12) = lambda;
		res(10,12) = lambda;
	}
	else if (L==22) // for triangulene proper, lambda2 is the NNN hopping t3
	{
		res(11,15) = lambda;
		res(15,19) = lambda;
		
		res(8,11) = lambda;
		res(16,19) = lambda;
		
		res(5,8) = lambda;
		res(8,12) = lambda;
		res(12,16) = lambda;
		res(16,20) = lambda;
		
		res(3,5) = lambda;
		res(9,12) = lambda;
		res(17,20) = lambda;
		
		res(1,3) = lambda;
		res(3,6) = lambda;
		res(6,9) = lambda;
		res(9,13) = lambda;
		res(13,17) = lambda;
		res(17,21) = lambda;
		
		res(0,1) = lambda;
		res(4,6) = lambda;
		res(10,13) = lambda;
		res(18,21) = lambda;
		
		res(0,2) = lambda;
		res(2,4) = lambda;
		res(4,7) = lambda;
		res(7,10) = lambda;
		res(10,14) = lambda;
		res(14,18) = lambda;
		
		G.push_back(static_cast<SUB_LATTICE>(1)); //0
		G.push_back(static_cast<SUB_LATTICE>(-1)); //1
		G.push_back(static_cast<SUB_LATTICE>(-1)); //2
		G.push_back(static_cast<SUB_LATTICE>(1)); //3
		G.push_back(static_cast<SUB_LATTICE>(1)); //4
		G.push_back(static_cast<SUB_LATTICE>(-1)); //5
		G.push_back(static_cast<SUB_LATTICE>(-1)); //6
		G.push_back(static_cast<SUB_LATTICE>(-1)); //7
		G.push_back(static_cast<SUB_LATTICE>(1)); //8
		G.push_back(static_cast<SUB_LATTICE>(1)); //9
		G.push_back(static_cast<SUB_LATTICE>(1)); //10
		G.push_back(static_cast<SUB_LATTICE>(-1)); //11
		G.push_back(static_cast<SUB_LATTICE>(-1)); //12
		G.push_back(static_cast<SUB_LATTICE>(-1)); //13
		G.push_back(static_cast<SUB_LATTICE>(-1)); //14
		G.push_back(static_cast<SUB_LATTICE>(1)); //15
		G.push_back(static_cast<SUB_LATTICE>(1)); //16
		G.push_back(static_cast<SUB_LATTICE>(1)); //17
		G.push_back(static_cast<SUB_LATTICE>(1)); //18
		G.push_back(static_cast<SUB_LATTICE>(-1)); //19
		G.push_back(static_cast<SUB_LATTICE>(-1)); //20
		G.push_back(static_cast<SUB_LATTICE>(-1)); //21
		
		res(12,15) = lambda2;
		res(11,16) = lambda2;
		res(8,19) = lambda2;
		
		res(5,9) = lambda2;
		res(3,12) = lambda2;
		res(6,8) = lambda2;
		
		res(13,16) = lambda2;
		res(12,17) = lambda2;
		res(9,20) = lambda2;
		
		res(2,3) = lambda2;
		res(1,4) = lambda2;
		res(0,6) = lambda2;
		
		res(6,10) = lambda2;
		res(7,9) = lambda2;
		res(4,13) = lambda2;
		
		res(10,21) = lambda2;
		res(14,17) = lambda2;
		res(13,18) = lambda2;
	}
	
	res += res.transpose().eval();
	
	//cout << "VARIANT=" << VARIANT << endl;
	if (VARIANT==0)
	{
		//auto res_ = compress_CuthillMcKee(res,true);
		CuthillMcKeeCompressor CMK(res,true);
		CMK.apply_compression(res);
		
		vector<SUB_LATTICE> G_ = G;
		for (int i=0; i<L; ++i)
		{
			G[CMK.get_transform()[i]] = G_[i];
		}
		
//		for (int i=0; i<L; ++i)
//		{
//			lout << "i=" << i << ", G[i]=" << G[i]<< ", orig.G[i]=" << G_[i] << endl;
//		}
	}
	
	pair<ArrayXXd,vector<SUB_LATTICE> > ret(res,G);
	return ret;
}

pair<ArrayXXd,vector<SUB_LATTICE> > hopping_triangulene_dimer (int VARIANT=0, double lambda=1., double lambda2=1., string BC="")
{
	auto [T,G0] = hopping_triangulene(22,1,lambda,lambda2);
	ArrayXXd res(44,44); res = 0.;
	res.topLeftCorner(22,22) = T;
	res.bottomRightCorner(22,22) = T;
	
	ArrayXXd coupling(44,44); coupling = 0.;
	coupling(2,1+22) = lambda;
	coupling(7,5+22) = lambda;
	coupling(14,11+22) = lambda;
	
	coupling(2,1+22) = lambda2;
	coupling(2,5+22) = lambda2;
	
	coupling(4,3+22) = lambda2;
	
	coupling(7,1+22) = lambda2;
	coupling(7,11+22) = lambda2;
	
	coupling(10,8+22) = lambda2;
	
	coupling(14,5+22) = lambda2;
	coupling(14,11+22) = lambda2;
	
	coupling += coupling.transpose().eval();
	
	res += coupling;
	
	vector<SUB_LATTICE> G;
	for (int i=0; i<22; ++i) G.push_back(G0[i]);
	for (int i=0; i<22; ++i) G.push_back(flip_sublattice(G0[i]));
	
	if (VARIANT==0)
	{
		CuthillMcKeeCompressor CMK(res,true);
		CMK.apply_compression(res);
		
		vector<SUB_LATTICE> G_ = G;
		for (int i=0; i<44; ++i)
		{
			G[CMK.get_transform()[i]] = G_[i];
		}
	}
	
	pair<ArrayXXd,vector<SUB_LATTICE> > ret(res,G);
	return ret;
}

ArrayXXd hopping_coronene (int L, int VARIANT=0, double lambda=1.)
{
	ArrayXXd res(L,L); res.setZero();
	assert(L==24);
	
	if (L==24)
	{
		res(0,2) = lambda;
		res(0,3) = lambda;
		res(1,3) = lambda;
		res(1,4) = lambda;
		res(2,5) = lambda;
		res(3,6) = lambda;
		res(4,7) = lambda;
		res(5,8) = lambda;
		res(5,9) = lambda;
		res(6,9) = lambda;
		res(6,10) = lambda;
		res(7,10) = lambda;
		res(7,11) = lambda;
		res(8,12) = lambda;
		res(9,13) = lambda;
		res(10,14) = lambda;
		res(11,15) = lambda;
		res(12,16) = lambda;
		res(13,16) = lambda;
		res(13,17) = lambda;
		res(14,17) = lambda;
		res(14,18) = lambda;
		res(15,18) = lambda;
		res(16,19) = lambda;
		res(17,20) = lambda;
		res(18,21) = lambda;
		res(19,22) = lambda;
		res(20,22) = lambda;
		res(20,23) = lambda;
		res(21,23) = lambda;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		compress_CuthillMcKee(res,true);
	}
	
	return res;
}

ArrayXXd hopping_corannulene (int L, int VARIANT=0, double lambda=1.)
{
	ArrayXXd res(L,L); res.setZero();
	assert(L==20);
	
	if (L==20)
	{
		for (int i=0; i<=13; ++i) res(i,i+1) = lambda;
		res(0,14) = lambda;
		
		res(13,15) = lambda;
		res(10,16) = lambda;
		res(7,17) = lambda;
		res(4,18) = lambda;
		res(1,19) = lambda;
		
		for (int i=15; i<=18; ++i) res(i,i+1) = lambda;
		res(15,19) = lambda;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		compress_CuthillMcKee(res,true);
	}
	
	return res;
}

ArrayXXd hopping_square_plaquette (int L, int VARIANT=0, double lambda=1.)
{
	ArrayXXd res(L,L); res.setZero();
	
	if (L==16)
	{
		for (int i=0; i<3; ++i) res(i,i+1) = lambda;
		for (int i=4; i<7; ++i) res(i,i+1) = lambda;
		for (int i=8; i<11; ++i) res(i,i+1) = lambda;
		for (int i=12; i<15; ++i) res(i,i+1) = lambda;
		
		for (int i=0; i<=2; ++i) res(4*i,4*i+4) = lambda;
		for (int i=0; i<=2; ++i) res(1+4*i,1+4*i+4) = lambda;
		for (int i=0; i<=2; ++i) res(2+4*i,2+4*i+4) = lambda;
		for (int i=0; i<=2; ++i) res(3+4*i,3+4*i+4) = lambda;
		
//		res(0,3) = lambda;
//		res(4,7) = lambda;
//		res(8,11) = lambda;
//		res(12,15) = lambda;
//		
//		res(0,12) = lambda;
//		res(1,13) = lambda;
//		res(2,14) = lambda;
//		res(3,15) = lambda;
	}
	if (L==20)
	{
		for (int i=0; i<3; ++i) res(i,i+1) = lambda;
		for (int i=4; i<7; ++i) res(i,i+1) = lambda;
		for (int i=8; i<11; ++i) res(i,i+1) = lambda;
		for (int i=12; i<15; ++i) res(i,i+1) = lambda;
		for (int i=16; i<19; ++i) res(i,i+1) = lambda;
		
		for (int i=0; i<=3; ++i) res(4*i,4*i+4) = lambda;
		for (int i=0; i<=3; ++i) res(1+4*i,1+4*i+4) = lambda;
		for (int i=0; i<=3; ++i) res(2+4*i,2+4*i+4) = lambda;
		for (int i=0; i<=3; ++i) res(3+4*i,3+4*i+4) = lambda;
		
		res(0,3) = lambda;
		res(4,7) = lambda;
		res(8,11) = lambda;
		res(12,15) = lambda;
		res(16,19) = lambda;
		
		res(0,16) = lambda;
		res(1,17) = lambda;
		res(2,18) = lambda;
		res(3,19) = lambda;
	}
	
	res += res.transpose().eval();
	
	if (VARIANT==0)
	{
		compress_CuthillMcKee(res,true);
	}
	
	return res;
}

// Calculates distance matrix from adjacency matrix of a graph
// The distance matrix has 1 for nearest neighbours, 2 for next-nearest neighbours etc.
ArrayXXi calc_distanceMatrix (ArrayXXd adjacencyMatrix)
{
	int L = adjacencyMatrix.rows();
	assert(adjacencyMatrix.cols() == L);
	
	ArrayXXi dist0(L,L); dist0 = 0;
	dist0.matrix().diagonal().setConstant(1);
	
	ArrayXXi dist1(L,L); dist1 = 0;
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		if (abs(adjacencyMatrix(i,j)) > 0.)
		{
			dist1(i,j) = 1;
		}
	}
	
	ArrayXXi next = dist1;
	
	// each entry is a matrix with 1 for the given distance, 0 otherwise
	vector<ArrayXXi> dist;
	dist.push_back(dist0);
	dist.push_back(dist1);
	
	while (next.sum() != 0)
	{
		// calculate d-th power of adjacency matrix
		next = next.matrix()*dist1.matrix();
		
		// remove already known distances (multiply array-wise with 0)
		for (int d=0; d<dist.size(); ++d) next = next*(1-dist[d]);
		// set non-zero entries to 1
		for (int i=0; i<L; ++i) for (int j=0; j<L; ++j) if (next(i,j)>0) next(i,j) = 1;
		
		dist.push_back(next);
	}
	
	ArrayXXi res(L,L); res = 0;
	for (int d=0; d<dist.size(); ++d) res = res+d*dist[d];
	return res;
}

ArrayXXi hopping_MnRing (int Ncells, double J1, double J2, double J3, double J4=0., double J5=0., double J6=0., double J7=0.)
{
	int L = 7*Ncells;
	ArrayXXi res(L,L);
	
	for (int ic=0; ic<Ncells; ++ic)
	{
		int icell = ic*7;
	
		res((0+icell)%L,(1+icell)%L) = 1.;
		res((1+icell)%L,(2+icell)%L) = 1.;
		
		res((0+icell)%L,(3+icell)%L) = J2/J1;
		res((1+icell)%L,(3+icell)%L) = J2/J1;
		res((1+icell)%L,(4+icell)%L) = J2/J1;
		res((2+icell)%L,(4+icell)%L) = J2/J1;
		
		res((3+icell)%L,(4+icell)%L) = J3/J1;
		res((3+icell)%L,(5+icell)%L) = J3/J1;
		res((3+icell)%L,(6+icell)%L) = J3/J1;
		res((4+icell)%L,(5+icell)%L) = J3/J1;
		res((5+icell)%L,(6+icell)%L) = J3/J1;
		
		res((5+icell)%L,(7+icell)%L) = J2/J1;
		res((5+icell)%L,(8+icell)%L) = J2/J1;
		res((6+icell)%L,(8+icell)%L) = J2/J1;
		res((6+icell)%L,(9+icell)%L) = J2/J1;
		
		res((0+icell)%L,(4+icell)%L) = J4/J1;
		res((2+icell)%L,(3+icell)%L) = J4/J1;
		res((1+icell)%L,(6+icell)%L) = J4/J1;
		
		res((5+icell)%L,(9+icell)%L) = J4/J1;
		res((6+icell)%L,(7+icell)%L) = J4/J1;
		res((4+icell)%L,(8+icell)%L) = J4/J1;
	}
	
	res += res.transpose().eval();
	compress_CuthillMcKee(res,true);
	
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
////			cout << ix << ", " << iy << " → index=" << index_i << endl;
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

// Warning: changed arguments to L,Ly,maxPower from Ly,maxPower on 2024-01-18
vector<Param> Tinf_params_fermions (size_t L, size_t Ly, size_t maxPower=1ul)
{
	vector<Param> res;
	res.push_back({"Ly",Ly});
	res.push_back({"maxPower",maxPower});
	res.push_back({"OPEN_BC",true});
	if (Ly == 2ul)
	{
		res.push_back({"t",0.});
		res.push_back({"tRung",1.});
	}
	else
	{
		res.push_back({"t",1.,0});
		res.push_back({"t",0.,1});
	}
	vector<SUB_LATTICE> G;
	for (int l=0; l<L; l+=4)
	{
		G.push_back(static_cast<SUB_LATTICE>(1));
		G.push_back(static_cast<SUB_LATTICE>(-1));
		G.push_back(static_cast<SUB_LATTICE>(-1));
		G.push_back(static_cast<SUB_LATTICE>(1));
	}
	res.push_back({"G",G});
	//for (size_t l=0; l<L; ++l) {lout << G[l];}
	//lout << endl;
	return res;
}

vector<Param> Tinf_params_spins (size_t L, size_t Ly, size_t maxPower=1ul, size_t DA=2ul, size_t DB=2ul, bool SOFT=false)
{
	vector<Param> res;
	res.push_back({"Ly",Ly});
	res.push_back({"maxPower",maxPower});
	if (SOFT and DA==3ul and DB==3ul)
	{
		size_t D = 3ul;
		if (Ly==2)
		{
			for (size_t l=1; l<L-1; ++l)
			{
				res.push_back({"D",D,l});
			}
			res.push_back({"D",2ul,0});
			res.push_back({"D",2ul,L-1});
		}
		else if (Ly==1)
		{
			for (size_t l=2; l<2*L-2; ++l)
			{
				res.push_back({"D",D,l});
			}
			res.push_back({"D",2ul,0});
			res.push_back({"D",2ul,1});
			res.push_back({"D",2ul,L-2});
			res.push_back({"D",2ul,L-1});
		}
	}
	else
	{
		if (Ly == 2ul)
		{
			for (size_t l=0; l<L; l+=2)
			{
				res.push_back({"D",DA,l});
				res.push_back({"D",DB,l+1});
			}
		}
		else
		{
			for (size_t l=0; l<L; l+=4)
			{
				res.push_back({"D",DA,l});
				res.push_back({"D",DA,l+1});
				res.push_back({"D",DB,l+2});
				res.push_back({"D",DB,l+3});
			}
		}
	}
	if (Ly == 2ul)
	{
		res.push_back({"J",0.});
		res.push_back({"Jrung",1.});
	}
	else
	{
		res.push_back({"J",1.,0});
		res.push_back({"J",0.,1});
	}
	return res;
}

inline double conjIfImag (double x) {return x;}
inline std::complex<double> conjIfImag (std::complex<double> x) {return conj(x);}

template<typename Scalar>
Array<Scalar,Dynamic,Dynamic> hopping_PAM (int L, Scalar tfc, Scalar tcc, Scalar tff, Scalar tx, Scalar ty, bool PBC=false)
{
	Array<Scalar,Dynamic,Dynamic> t1site(2,2); t1site = 0;
	t1site(0,1) = tfc;
	
	// L: Anzahl der physikalischen fc-Sites
	Array<Scalar,Dynamic,Dynamic> res(2*L,2*L); res = 0;
	
	for (int l=0; l<L; ++l)
	{
		res.block(2*l,2*l, 2,2) = t1site;
	}
	
	for (int l=0; l<L-1; ++l)
	{
		res(2*l,   2*l+2) = tcc;
		res(2*l+1, 2*l+3) = tff;
		res(2*l+1, 2*l+2) = tx;
		res(2*l,   2*l+3) = conjIfImag(ty);
	}
	
//	cout << "OBC:" << endl;
//	cout << endl << res << endl << endl;
	if (PBC)
	{
		for (int l=0; l<2*L-4; l+=2)
		{
			// increase hopping range
			res(l,  l+4) = res(l,  l+2); // d=2 -> d=4
			res(l,  l+5) = res(l,  l+3); // d=3 -> d=5
			res(l+1,l+4) = res(l+1,l+2); // d=1 -> d=3
			res(l+1,l+5) = res(l+1,l+3); // d=2 -> d=4
			// delete NN hopping except for edges
			if (l>=2)
			{
				res(l,  l+2) = 0.;
				res(l,  l+3) = 0.;
				res(l+1,l+2) = 0.;
				res(l+1,l+3) = 0.;
			}
		}
//		res(0,2*L-2) = res(0,2);
//		res(0,2*L-1) = res(0,3);
//		res(1,2*L-2) = res(1,2);
//		res(1,2*L-1) = res(1,3);
	}
//	cout << "PBC:" << endl;
//	cout << endl << res << endl << endl;
	
	res.matrix() += res.matrix().adjoint().eval();
	
	return res;
}

template<typename Scalar>
Array<Scalar,Dynamic,Dynamic> hopping_PAM_T (int L, Scalar tfc, Scalar tcc, Scalar tff, Scalar tx, Scalar ty, bool ANCILLA_HOPPING=false, double bugfix=1e-7)
{
	Array<Scalar,Dynamic,Dynamic> res_tmp = hopping_PAM(L/2,tfc,tcc,tff,tx,ty);
	Array<Scalar,Dynamic,Dynamic> res(2*L,2*L); res = 0;
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(2*i,2*j) = res_tmp(i,j);
		if (ANCILLA_HOPPING)
		{
			res(2*i+1,2*j+1) = res_tmp(i,j);
		}
	}
	
	for (int i=0; i<2*L-1; ++i)
	{
		res(i,i+1) += bugfix;
		res(i+1,i) += bugfix;
	}
	
//	cout << res.real() << endl;
//	cout << endl;
//	cout << res.imag() << endl;
	return res;
}

ArrayXXd hopping_spinChain (int L, double JA, double JB, double JpA, double JpB, bool ANCILLA_HOPPING=false, bool PBC=false)
{
	ArrayXXd res_tmp(L,L);
	
	if (PBC)
	{
		lout << termcolor::yellow << "Warning: ignoring JB, JpB for PBC!" << termcolor::reset << endl;
		res_tmp = create_1D_PBC(L,JA,JpA);
	}
	else
	{
		res_tmp.setZero();
		res_tmp.matrix().diagonal<1> ()(Eigen::seq(0,Eigen::last,2)).setConstant(JA);
		res_tmp.matrix().diagonal<1> ()(Eigen::seq(1,Eigen::last,2)).setConstant(JB);
		res_tmp.matrix().diagonal<2> ()(Eigen::seq(0,Eigen::last,2)).setConstant(JpA);
		res_tmp.matrix().diagonal<2> ()(Eigen::seq(1,Eigen::last,2)).setConstant(JpB);
		res_tmp += res_tmp.transpose().eval();
	}
	return res_tmp;
}

ArrayXXd hopping_spinChain_T (int L, double JA, double JB, double JpA, double JpB, bool ANCILLA_HOPPING=false, double bugfix=1e-7, bool PBC=false)
{
	ArrayXXd res_tmp;
	if (PBC)
	{
		lout << termcolor::yellow << "Warning: ignoring JB, JpB for PBC!" << termcolor::reset << endl;
		res_tmp = create_1D_PBC(L,JA,JpA);
	}
	else
	{
		res_tmp = hopping_spinChain(L,JA,JB,JpA,JpB,false);
	}
	
	ArrayXXd res(2*L,2*L); res = 0;
	
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(2*i,2*j) = res_tmp(i,j);
		if (ANCILLA_HOPPING)
		{
			res(2*i+1,2*j+1) = res_tmp(i,j);
		}
	}
	
	for (int i=0; i<2*L-1; ++i)
	{
		res(i,i+1) += bugfix;
		res(i+1,i) += bugfix;
	}
	
	return res;
}

ArrayXXd hopping_ladder (int L, double tPara=1., double tPerp=1., double tPrime=0., double tPPrime=0., bool PBC=false, bool BABA=false)
{
	ArrayXXd res(L,L);
	res.setZero();
	if (!PBC)
	{
		if (!BABA)
		{
			for (int l=0; l<L; ++l)
			{
				if (l+1<L) res(l,l+1) = (l%2==0)? tPerp:0.;
				if (l+2<L) res(l,l+2) = tPara;
				
				if (l%2==0 and l+3<L) res(l,l+3) = tPrime;
				if (l%2==1 and l+1<L) res(l,l+1) = tPrime;
				
				if (l+4<L) res(l,l+4) = tPPrime;
			}
		}
		else
		{
			for (int l=0; l<L; ++l)
			{
				if (l+1<L) res(l,l+1) = (l%2==1)? tPerp:0.;
				if (l+2<L) res(l,l+2) = tPara;
				
				if (l%2==0 and l+1<L) res(l,l+1) = tPrime;
				if (l%2==1 and l+3<L) res(l,l+3) = tPrime;
				
//				// different enumeration, doesn't seem to be correct:
//				if (l+1<L) res(l,l+1) = (l%4==0 or l%4==2)? tPara:0.;
//				if (l+3<L) res(l,l+3) = (l%4==1 or l%4==3)? tPara:0.;
//				if (l+1<L) res(l,l+1) = (l%4==1 or l%4==3)? tPerp:0.;
//				if (l+2<L) res(l,l+2) = tPrime;
			}
		}
	}
	else
	{
		for (int l=0; l<L; ++l)
		{
			res(l,(l+1)%L) = (l%2==0)? tPerp:0.;
			res(l,(l+2)%L) = tPara;
			
			if (l%2==0) res(l,(l+3)%L) = tPrime;
			if (l%2==1) res(l,(l+1)%L) = tPrime;
			
			res(l,(l+4)%L) = tPPrime;
		}
	}
	res += res.transpose().eval();
	return res;
}

tuple<double,double,double> params_bilineraBiquadratic_beta (double beta, double J_input=1.)
{
	double J = J_input+0.5*beta;
	double R = -0.5*beta;
	double offset = -4./3.*beta;
	return make_tuple(J,R,offset);
}

// returns J, R and offset for bilinear-biquadratic Hamiltonian with J1=1 and J2=beta
// H = J*S_i*S_{i+1} - beta*(S_i*S_{i+1})^2
//   = (J+beta/2)*S_i*S_{i+1} - beta/2 Q_i*Q_{i+1}
tuple<double,double,double> params_bilineraBiquadratic_beta (boost::rational<int> beta_rational = boost::rational<int>(-1,3), double J_input=1.)
{
//	double beta = boost::rational_cast<double>(beta_rational);
//	double J = J_input+0.5*beta;
//	double R = -0.5*beta;
//	double offset = -4./3.*beta;
//	return make_tuple(J,R,offset);
	return params_bilineraBiquadratic_beta(boost::rational_cast<double>(beta_rational), J_input);
}

// H = cos(theta)*S_i*S_{i+1} + sin(theta)*(S_i*S_{i+1})^2
tuple<double,double,double> params_bilineraBiquadratic_theta (double theta)
{
	double beta = -sin(theta);
	double J = cos(theta)+0.5*beta;
	double R = -0.5*beta;
	double offset = -4./3.*beta;
	return make_tuple(J,R,offset);
}

#endif
