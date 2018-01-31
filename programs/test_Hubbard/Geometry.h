#ifndef MYGEOMETRY
#define MYGEOMETRY

#include <set>

#include "HilbertTypedefs.h"
#include "NestedLoopIterator.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;

class Geometry
{
public:
	
	Geometry(){};
	
	MatrixXi neighbours (int order, int L_edge, DIM spacedim=DIM1, BOUNDARY_CONDITION BC_input=BC_PERIODIC);
	
	SparseMatrixXd BondMatrix (int L_edge, DIM spacedim, BOUNDARY_CONDITION BC_input, VectorXd hoppings, BOND_STORAGE BS_input);
	SparseMatrixXd BondMatrix (int L_edge, DIM spacedim, BOUNDARY_CONDITION BC_input, double hopping,    BOND_STORAGE BS_input);
};

MatrixXi Geometry::
neighbours (int order, int L_edge, DIM spacedim, BOUNDARY_CONDITION BC_input)
{
	int Volume = pow(L_edge,(int)spacedim);
	
	if (order == 0)
	{
		MatrixXi Mout(Volume,1);
		for (int i=0; i<Mout.rows(); ++i)
		{
			Mout(i,0) = i;
		}
		return Mout;
	}
	// adapted C-code from computational physics course at Humboldt University (Prof. Wolff)
	else if (order>0 and spacedim==1 or order==1 and spacedim>1)
	{
		MatrixXi Mout(Volume,2*(int)spacedim);
		for (int isite=0; isite<Volume; ++isite)
		{
			int Lmu = Volume;
			int rx = isite;
			for (int mu=(int)spacedim-1; mu>=0; --mu)
			{
				Lmu = Lmu/L_edge; // = L^mu
				int xmu = rx/Lmu; // get mu-th component
				rx = rx-xmu*Lmu; // rest index from x_0 .. x_(mu-1)
				
				// forwards, modulo because of overflow
				int xmuP = (xmu+order)%L_edge;
				if (BC_input == BC_PERIODIC)
				{
					Mout(isite,mu) = isite+(xmuP-xmu)*Lmu;
				}
				else if (BC_input == BC_DANGLING)
				{
					if (xmuP-xmu<0) {Mout(isite,mu) = -1;}
					else            {Mout(isite,mu) = isite+(xmuP-xmu)*Lmu;}
				}
				
				// backwards, +L_edge for correct overflow
				int xmuM = (xmu-order+L_edge)%L_edge;
				if (BC_input == BC_PERIODIC)
				{
					Mout(isite,mu+(int)spacedim) = isite+(xmuM-xmu)*Lmu;
				}
				else if (BC_input == BC_DANGLING)
				{
					if (xmuM-xmu>0) {Mout(isite,mu+(int)spacedim) = -1;}
					else            {Mout(isite,mu+(int)spacedim) = isite+(xmuM-xmu)*Lmu;}
				}
			}
		}
		return Mout;
	}
	// my code
	else
	{
		vector<multimap<double,size_t> > dist_to_site(Volume);
		set<double> distances;
		
		for (size_t isite=0; isite<Volume; ++isite)
		{
			NestedLoopIterator Nina(spacedim,L_edge); Nina = isite;
			NestedLoopIterator Nelly(spacedim,L_edge);
			
			for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
			{
				VectorXi ri((int)spacedim);
				for (int l=0; l<spacedim; ++l) ri(l) = Nina(l);
				VectorXi rj((int)spacedim);
				for (int l=0; l<spacedim; ++l) rj(l) = Nelly(l);
				
				VectorXd rdiff((int)spacedim);
				rdiff.setZero();
				for (int l=0; l<spacedim; ++l)
				{
					if (BC_input == BC_PERIODIC)
					{
						rdiff(l) = min(min( abs((ri(l)+L_edge-rj(l))%L_edge) , abs((ri(l)-L_edge-rj(l))%L_edge) ),
						               abs((ri(l)-rj(l))%L_edge)
						              );
					}
					else if (BC_input == BC_DANGLING)
					{
						rdiff(l) = ri(l)-rj(l);
					}
				}
				double dist = rdiff.norm();
				distances.insert(dist);
				
				dist_to_site[isite].insert(pair<double,size_t>(dist,*Nelly));
			}
		}
		
		assert(order <= distances.size() and "Geometry: Too small a lattice for that kind of hopping!");
		auto Delta = distances.begin();
		advance(Delta,order);
		int N_neighbours = dist_to_site[0].count(*Delta);
		
		vector<vector<int> > neighbour_list(Volume);
		for (size_t isite=0; isite<Volume; ++isite)
		{
			for (auto it=dist_to_site[isite].begin(); it!=dist_to_site[isite].end(); ++it)
			{
				if (it->first == *Delta)
				{
					neighbour_list[isite].push_back(it->second);
				}
			}
		}
		
		MatrixXi Mout(Volume,N_neighbours);
		Mout.setConstant(-1);
		for (size_t isite=0; isite<Volume; ++isite)
		for (size_t nsite=0; nsite<N_neighbours; ++nsite)
		{
			Mout(isite,nsite) = neighbour_list[isite][nsite];
		}
		return Mout;
	}
}

SparseMatrixXd Geometry::
BondMatrix (int L_edge, DIM spacedim, BOUNDARY_CONDITION BC_input, VectorXd hoppings, BOND_STORAGE BS_input)
{
	if (spacedim == DIM1)
	{
		if (BC_input == BC_PERIODIC)
		{
			assert(L_edge > 2*hoppings.rows() and "Geometry: Too small a lattice for that kind of hopping!");
		}
		else if (BC_input == BC_DANGLING)
		{
			assert(L_edge > hoppings.rows() and "Geometry: Too small a lattice for that kind of hopping!");
		}
	}
	
	int Volume = pow(L_edge,static_cast<int>(spacedim));
	SparseMatrixXd Mout(Volume,Volume);
	
	for (int i=0; i<hoppings.rows(); ++i)
	{
		if (hoppings(i)!=0.)
		{
			SparseMatrixXd contribution(Volume,Volume);
			MatrixXi bonds = neighbours(i+1,L_edge,spacedim,BC_input);
			
			int kmin, kmax;
			if      (BS_input == BS_NO)       {kmin=0; kmax=0;}
			else if (BS_input == BS_FORWARDS) {kmin=0; kmax=bonds.cols()/2;}
//			else if (BS_input == BS_BACKWARDS){kmin=bonds.cols()/2; kmax=bonds.cols();}
			else                              {kmin=0; kmax=bonds.cols();}
			
			for (int j=0; j<bonds.rows(); ++j)
			for (int k=kmin; k<kmax; ++k)
			{
				if (bonds(j,k)!=-1)
				{
					if (BS_input == BS_UPPER)
					{
						if (j<bonds(j,k))
						{
							contribution.insert(j,bonds(j,k)) = hoppings(i);
						}
					}
//					else if (BS_input == BS_LOWER)
//					{
//						if (j>bonds(j,k))
//						{
//							contribution.insert(j,bonds(j,k)) = hoppings(i);
//						}
//					}
					else
					{
						contribution.insert(j,bonds(j,k)) = hoppings(i);
					}
				}
			}
			Mout += contribution;
		}
	}
	return Mout;
}

inline SparseMatrixXd Geometry::
BondMatrix (int L_edge, DIM spacedim, BOUNDARY_CONDITION BC_input, double hopping, BOND_STORAGE BS_input)
{
	VectorXd hoppings(1);
	hoppings(0) = hopping;
	return BondMatrix(L_edge,spacedim,BC_input,hoppings,BS_input);
}

#endif
