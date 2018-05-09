#ifndef DMRGPIVOTOVERLAP1
#define DMRGPIVOTOVERLAP1

#include "DmrgTypedefs.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "pivot/DmrgPivotVector.h"

template<typename Symmetry, typename Scalar>
struct PivotOverlap1
{
	PivotOverlap1(){};
	PivotOverlap1(const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L_input, 
	              const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R_input, 
	              const vector<qarray<Symmetry::Nq> > &qloc_input)
	:L(L_input), R(R_input), qloc(qloc_input)
	{};
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	
	vector<qarray<Symmetry::Nq> > qloc;
};

template<typename Symmetry, typename Scalar>
void LRxV (const PivotOverlap1<Symmetry,Scalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
// 	Vout = Vin;
// 	for (size_t i=0; i<Vout.data.size(); ++i)
// 	{
// 		Vout.data[i].setZero();
// 	}
	Vout.outerResize(Vin);
	
	for (size_t s=0; s<H.qloc.size(); ++s)
	for (size_t qL=0; qL<H.L.dim; ++qL)
	{
		vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > ix;
		bool FOUND_MATCH = AA(H.L.in[qL], H.L.out[qL], s, H.qloc, Vout.data, Vin.data, ix);
		
		if (FOUND_MATCH)
		{
			for (size_t n=0; n<ix.size(); ++n)
			{
				size_t qAbra = get<1>(ix[n]);
				size_t qAket = get<2>(ix[n]);
				auto qR = H.R.dict.find(get<0>(ix[n]));
				
				if (qR != H.R.dict.end() and
				    H.L.block[qL].size() != 0 and
				    Vin.data[s].block[qAket].size() != 0 and
				    H.R.block[qR->second].size() != 0)
				{
					Matrix<Scalar,Dynamic,Dynamic> Mtmp;
					
					optimal_multiply(1.,
					                 H.L.block[qL],
					                 Vin.data[s].block[qAket],
					                 H.R.block[qR->second],
					                 Mtmp);
					
					if (Mtmp.size() != 0)
					{
						if (Vout.data[s].block[qAbra].size() != 0)
						{
							Vout.data[s].block[qAbra] += Mtmp;
						}
						else
						{
							Vout.data[s].block[qAbra] = Mtmp;
						}
					}
				}
			}
		}
	}
}

#endif
