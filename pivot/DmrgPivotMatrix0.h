#ifndef STRAWBERRY_DMRG_HEFF_STUFF_0SITE_WITH_Q
#define STRAWBERRY_DMRG_HEFF_STUFF_0SITE_WITH_Q

//include "DmrgExternal.h"
//include "tensors/Biped.h"
#include "pivot/DmrgPivotMatrix1.h"
//include "pivot/DmrgPivotVector.h"

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix0
{
	PivotMatrix0(){};
	
	PivotMatrix0 (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L_input,
	              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R_input)
	:L(L_input), R(R_input)
	{}
	
	PivotMatrix0 (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H)
	:L(H.L), R(H.R)
	{}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
};

//-----------<matrix*vector>-----------
/**Calculates the following contraction:
\dotfile HxV_0site.dot*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix0<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	Vout.outerResize(Vin);
	Vout.data[0].setZero();
	
	for (size_t qL=0; qL<H.L.dim; ++qL)
	{
		qarray3<Symmetry::Nq> qupleR = {H.L.out(qL), H.L.in(qL), H.L.mid(qL)};
		auto qR = H.R.dict.find(qupleR);
		
		if (qR != H.R.dict.end())
		{
			qarray2<Symmetry::Nq> qupleAin = {H.L.out(qL), H.L.out(qL)};
			auto qAin = Vin.data[0].dict.find(qupleAin);
			
			if (qAin != Vin.data[0].dict.end())
			{
				qarray2<Symmetry::Nq> qupleAout = {H.R.out(qR->second), H.R.out(qR->second)};
				auto qAout = Vout.data[0].dict.find(qupleAout);
				
				if (qAout != Vout.data[0].dict.end())
				{
					assert(H.L.block[qL].shape()[0] == H.R.block[qR->second].shape()[0]);
					for (size_t a=0; a<H.L.block[qL].shape()[0]; ++a)
					{
						Matrix<Scalar,Dynamic,Dynamic> Mtmp;
						
						if (H.L.block[qL][a][0].size() != 0 and
						    H.R.block[qR->second][a][0].size() !=0)
						{
							// print_size(H.L.block[qL][a][0], "H.L.block[qL][a][0]");
							// print_size(Vin.data[0].block[qAin->second], "Vin.data[0].block[qAin->second]");
							// print_size(H.R.block[qR->second][a][0], "H.R.block[qR->second][a][0]");
							optimal_multiply(1., 
							                 H.L.block[qL][a][0],
							                 Vin.data[0].block[qAin->second],
							                 H.R.block[qR->second][a][0],
							                 Mtmp);
						}
						
//						Scalar norm = Mtmp.norm();
//						if (norm > 0)
//						{
//							cout << "q=" << qupleAout[0] << ", a=" << a << ", Mtmp.norm()=" << Mtmp.norm() << endl;
//						}
						
						if (Mtmp.size() != 0)
						{
							if (Vout.data[0].block[qAout->second].size() != 0)
							{
//								cout << "adding L: a=" << a << ", q=" << H.L.out(qL) << ", " << H.L.in(qL) << ", " << H.L.mid(qL) << endl;
								Vout.data[0].block[qAout->second] += Mtmp;
							}
							else
							{
//								cout << "adding L: a=" << a << ", q=" << H.L.out(qL) << ", " << H.L.in(qL) << ", " << H.L.mid(qL) << endl;
								Vout.data[0].block[qAout->second] = Mtmp;
							}
						}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix0<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivotMatrix0<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivotMatrix0<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

#endif
