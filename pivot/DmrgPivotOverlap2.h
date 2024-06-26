#ifndef DMRGPIVOTOVERLAP2
#define DMRGPIVOTOVERLAP2

//#include "pivot/DmrgPivotOverlap1.h"
#include "pivot/DmrgPivotVector.h"

template<typename Symmetry, typename Scalar>
struct PivotOverlap2
{
	PivotOverlap2(){};
	PivotOverlap2(const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L_input, 
	              const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R_input, 
	              const vector<qarray<Symmetry::Nq> > &qloc1_input,
	              const vector<qarray<Symmetry::Nq> > &qloc2_input)
	:L(L_input), R(R_input), qloc1(qloc1_input), qloc2(qloc2_input)
	{};
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	
	vector<qarray<Symmetry::Nq> > qloc1;
	vector<qarray<Symmetry::Nq> > qloc2;
};

template<typename Symmetry, typename Scalar>
void LRxV (const PivotOverlap2<Symmetry,Scalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
//	auto tensor_basis = Symmetry::tensorProd(H.qloc1, H.qloc2);
	Vout.outerResize(Vin);
	
	Qbasis<Symmetry> loc1; loc1.pullData(H.qloc1);
	Qbasis<Symmetry> loc2; loc2.pullData(H.qloc2);
	Qbasis<Symmetry> tensor_basis = loc1.combine(loc2);
	
	for (size_t s1=0; s1<H.qloc1.size(); ++s1)
	for (size_t s2=0; s2<H.qloc2.size(); ++s2)
	{
		auto qmerges12 = Symmetry::reduceSilent(H.qloc1[s1], H.qloc2[s2]);
		
		for (const auto &qmerge12:qmerges12)
		{
//			auto qtensor12 = make_tuple(H.qloc1[s1], s1, H.qloc2[s2], s2, qmerge12);
//			auto s1s2 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor12));
			size_t s1s2 = tensor_basis.outer_num(qmerge12) + tensor_basis.leftOffset(qmerge12,{H.qloc1[s1],H.qloc2[s2]},{loc1.inner_num(s1),loc2.inner_num(s2)});
			
			for (size_t qL=0; qL<H.L.dim; ++qL)
			{
				vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > ixs;
				bool FOUND_MATCH = AAAA(H.L.in[qL], H.L.out[qL], s1s2, qmerge12, Vout.data, Vin.data, ixs);
				
				if (FOUND_MATCH)
				{
					for (const auto &ix:ixs)
					{
						auto qR = H.R.dict.find(get<0>(ix));
						size_t qAbra = get<1>(ix);
						size_t qAket = get<2>(ix);
						
						if (qR != H.R.dict.end())
						{
							Matrix<Scalar,Dynamic,Dynamic> Mtmp;
							
							if (H.L.block[qL].size() != 0 and
							    H.R.block[qR->second].size() !=0 and
							    Vin.data[s1s2].block[qAket].size() != 0)
							{
								if (H.L.block[qL].cols() == Vin.data[s1s2].block[qAket].rows() and
								    Vin.data[s1s2].block[qAket].cols() == H.R.block[qR->second].rows())
								{
									optimal_multiply(1., 
									                 H.L.block[qL],
									                 Vin.data[s1s2].block[qAket],
									                 H.R.block[qR->second],
									                 Mtmp);
								}
								else
								{
									lout << termcolor::red << "Warning: Mismatching matrix dimensions in LRxV (PivotOverlap2)!" << termcolor::reset << endl;
									print_size(H.L.block[qL],"H.L.block[qL]");
									print_size(Vin.data[s1s2].block[qAket],"Vin.data[s1s2].block[qAket]");
									print_size(H.R.block[qR->second],"H.R.block[qR->second]");
									lout << "qnums: " << H.L.in[qL] << "-" << H.L.out[qL] << ", " 
												      << Vin.data[s1s2].in[qAket] << "-" << Vin.data[s1s2].out[qAket] << ", " 
												      << H.R.in[qR->second] << "-" << H.R.out[qR->second]
												      << endl;
									lout << endl;
//									
//									lout << H.L.print() << endl;
//									lout << H.R.print() << endl;
								}
							}
							
							if (Mtmp.size() != 0)
							{
								if (Vout.data[s1s2].block[qAbra].rows() == Mtmp.rows() and
									Vout.data[s1s2].block[qAbra].cols() == Mtmp.cols())
								{
									Vout.data[s1s2].block[qAbra] += Mtmp;
								}
								else
								{
									Vout.data[s1s2].block[qAbra] = Mtmp;
								}
							}
						}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar>
void LRxV (const PivotOverlap2<Symmetry,Scalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	LRxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif
