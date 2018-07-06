#ifndef VUMPSPIVOTSTUFF
#define VUMPSPIVOTSTUFF

#include "tensors/Biped.h"
#include "pivot/DmrgPivotVector.h"

//-----------<definitions>-----------

/**Structure to update \f$A_C\f$ (eq. 11) with a 2-site Hamiltonian. Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix1
{
	PivumpsMatrix1(){};
	
	// Produces an error with boost::multi_array!
//	PivumpsMatrix (const Matrix<Scalar,Dynamic,Dynamic> &L_input,
//	               const Matrix<Scalar,Dynamic,Dynamic> &R_input,
//	               const boost::multi_array<MpoScalar,4> &h_input,
//	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AL_input,
//	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AR_input)
//	:L(L_input), R(R_input), AL(AL_input), AR(AR_input)
//	{
//		size_t D = h_input.shape()[0];
//		h.resize(boost::extents[D][D][D][D]);
//		h = h_input;
//	}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	
	boost::multi_array<MpoScalar,4> h;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AR;
	
	vector<qarray<Symmetry::Nq> > qloc;
};

/**Structure to update \f$C\f$ (eq. 16) with a 2-site Hamiltonian. Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix0
{
	PivumpsMatrix0(){};
	
	PivumpsMatrix0 (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H)
	:L(H.L), R(H.R), h(H.h), AL(H.AL), AR(H.AR), qloc(H.qloc)
	{}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	
	boost::multi_array<MpoScalar,4> h;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AR;
	
	vector<qarray<Symmetry::Nq> > qloc;
};
//-----------</definitions>-----------

/**Performs the local update of \f$A_C\f$ (eq. 11) with a 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	size_t D = H.h.shape()[0];
	
	Vout.outerResize(Vin);
	Vout.setZero();
	
	// term 1 AL
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			for (size_t q1=0; q1<H.AL[s1].dim; ++q1)
			{
				auto A2outs = Symmetry::reduceSilent(H.AL[s1].in[q1], H.qloc[s2]);
				for (const auto &A2out : A2outs)
				{
					auto it2 = H.AL[s2].dict.find(qarray2<Symmetry::Nq>{H.AL[s1].in[q1], A2out});
					if (it2 != H.AL[s2].dict.end())
					{
						auto A4outs = Symmetry::reduceSilent(H.AL[s2].out[it2->second], H.qloc[s4]);
						for (const auto &A4out : A4outs)
						{
							auto it4 = Vin.data[s4].dict.find(qarray2<Symmetry::Nq>{H.AL[s2].out[it2->second], A4out});
							if (it4 != H.AL[s4].dict.end())
							{
								Matrix<Scalar,Dynamic,Dynamic> Mtmp;
								optimal_multiply(H.h[s1][s2][s3][s4],
								                 H.AL[s1].block[q1].adjoint(),
								                 H.AL[s2].block[it2->second],
								                 Vin.data[s4].block[it4->second],
								                 Mtmp
								                );
								
								if (Mtmp.size() != 0)
								{
									qarray2<Symmetry::Nq> quple = {H.AL[s1].out[q1], H.AL[s4].out[it4->second]};
									auto it = Vout.data[s3].dict.find(quple);
									
									if (it != Vout.data[s3].dict.end())
									{
										if (Vout.data[s3].block[it->second].rows() != Mtmp.rows() and
											Vout.data[s3].block[it->second].cols() != Mtmp.cols())
										{
											Vout.data[s3].block[it->second] = Mtmp;
										}
										else
										{
											Vout.data[s3].block[it->second] += Mtmp;
										}
									}
									else
									{
										cout << termcolor::red << "pushback that shouldn't be: PivumpsMatrix1 term 1" << termcolor::reset << endl;
										Vout.data[s3].push_back(quple, Mtmp);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	// term 2 AR
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			for (size_t q2=0; q2<Vin.data[s2].dim; ++q2)
			{
				auto A4outs = Symmetry::reduceSilent(Vin.data[s2].out[q2], H.qloc[s4]);
				for (const auto &A4out : A4outs)
				{
					auto it4 = H.AR[s4].dict.find(qarray2<Symmetry::Nq>{Vin.data[s2].out[q2], A4out});
					if (it4 != H.AR[s4].dict.end())
					{
						auto A3ins = Symmetry::reduceSilent(H.AR[s4].out[it4->second], Symmetry::flip(H.qloc[s3]));
						for (const auto &A3in : A3ins)
						{
							auto it3 = H.AR[s3].dict.find(qarray2<Symmetry::Nq>{A3in, H.AR[s4].out[it4->second]});
							if (it3 != H.AR[s3].dict.end())
							{
								Matrix<Scalar,Dynamic,Dynamic> Mtmp;
								optimal_multiply(H.h[s1][s2][s3][s4],
								                 Vin.data[s2].block[q2],
								                 H.AR[s4].block[it4->second],
								                 H.AR[s3].block[it3->second].adjoint(),
								                 Mtmp
								                );
								
								if (Mtmp.size() != 0)
								{
									qarray2<Symmetry::Nq> quple = {Vin.data[s2].in[q2], H.AR[s3].in[it3->second]};
									auto it = Vout.data[s1].dict.find(quple);
									
									if (it != Vout.data[s1].dict.end())
									{
										if (Vout.data[s1].block[it->second].rows() != Mtmp.rows() and
											Vout.data[s1].block[it->second].cols() != Mtmp.cols())
										{
											Vout.data[s1].block[it->second] = Mtmp;
										}
										else
										{
											Vout.data[s1].block[it->second] += Mtmp;
										}
									}
									else
									{
										cout << termcolor::red << "pushback that shouldn't be: PivumpsMatrix1 term 2" << termcolor::reset << endl;
										Vout.data[s1].push_back(quple, Mtmp);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	for (size_t s=0; s<D; ++s)
	{
		auto Vtmp = H.L.contract(Vin.data[s]);
		Vout[s] += Vtmp;
		
		Vtmp.clear();
		Vtmp = Vin.data[s].contract(H.R);
		Vout[s] += Vtmp;
	}
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

/**Performs the local update of \f$C\f$ (eq. 16) with an explicit 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	size_t D = H.h.shape()[0];
	
	Vout.outerResize(Vin);
	Vout.setZero();
	
	// term 1
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[s1][s2][s3][s4] != 0.)
		{
			for (size_t q1=0; q1<H.AL[s1].dim; ++q1)
			{
				auto A2outs = Symmetry::reduceSilent(H.AL[s1].in[q1], H.qloc[s2]);
				for (const auto &A2out : A2outs)
				{
					qarray2<Symmetry::Nq> qupleC = {A2out, A2out};
					auto itC = Vin.data[0].dict.find(qupleC);
					auto it2 = H.AL[s2].dict.find(qarray2<Symmetry::Nq>{H.AL[s1].in[q1], A2out});
					
					if (it2 != H.AL[s2].dict.end() and itC != Vin.data[0].dict.end())
					{
						auto A4outs = Symmetry::reduceSilent(H.AL[s2].out[it2->second], H.qloc[s4]);
						for (const auto &A4out : A4outs)
						{
							auto it4 = H.AR[s4].dict.find(qarray2<Symmetry::Nq>{H.AL[s2].out[it2->second], A4out});
							if (it4 != H.AR[s4].dict.end())
							{
								auto A3ins = Symmetry::reduceSilent(H.AR[s4].out[it4->second], Symmetry::flip(H.qloc[s3]));
								for (const auto &A3in : A3ins)
								{
									auto it3 = H.AR[s3].dict.find(qarray2<Symmetry::Nq>{A3in, H.AR[s4].out[it4->second]});
									if (it3 != H.AR[s3].dict.end())
									{
										Matrix<Scalar,Dynamic,Dynamic> Mtmp;
										Mtmp = H.h[s1][s2][s3][s4] *
										       H.AL[s1].block[q1].adjoint() * 
										       H.AL[s2].block[it2->second] *
										       Vin.data[0].block[itC->second] *
										       H.AR[s4].block[it4->second] * 
										       H.AR[s3].block[it3->second].adjoint();
										
										if (Mtmp.size() != 0)
										{
											qarray2<Symmetry::Nq> quple = {H.AL[s1].out[q1], H.AR[s3].in[it3->second]};
											auto it = Vout.data[0].dict.find(quple);
											
											if (it != Vout.data[0].dict.end())
											{
												if (Vout.data[0].block[it->second].rows() != Mtmp.rows() and
													Vout.data[0].block[it->second].cols() != Mtmp.cols())
												{
													Vout.data[0].block[it->second] = Mtmp;
												}
												else
												{
													Vout.data[0].block[it->second] += Mtmp;
												}
											}
											else
											{
												cout << termcolor::red << "pushback that shouldn't be: HxC term 1" << termcolor::reset << endl;
												cout << "q=" << quple[0] << ", " << quple[1] << endl;
												Vout.data[0].push_back(quple, Mtmp);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	// term 2 HL
	auto Vtmp = H.L.contract(Vin.data[0]);
	Vout.data[0] += Vtmp;
	
	// term 2 HR
	Vtmp.clear();
	Vtmp = Vin.data[0].contract(H.R);
	Vout.data[0] += Vtmp;
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivumpsMatrix1<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivumpsMatrix0<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

#endif
