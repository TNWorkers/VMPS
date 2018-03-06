#ifndef STRAWBERRY_DMRGINDEXGYMNASTICS
#define STRAWBERRY_DMRGINDEXGYMNASTICS

#include <tuple>

#include "symmetry/qarray.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "numeric_limits.h" // from HELPERS

/**
 * Calculates the matching right indices when contracting a left transfer matrix with two MpsQ and an MpoQ.
 * \dotfile AWA.dot
 * \param Lin
 * \param Lout
 * \param Lmid
 * \param s1
 * \param s2
 * \param qloc : local basis
 * \param k
 * \param qOp : operator basis
 * \param Abra
 * \param Aket
 * \param result : tuple of: an array with \p Rin, \p Rout, \p Rmid; block index of \p Abra; block index of \p Aket
 * \returns \p true if a match is found, \p false if not
 * \warning When using this function to create the left block on the next site, one needs to swap \p Rin and \p Rout.
 */
template<typename Symmetry, typename MatrixType>
bool AWA (qarray<Symmetry::Nq> Lin, qarray<Symmetry::Nq> Lout, qarray<Symmetry::Nq> Lmid,
		  size_t s1, size_t s2, vector<qarray<Symmetry::Nq> > qloc,
		  size_t k, vector<qarray<Symmetry::Nq> > qOp,
          const vector<Biped<Symmetry,MatrixType> > &Abra, 
          const vector<Biped<Symmetry,MatrixType> > &Aket, 
          vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > &result)
{
	bool out = false;
	result.clear();
	
	auto qRouts = Symmetry::reduceSilent(Lin,qloc[s1]);
	for (const auto& qRout : qRouts)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
		auto q1 = Abra[s1].dict.find(cmp1);
		if (q1 != Abra[s1].dict.end())
		{
			auto qRins = Symmetry::reduceSilent(Lout,qloc[s2]);
			for (const auto &qRin:qRins)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
				auto q2 = Aket[s2].dict.find(cmp2);
				if (q2 != Aket[s2].dict.end())
				{
					auto qRmids = Symmetry::reduceSilent(Lmid,qOp[k]);
					for (const auto &qRmid:qRmids)
					{
						result.push_back(make_tuple(qarray3<Symmetry::Nq>{qRin,qRout,qRmid}, q1->second, q2->second));
						out = true;
					}
				}
			}
		}
	}
	return out;
}

template<typename Symmetry, typename MatrixType>
bool AA (qarray<Symmetry::Nq> Lin,
         qarray<Symmetry::Nq> Lout,
		 size_t s, 
		 vector<qarray<Symmetry::Nq> > qloc,
         const vector<Biped<Symmetry,MatrixType> > &Abra, 
         const vector<Biped<Symmetry,MatrixType> > &Aket, 
         vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > &result)
{
	bool out = false;
	result.clear();
	
	auto qRouts = Symmetry::reduceSilent(Lin,qloc[s]);
	for (const auto& qRout : qRouts)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
		auto q1 = Abra[s].dict.find(cmp1);
		if (q1 != Abra[s].dict.end())
		{
			auto qRins = Symmetry::reduceSilent(Lout,qloc[s]);
			for (const auto &qRin:qRins)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
				auto q2 = Aket[s].dict.find(cmp2);
				if (q2 != Aket[s].dict.end())
				{
					result.push_back(make_tuple(qarray2<Symmetry::Nq>{qRin,qRout}, q1->second, q2->second));
					out = true;
				}
			}
		}
	}
	return out;
}

//template<typename Symmetry, typename MatrixType>
//bool AAWWAA (qarray<Symmetry::Nq> Lin, qarray<Symmetry::Nq> Lout, qarray<Symmetry::Nq> Lmid, 
//             size_t s1, size_t s2, vector<qarray<Symmetry::Nq> > qloc12, size_t k12, vector<qarray<Symmetry::Nq> > qOp12,
//             size_t s3, size_t s4, vector<qarray<Symmetry::Nq> > qloc34, size_t k34, vector<qarray<Symmetry::Nq> > qOp34,
//             const vector<Biped<Symmetry,MatrixType> > &AA13, const vector<Biped<Symmetry,MatrixType> > &AA24, 
//             vector<tuple<qarray3<Symmetry::Nq>,qarray<Symmetry::Nq>,size_t,size_t> > &result)
template<typename Symmetry, typename MatrixType>
bool AAWWAA (qarray<Symmetry::Nq> Lin, qarray<Symmetry::Nq> Lout, qarray<Symmetry::Nq> Lmid, 
             size_t k12, vector<qarray<Symmetry::Nq> > qOp12, 
             size_t k34, vector<qarray<Symmetry::Nq> > qOp34,
             size_t s1s3, const qarray<Symmetry::Nq> &qmerge13, 
             size_t s2s4, const qarray<Symmetry::Nq> &qmerge24, 
             const vector<Biped<Symmetry,MatrixType> > &AA13, const vector<Biped<Symmetry,MatrixType> > &AA24, 
             vector<tuple<qarray3<Symmetry::Nq>,qarray<Symmetry::Nq>,size_t,size_t> > &result)
{
//	qarray<Symmetry::Nq> qRout = Lin + qloc12[s1] + qloc34[s3];
//	qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
//	auto q13 = AA13[s1][s3].dict.find(cmp1);
//	
//	if (q13 != AA13[s1][s3].dict.end())
//	{
//		qarray<Symmetry::Nq> qRin = Lout + qloc12[s2] + qloc34[s4];
//		qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
//		auto q24 = AA24[s2][s4].dict.find(cmp2);
//		
//		if (q24 != AA24[s2][s4].dict.end())
//		{
//			qarray<Symmetry::Nq> qRmid = Lmid + qloc12[s1] + qloc34[s3] - qloc12[s2] - qloc34[s4];
//			
//			result = make_tuple(qarray3<Symmetry::Nq>{qRin,qRout,qRmid}, q13->second, q24->second);
//			return true;
//		}
//	}
//	return false;
	
	bool out = false;
	result.clear();
	
	auto qRouts = Symmetry::reduceSilent(Lin, qmerge13);
	
	for (const auto &qRout:qRouts)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
		auto q13 = AA13[s1s3].dict.find(cmp1);
		
		if (q13 != AA13[s1s3].dict.end())
		{
			auto qRins = Symmetry::reduceSilent(Lout, qmerge24);
			
			for (const auto &qRin:qRins)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
				auto q24 = AA24[s2s4].dict.find(cmp2);
				
				if (q24 != AA24[s2s4].dict.end())
				{
//					auto qRmids = Symmetry::reduceSilent(Lmid, qOp12[k12], qOp34[k34]);
					auto qWs = Symmetry::reduceSilent(Lmid, qOp12[k12]);
					for (const auto &qW:qWs)
					{
						auto qRmids = Symmetry::reduceSilent(qW, qOp34[k34]);
						for (const auto &qRmid:qRmids)
						{
							if (Symmetry::validate(qarray3<Symmetry::Nq>{qRin,qRmid,qRout}))
							{
								result.push_back(make_tuple(qarray3<Symmetry::Nq>{qRin,qRout,qRmid}, qW, q13->second, q24->second));
								out = true;
							}
						}
					}
				}
			}
		}
	}
	return out;
}

template<typename Symmetry, typename MatrixType>
bool AAAA (qarray<Symmetry::Nq> Lin, qarray<Symmetry::Nq> Lout, 
           size_t s1s2, const qarray<Symmetry::Nq> &qmerge12, 
           const vector<Biped<Symmetry,MatrixType> > &AAbra,
           const vector<Biped<Symmetry,MatrixType> > &AAket, 
           vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > &result)
{	
	bool out = false;
	result.clear();
	
	auto qRouts = Symmetry::reduceSilent(Lin, qmerge12);
	
	for (const auto &qRout:qRouts)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
		auto qbra = AAbra[s1s2].dict.find(cmp1);
		
		if (qbra != AAbra[s1s2].dict.end())
		{
			auto qRins = Symmetry::reduceSilent(Lout, qmerge12);
			
			for (const auto &qRin:qRins)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
				auto qket = AAket[s1s2].dict.find(cmp2);
				
				if (qket != AAket[s1s2].dict.end())
				{
					result.push_back(make_tuple(qarray2<Symmetry::Nq>{qRin,qRout}, qbra->second, qket->second));
					out = true;
				}
			}
		}
	}
	return out;
}

/**
 * Calculates the matching right indices when contracting a left transfer matrix with two MpsQ and two MpoQ.
 * \dotfile AWWA.dot
 * \param Lin
 * \param Lout
 * \param Lbot
 * \param Ltop
 * \param s1
 * \param s2
 * \param s3
 * \param qloc : local basis
 * \param k1
 * \param qOpBot : operator basis of bottom operator
 * \param k2
 * \param qOpTop : operator basis of top operator
 * \param Abra
 * \param Aket
 * \param result : tuple of: an array with \p Rin, \p Rout, \p Rbot, \p Rtop; block index of \p Abra; block index of \p Aket
 * \returns \p true if a match is found, \p false if not
 * \warning When using this function to create the left block on the next site, one needs to swap \p Rin and \p Rout.
 */
template<typename Symmetry, typename MatrixType>
bool AWWA (qarray<Symmetry::Nq> Lin, qarray<Symmetry::Nq> Lout, qarray<Symmetry::Nq> Lbot, qarray<Symmetry::Nq> Ltop, 
          size_t s1, size_t s2, size_t s3, vector<qarray<Symmetry::Nq> > qloc,
		  size_t k1, vector<qarray<Symmetry::Nq> > qOpBot, size_t k2, vector<qarray<Symmetry::Nq> > qOpTop,
          const vector<Biped<Symmetry,MatrixType> > &Abra, 
          const vector<Biped<Symmetry,MatrixType> > &Aket, 
          tuple<qarray4<Symmetry::Nq>,size_t,size_t> &result)
{
	qarray<Symmetry::Nq> qRout = Lin + qloc[s1];
	qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
	auto q1 = Abra[s1].dict.find(cmp1);
	
	if (q1 != Abra[s1].dict.end())
	{
		qarray<Symmetry::Nq> qRin = Lout + qloc[s3];
		qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
		auto q2 = Aket[s3].dict.find(cmp2);
		
		if (q2 != Aket[s3].dict.end())
		{
			qarray<Symmetry::Nq> qRbot = Lbot + qloc[s1] - qloc[s2];
			qarray<Symmetry::Nq> qRtop = Ltop + qloc[s2] - qloc[s3];
			
			result = make_tuple(qarray4<Symmetry::Nq>{qRin,qRout,qRbot,qRtop}, q1->second, q2->second);
			return true;
		}
	}
	return false;
}


/**Updates the quantum Numbers of a right environment when a new site with quantum numbers qloc and qOp is added.*/
template<typename Symmetry, typename Scalar>
void updateInset (const std::vector<std::array<typename Symmetry::qType,3> > &insetOld, 
				  const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
				  const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
				  const vector<qarray<Symmetry::Nq> > &qloc,
				  const vector<qarray<Symmetry::Nq> > &qOp,
				  std::vector<std::array<typename Symmetry::qType,3> > &insetNew)
{
	std::array<typename Symmetry::qType,3> qCheck;
	Scalar factor_cgc;
	std::unordered_set<std::array<typename Symmetry::qType,3> > uniqueControl;
	
	insetNew.clear();
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		for(const auto & [qIn_old,qOut_old,qMid_old] : insetOld)
		{
			auto qRouts = Symmetry::reduceSilent(qOut_old,Symmetry::flip(qloc[s1]));
			auto qRins = Symmetry::reduceSilent(qIn_old,Symmetry::flip(qloc[s2]));
			for(const auto& qOut_new : qRouts)
				for(const auto& qIn_new : qRins)
				{
					qarray2<Symmetry::Nq> cmp1 = {qOut_new, qOut_old};
					qarray2<Symmetry::Nq> cmp2 = {qIn_new, qIn_old};
		
					auto q1 = Abra[s1].dict.find(cmp1);
					auto q2 = Aket[s2].dict.find(cmp2);

					if (q1!=Abra[s1].dict.end() and 
						q2!=Aket[s2].dict.end())
					{
						// qarray<Symmetry::Nq> new_qin  = Aket[s2].in[q2->second]; // A.in
						// qarray<Symmetry::Nq> new_qout = Abra[s1].in[q1->second]; // A†.out = A.in
						auto qRmids = Symmetry::reduceSilent(qMid_old,Symmetry::flip(qOp[k]));
						for(const auto& qMid_new : qRmids)
						{
							// qarray3<Symmetry::Nq> quple = {new_qin, new_qout, new_qmid};
							factor_cgc = Symmetry::coeff_buildR(Aket[s2].out[q2->second],qloc[s2],Aket[s2].in[q2->second],
																qMid_old,qOp[k],qMid_new,
																Abra[s1].out[q1->second],qloc[s1],Abra[s1].in[q1->second]);
							if (std::abs(factor_cgc) < ::mynumeric_limits<Scalar>::epsilon()) { continue; }
							if( auto it=uniqueControl.find({qIn_new,qOut_new,qMid_new}) == uniqueControl.end() )
							{
								uniqueControl.insert({qIn_new,qOut_new,qMid_new});
								insetNew.push_back({qIn_new,qOut_new,qMid_new});
							}
						}
					}
				}
		}
	}
}

/**Prepares a PivotMatrix by filling PivotMatrix::qlhs and PivotMatrix::qrhs with the corresponding subspace indices.
Uses OpenMP.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void precalc_blockStructure (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L, 
                             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                             const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                             const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                             const vector<qarray<Symmetry::Nq> > &qloc,
                             const vector<qarray<Symmetry::Nq> > &qOp, 
                             vector<std::array<size_t,2> > &qlhs, 
                             vector<vector<std::array<size_t,5> > > &qrhs,
                             vector<vector<Scalar> > &factor_cgcs)
{
//	Heff.W = W;
	
	unordered_map<std::array<size_t,2>, std::pair<vector<std::array<size_t,5> >, vector<Scalar> > > lookup;
	std::array<typename Symmetry::qType,3> qCheck;
	Scalar factor_cgc;
	
	#ifndef DMRG_DONT_USE_OPENMP
	#ifndef __INTEL_COMPILER
	#pragma omp parallel for collapse(3)
	#elif __INTEL_COMPILER
	#pragma omp parallel for
	#endif
	#endif
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for(size_t k=0; k<qOp.size(); ++k)
	{
		if (!Symmetry::validate(qarray3<Symmetry::Nq>{qloc[s2],qOp[k],qloc[s1]})) {continue;}
		
		for (size_t qL=0; qL<L.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = AWA(L.in(qL), L.out(qL), L.mid(qL), s1,s2, qloc, k, qOp, Abra,Aket, ix);
			
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); ++n)
				{
					auto qR = R.dict.find(get<0>(ix[n]));
					
					if (qR != R.dict.end())
					{
						bool ALL_BLOCKS_ARE_EMPTY = true;
						
						for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
						for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
						{
							if (L.block[qL][iW.row()][0].rows() != 0 and 
							    R.block[qR->second][iW.col()][0].rows() != 0)
							{
								ALL_BLOCKS_ARE_EMPTY = false;
							}
						}
						if (ALL_BLOCKS_ARE_EMPTY == false)
						{
							if constexpr ( Symmetry::NON_ABELIAN )
							{
								factor_cgc = Symmetry::coeff_HPsi(Aket[s2].out[get<2>(ix[n])], qloc[s2], Aket[s2].in[get<2>(ix[n])],
								                                  get<0>(ix[n])[2], qOp[k], L.mid(qL),
								                                  Abra[s1].out[get<1>(ix[n])], qloc[s1], Abra[s1].in[get<1>(ix[n])]);
							}
							else
							{
								factor_cgc = static_cast<Scalar>(1.);
							}
							if (std::abs(factor_cgc) < std::abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
							
							std::array<size_t,2> key = {s1, get<1>(ix[n])};
							std::array<size_t,5> val = {s2, get<2>(ix[n]), qL, qR->second, k};
							#ifndef DMRG_DONT_USE_OPENMP
							#pragma omp critical
							#endif
							{
								lookup[key].first.push_back(val);
								lookup[key].second.push_back(factor_cgc);
							}
						}
					}
				}
			}
		}
	}
	
	qlhs.clear();
	qrhs.clear();
	factor_cgcs.clear();
	
	qlhs.reserve(lookup.size());
	qrhs.reserve(lookup.size());
	factor_cgcs.reserve(lookup.size());
	
	for (auto it=lookup.begin(); it!=lookup.end(); ++it)
	{
		qlhs.push_back(it->first);
		qrhs.push_back((it->second).first);
		factor_cgcs.push_back((it->second).second);
	}
}

#endif
