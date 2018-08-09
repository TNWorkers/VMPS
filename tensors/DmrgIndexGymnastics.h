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
bool LAWA (qarray<Symmetry::Nq> Lin, qarray<Symmetry::Nq> Lout, qarray<Symmetry::Nq> Lmid,
          size_t s1, size_t s2, vector<qarray<Symmetry::Nq> > qloc,
          size_t k, vector<qarray<Symmetry::Nq> > qOp,
          const vector<Biped<Symmetry,MatrixType> > &Abra, 
          const vector<Biped<Symmetry,MatrixType> > &Aket, 
          vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > &result,
          const bool &IS_HAMILTONIAN = false)
{
	bool out = false;
	result.clear();
	
	auto Routs = Symmetry::reduceSilent(Lin,qloc[s1]);
	for (const auto &Rout:Routs)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, Rout};
		auto qAbra = Abra[s1].dict.find(cmp1);
		if (qAbra != Abra[s1].dict.end())
		{
			auto Rins = Symmetry::reduceSilent(Lout,qloc[s2]);
			for (const auto &Rin:Rins)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, Rin};
				auto qAket = Aket[s2].dict.find(cmp2);
				if (qAket != Aket[s2].dict.end())
				{
					auto Rmids = Symmetry::reduceSilent(Lmid,qOp[k]);
					for (const auto &Rmid:Rmids)
					{
						if (Symmetry::validate(qarray3<Symmetry::Nq>{Rin,Rmid,Rout}))
						{
							if (IS_HAMILTONIAN and find(qOp.begin(), qOp.end(), Rmid) == qOp.end()) {continue;}
							result.push_back(make_tuple(qarray3<Symmetry::Nq>{Rin,Rout,Rmid}, qAbra->second, qAket->second));
							out = true;
						}
					}
				}
			}
		}
	}
	return out;
}

template<typename Symmetry, typename MatrixType>
bool AWAR (qarray<Symmetry::Nq> Rin, qarray<Symmetry::Nq> Rout, qarray<Symmetry::Nq> Rmid,
          size_t s1, size_t s2, vector<qarray<Symmetry::Nq> > qloc,
          size_t k, vector<qarray<Symmetry::Nq> > qOp,
          const vector<Biped<Symmetry,MatrixType> > &Abra, 
          const vector<Biped<Symmetry,MatrixType> > &Aket, 
          vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > &result,
          const bool &IS_HAMILTONIAN = false)
{
	bool out = false;
	result.clear();
	
	auto Lins = Symmetry::reduceSilent(Rout,Symmetry::flip(qloc[s1]));
	for (const auto& Lin : Lins)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, Rout};
		auto qAbra = Abra[s1].dict.find(cmp1);
		if (qAbra != Abra[s1].dict.end())
		{
			auto Louts = Symmetry::reduceSilent(Rin,Symmetry::flip(qloc[s2]));
			for (const auto &Lout : Louts)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, Rin};
				auto qAket = Aket[s2].dict.find(cmp2);
				if (qAket != Aket[s2].dict.end())
				{
					auto Lmids = Symmetry::reduceSilent(Rmid,Symmetry::flip(qOp[k]));
					for (const auto &Lmid : Lmids)
					{
						if (Symmetry::validate(qarray3<Symmetry::Nq>{Lout,Lmid,Lin}))
						{
							if (IS_HAMILTONIAN and find(qOp.begin(), qOp.end(), Lmid) == qOp.end()) {continue;}
							result.push_back(make_tuple(qarray3<Symmetry::Nq>{Lin,Lout,Lmid}, qAbra->second, qAket->second));
							out = true;
						}
					}
				}
			}
		}
	}
	return out;
}

template<typename Symmetry, typename MatrixType>
bool LAA (qarray<Symmetry::Nq> Lin,
          qarray<Symmetry::Nq> Lout,
          size_t s, 
          vector<qarray<Symmetry::Nq> > qloc,
          const vector<Biped<Symmetry,MatrixType> > &Abra, 
          const vector<Biped<Symmetry,MatrixType> > &Aket, 
          vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > &result)
{
	bool out = false;
	result.clear();
	
	qarray<Symmetry::Nq> Linout = Lin; // Lin = Lout = Linout;
	
	auto Rinouts = Symmetry::reduceSilent(Linout,qloc[s]);
	
	for (const auto &Rinout : Rinouts)
	{
		qarray2<Symmetry::Nq> cmp = {Linout, Rinout};
		auto qAket = Aket[s].dict.find(cmp);
		
		if (qAket != Aket[s].dict.end())
		{
			auto qAbra = Abra[s].dict.find(cmp);
			
			if (qAbra != Abra[s].dict.end())
			{
				result.push_back(make_tuple(qarray2<Symmetry::Nq>{Rinout,Rinout}, qAbra->second, qAket->second));
				out = true;
			}
		}
	}
	return out;
}

template<typename Symmetry, typename MatrixType>
bool AAR (qarray<Symmetry::Nq> Rin,
          qarray<Symmetry::Nq> Rout,
          size_t s, 
          vector<qarray<Symmetry::Nq> > qloc,
          const vector<Biped<Symmetry,MatrixType> > &Abra, 
          const vector<Biped<Symmetry,MatrixType> > &Aket, 
          vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > &result)
{
	bool out = false;
	result.clear();
	
	qarray<Symmetry::Nq> Rinout = Rin; // Rin = Rout = Rinout;
	
	auto Linouts = Symmetry::reduceSilent(Rinout, Symmetry::flip(qloc[s]));
	
	for (const auto &Linout : Linouts)
	{
		qarray2<Symmetry::Nq> cmp = {Linout, Rinout};
		auto qAket = Aket[s].dict.find(cmp);
		
		if (qAket != Aket[s].dict.end())
		{
			auto qAbra = Abra[s].dict.find(cmp);
			
			if (qAbra != Abra[s].dict.end())
			{
				result.push_back(make_tuple(qarray2<Symmetry::Nq>{Linout,Linout}, qAbra->second, qAket->second));
				out = true;
			}
		}
	}
	return out;
}

template<typename Symmetry, typename MatrixType>
bool AAWWAA (const qarray<Symmetry::Nq> &Lin, 
             const qarray<Symmetry::Nq> &Lout, 
             const qarray<Symmetry::Nq> &Lmid, 
             const size_t &k12, 
             const vector<qarray<Symmetry::Nq> > &qOp12, 
             const size_t &k34, 
             const vector<qarray<Symmetry::Nq> > &qOp34,
             const size_t &s1s3, 
             const qarray<Symmetry::Nq> &qmerge13, 
             const size_t &s2s4, 
             const qarray<Symmetry::Nq> &qmerge24, 
             const vector<Biped<Symmetry,MatrixType> > &AA13, 
             const vector<Biped<Symmetry,MatrixType> > &AA24, 
             vector<tuple<qarray3<Symmetry::Nq>,qarray<Symmetry::Nq>,size_t,size_t> > &result)
{
	bool out = false;
	result.clear();
	
	auto qRouts = Symmetry::reduceSilent(Lin,  qmerge13);
	auto qRins  = Symmetry::reduceSilent(Lout, qmerge24);
	
	auto qWcandidates = Symmetry::reduceSilent(Lmid, qOp12[k12]);
	vector<qarray<Symmetry::Nq> > qWs;
	for (const auto &qW:qWcandidates)
	{
//		if (find(qOp12.begin(), qOp12.end(), qW) != qOp12.end())
		{
			qWs.push_back(qW);
		}
	}
	
	for (const auto &qRout:qRouts)
	{
		qarray2<Symmetry::Nq> cmp1 = {Lin, qRout};
		auto q13 = AA13[s1s3].dict.find(cmp1);
		
		if (q13 != AA13[s1s3].dict.end())
		{
			for (const auto &qRin:qRins)
			{
				qarray2<Symmetry::Nq> cmp2 = {Lout, qRin};
				auto q24 = AA24[s2s4].dict.find(cmp2);
				
				if (q24 != AA24[s2s4].dict.end())
				{
					for (const auto &qW:qWs)
					{
						auto qRmids = Symmetry::reduceSilent(qW, qOp34[k34]);
						
						for (const auto &qRmid:qRmids)
						{
							if (Symmetry::validate(qarray3<Symmetry::Nq>{qRin,qRmid,qRout}))
//							and find(qOp34.begin(), qOp34.end(), qRmid) != qOp34.end())
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

template<typename Symmetry, typename Scalar>
vector<qarray<Symmetry::Nq> > calc_qsplit (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
                                           const vector<qarray<Symmetry::Nq> > &qloc1, 
                                           const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
                                           vector<qarray<Symmetry::Nq> > qloc2,
                                           const qarray<Symmetry::Nq> &Qtop, 
                                           const qarray<Symmetry::Nq> &Qbot)
{
	set<qarray<Symmetry::Nq> > qmid_fromL;
	set<qarray<Symmetry::Nq> > qmid_fromR;
	vector<qarray<Symmetry::Nq> > A1in;
	vector<qarray<Symmetry::Nq> > A2out;
	
	// gather all qin at the left:
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t q=0; q<A1[s1].dim; ++q)
	{
		A1in.push_back(A1[s1].in[q]);
	}
	// gather all qout at the right:
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	for (size_t q=0; q<A2[s2].dim; ++q)
	{
		A2out.push_back(A2[s2].out[q]);
	}
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	{
		auto qls = Symmetry::reduceSilent(A1in, qloc1[s1]);
		for (auto const &ql:qls)
		{
			if (ql<=Qtop and ql>=Qbot)
			{
				qmid_fromL.insert(ql);
			}
		}
	}
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	{
		auto qrs = Symmetry::reduceSilent(A2out, Symmetry::flip(qloc2[s2]));
		for (auto const &qr:qrs)
		{
			if (qr<=Qtop and qr>=Qbot)
			{
				qmid_fromR.insert(qr);
			}
		}
	}
	
	vector<qarray<Symmetry::Nq> > qres;
//	sort(qmid_fromL.begin(), qmid_fromL.end());
//	sort(qmid_fromR.begin(), qmid_fromR.end());
	// take common elements between left and right:
	set_intersection(qmid_fromL.begin(), qmid_fromL.end(), qmid_fromR.begin(), qmid_fromR.end(), back_inserter(qres));
	// erase non-unique elements to be sure:
	sort(qres.begin(), qres.end());
	qres.erase(unique(qres.begin(), qres.end()), qres.end());
	
	return qres;
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
				if (Symmetry::validate(qarray2<Symmetry::Nq>{qRin,qRout}))
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
	for (size_t k=0; k<qOp.size(); ++k)
	{
		if (!Symmetry::validate(qarray3<Symmetry::Nq>{qloc[s2],qOp[k],qloc[s1]})) {continue;}
		
		for (size_t qL=0; qL<L.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = LAWA(L.in(qL), L.out(qL), L.mid(qL), s1,s2, qloc, k, qOp, Abra, Aket, ix);
			
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); ++n)
				{
//					if (Aket[s2].block[get<2>(ix[n])].size() == 0) {continue;}
					
					auto qR = R.dict.find(get<0>(ix[n]));
					
					if (qR != R.dict.end())
					{
						bool ALL_BLOCKS_ARE_EMPTY = true;
						
						for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
						for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
						{
							if (L.block[qL][iW.row()][0].size() != 0 and 
							    R.block[qR->second][iW.col()][0].size() != 0)
							{
								ALL_BLOCKS_ARE_EMPTY = false;
							}
						}
						if (ALL_BLOCKS_ARE_EMPTY == false)
						{
							factor_cgc = (Symmetry::NON_ABELIAN)? 
							Symmetry::coeff_HPsi(Aket[s2].out[get<2>(ix[n])], qloc[s2], Aket[s2].in[get<2>(ix[n])],
							                     get<0>(ix[n])[2], qOp[k], L.mid(qL),
							                     Abra[s1].out[get<1>(ix[n])], qloc[s1], Abra[s1].in[get<1>(ix[n])])
							                     :1.;
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

/**Prepares a PivotMatrix2 by filling PivotMatrix::qlhs and PivotMatrix::qrhs with the corresponding subspace indices.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void precalc_blockStructure (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L, 
                             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                             const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12, 
                             const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34, 
                             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                             const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                             const vector<qarray<Symmetry::Nq> > &qloc12,
                             const vector<qarray<Symmetry::Nq> > &qloc34,
                             const vector<qarray<Symmetry::Nq> > &qOp12,
                             const vector<qarray<Symmetry::Nq> > &qOp34,
                             const vector<TwoSiteData<Symmetry,MpoScalar> > &TSD, 
                             vector<std::array<size_t,2> > &qlhs, 
                             vector<vector<std::array<size_t,10> > > &qrhs,
                             vector<vector<Scalar> > &factor_cgcs)
{
	unordered_map<std::array<size_t,2>, 
	              std::pair<vector<std::array<size_t,10> >, vector<Scalar> > > lookup;
	Scalar factor_cgc;
	
	for (const auto &tsd:TSD)
	for (size_t qL=0; qL<L.dim; ++qL)
	{
		vector<tuple<qarray3<Symmetry::Nq>,qarray<Symmetry::Nq>,size_t,size_t> > ixs;
		bool FOUND_MATCH = AAWWAA(L.in(qL), L.out(qL), L.mid(qL), 
		                          tsd.k12, qOp12, tsd.k34, qOp34,
		                          tsd.s1s3, tsd.qmerge13, tsd.s2s4, tsd.qmerge24,
		                          Abra, Aket, ixs);
		
		if (FOUND_MATCH)
		{
			for (const auto &ix:ixs)
			{
				auto qR = R.dict.find(get<0>(ix));
				auto qW     = get<1>(ix);
				size_t qA13 = get<2>(ix);
				size_t qA24 = get<3>(ix);
				
				// multiplication of Op12, Op34 in the auxiliary space
				Scalar factor_cgc6 = (Symmetry::NON_ABELIAN)? 
				Symmetry::coeff_Apair(L.mid(qL), qOp12[tsd.k12], qW,
				                      qOp34[tsd.k34], get<0>(ix)[2], tsd.qOp)
				                      :1.;
				if (abs(factor_cgc6) < abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
				
				if (qR != R.dict.end())
				{
					// standard coefficient for H*Psi with environments
					Scalar factor_cgcHPsi = (Symmetry::NON_ABELIAN)?
					Symmetry::coeff_HPsi(Aket[tsd.s2s4].out[qA24], tsd.qmerge24, Aket[tsd.s2s4].in[qA24],
					                     R.mid(qR->second), tsd.qOp, L.mid(qL),
					                     Abra[tsd.s1s3].out[qA13], tsd.qmerge13, Abra[tsd.s1s3].in[qA13])
					                     :1.;
					
					std::array<size_t,2>  key = {static_cast<size_t>(tsd.s1s3), qA13};
					std::array<size_t,10> val = {static_cast<size_t>(tsd.s2s4), qA24, qL, qR->second, 
					                             tsd.s1, tsd.s2, tsd.k12, tsd.s3, tsd.s4, tsd.k34};
					lookup[key].first.push_back(val);
					lookup[key].second.push_back(factor_cgc6 * tsd.cgc9 * factor_cgcHPsi);
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

/**Prepares a PivotMatrix2 by filling PivotMatrix::qlhs and PivotMatrix::qrhs with the corresponding subspace indices.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void precalc_blockStructure (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L, 
                             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                             const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12, 
                             const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34, 
                             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                             const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                             const vector<qarray<Symmetry::Nq> > &qloc12,
                             const vector<qarray<Symmetry::Nq> > &qloc34,
                             const vector<qarray<Symmetry::Nq> > &qOp12,
                             const vector<qarray<Symmetry::Nq> > &qOp34,
                             vector<std::array<size_t,2> > &qlhs, 
                             vector<vector<std::array<size_t,10> > > &qrhs,
                             vector<vector<Scalar> > &factor_cgcs)
{
	unordered_map<std::array<size_t,2>, 
	              std::pair<vector<std::array<size_t,10> >, vector<Scalar> > > lookup;
	
	auto tensor_basis = Symmetry::tensorProd(qloc12, qloc34);
	
	for (size_t s1=0; s1<qloc12.size(); ++s1)
	for (size_t s2=0; s2<qloc12.size(); ++s2)
	for (size_t k12=0; k12<qOp12.size(); ++k12)
	{
		if (!Symmetry::validate(qarray3<Symmetry::Nq>{qloc12[s2], qOp12[k12], qloc12[s1]})) {continue;}
		
		for (size_t s3=0; s3<qloc34.size(); ++s3)
		for (size_t s4=0; s4<qloc34.size(); ++s4)
		for (size_t k34=0; k34<qOp34.size(); ++k34)
		{
			if (!Symmetry::validate(qarray3<Symmetry::Nq>{qloc34[s4], qOp34[k34], qloc34[s3]})) {continue;}
			
//			vector<qarray<Symmetry::Nq> > qOps;
//			if constexpr (Symmetry::NON_ABELIAN)
//			{
//				if (qOp12[k12] == qOp34[k34])
//				{
//					qOps.push_back(Symmetry::qvacuum());
//				}
//				else
//				{
//					qOps.push_back({3});
//				}
//			}
//			else
//			{
//				qOps = Symmetry::reduceSilent(qOp12[k12], qOp34[k34]);
//			}
			auto qOps = Symmetry::reduceSilent(qOp12[k12], qOp34[k34]);
			
			for (const auto &qOp:qOps)
			{
//				if (find(qOp34.begin(), qOp34.end(), qOp) == qOp34.end()) {continue;}
				
				auto qmerges13 = Symmetry::reduceSilent(qloc12[s1], qloc34[s3]);
				auto qmerges24 = Symmetry::reduceSilent(qloc12[s2], qloc34[s4]);
				
				for (const auto &qmerge13:qmerges13)
				for (const auto &qmerge24:qmerges24)
				{
					auto qtensor13 = make_tuple(qloc12[s1], s1, qloc34[s3], s3, qmerge13);
					auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor13));
					
					auto qtensor24 = make_tuple(qloc12[s2], s2, qloc34[s4], s4, qmerge24);
					auto s2s4 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor24));
					
					// tensor product of the MPO operators in the physical space
					Scalar factor_cgc9 = (Symmetry::NON_ABELIAN)? 
					Symmetry::coeff_tensorProd(qloc12[s2], qloc34[s4], qmerge24,
					                           qOp12[k12], qOp34[k34], qOp,
					                           qloc12[s1], qloc34[s3], qmerge13)
					                           :1.;
					if (abs(factor_cgc9) < abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
					
					for (size_t qL=0; qL<L.dim; ++qL)
					{
						vector<tuple<qarray3<Symmetry::Nq>,qarray<Symmetry::Nq>,size_t,size_t> > ixs;
						bool FOUND_MATCH = AAWWAA(L.in(qL), L.out(qL), L.mid(qL), 
						                          k12, qOp12, k34, qOp34,
						                          s1s3, qmerge13, s2s4, qmerge24,
						                          Abra, Aket, ixs);
						
						if (FOUND_MATCH)
						{
							for (const auto &ix:ixs)
							{
								auto qR = R.dict.find(get<0>(ix));
								auto qW     = get<1>(ix);
								size_t qA13 = get<2>(ix);
								size_t qA24 = get<3>(ix);
								
								// multiplication of Op12, Op34 in the auxiliary space
								Scalar factor_cgc6 = (Symmetry::NON_ABELIAN)? 
								Symmetry::coeff_Apair(L.mid(qL), qOp12[k12], qW,
								                      qOp34[k34], get<0>(ix)[2], qOp)
								                      :1.;
								if (abs(factor_cgc6) < abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
								
								if (qR != R.dict.end())
								{
									// standard coefficient for H*Psi with environments
									Scalar factor_cgcHPsi = (Symmetry::NON_ABELIAN)?
									Symmetry::coeff_HPsi(Aket[s2s4].out[qA24], qmerge24, Aket[s2s4].in[qA24],
									                     R.mid(qR->second), qOp, L.mid(qL),
									                     Abra[s1s3].out[qA13], qmerge13, Abra[s1s3].in[qA13])
									                     :1.;
									
									std::array<size_t,2>  key = {static_cast<size_t>(s1s3), qA13};
									std::array<size_t,10> val = {static_cast<size_t>(s2s4), qA24, qL, qR->second, 
									                             s1, s2, k12, s3, s4, k34};
									lookup[key].first.push_back(val);
									lookup[key].second.push_back(factor_cgc6 * factor_cgc9 * factor_cgcHPsi);
								}
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
