#ifndef STRAWBERRY_DMRGINDEXGYMNASTICS
#define STRAWBERRY_DMRGINDEXGYMNASTICS

#include <tuple>

#include "qarray.h"
#include "Biped.h"
#include "Multipede.h"

/**Calculates the matching right indices when contracting a left transfer matrix with two MpsQ and an MpoQ.
\dotfile AWA.dot
\param Lin
\param Lout
\param Lmid
\param s1
\param s2
\param qloc : local basis
\param Abra
\param Aket
\param result : tuple of: an array with \p Rin, \p Rout, \p Rmid; block index of \p Abra; block index of \p Aket
\returns \p true if a match is found, \p false if not
\warning When using this function to create the left block on the next site, one needs to swap \p Rin and \p Rout.*/
template<size_t Nq, typename MatrixType>
bool AWA (qarray<Nq> Lin, qarray<Nq> Lout, qarray<Nq> Lmid, size_t s1, size_t s2, vector<qarray<Nq> > qloc, 
          const vector<Biped<Nq,MatrixType> > &Abra, 
          const vector<Biped<Nq,MatrixType> > &Aket, 
          tuple<qarray3<Nq>,size_t,size_t> &result)
{
	qarray<Nq> qRout = Lin + qloc[s1];
	qarray2<Nq> cmp1 = {Lin, qRout};
	auto q1 = Abra[s1].dict.find(cmp1);
	
	if (q1 != Abra[s1].dict.end())
	{
		qarray<Nq> qRin = Lout + qloc[s2];
		qarray2<Nq> cmp2 = {Lout, qRin};
		auto q2 = Aket[s2].dict.find(cmp2);
		
		if (q2 != Aket[s2].dict.end())
		{
			qarray<Nq> qRmid = Lmid + qloc[s1] - qloc[s2];
			
			result = make_tuple(qarray3<Nq>{qRin,qRout,qRmid}, q1->second, q2->second);
			return true;
		}
	}
	return false;
}

/**Calculates the matching right indices when contracting a left transfer matrix with two MpsQ and two MpoQ.
\dotfile AWWA.dot
\param Lin
\param Lout
\param Lbot
\param Ltop
\param s1
\param s2
\param s3
\param qloc : local basis
\param Abra
\param Aket
\param result : tuple of: an array with \p Rin, \p Rout, \p Rbot, \p Rtop; block index of \p Abra; block index of \p Aket
\returns \p true if a match is found, \p false if not
\warning When using this function to create the left block on the next site, one needs to swap \p Rin and \p Rout.*/
template<size_t Nq, typename MatrixType>
bool AWWA (qarray<Nq> Lin, qarray<Nq> Lout, qarray<Nq> Lbot, qarray<Nq> Ltop, 
          size_t s1, size_t s2, size_t s3, vector<qarray<Nq> > qloc, 
          const vector<Biped<Nq,MatrixType> > &Abra, 
          const vector<Biped<Nq,MatrixType> > &Aket, 
          tuple<qarray4<Nq>,size_t,size_t> &result)
{
	qarray<Nq> qRout = Lin + qloc[s1];
	qarray2<Nq> cmp1 = {Lin, qRout};
	auto q1 = Abra[s1].dict.find(cmp1);
	
	if (q1 != Abra[s1].dict.end())
	{
		qarray<Nq> qRin = Lout + qloc[s3];
		qarray2<Nq> cmp2 = {Lout, qRin};
		auto q2 = Aket[s3].dict.find(cmp2);
		
		if (q2 != Aket[s3].dict.end())
		{
			qarray<Nq> qRbot = Lbot + qloc[s1] - qloc[s2];
			qarray<Nq> qRtop = Ltop + qloc[s2] - qloc[s3];
			
			result = make_tuple(qarray4<Nq>{qRin,qRout,qRbot,qRtop}, q1->second, q2->second);
			return true;
		}
	}
	return false;
}

/**Prepares a PivotMatrixQ by filling PivotMatrixQ::qlhs and PivotMatrixQ::qrhs with the corresponding subspace indices.
Uses OpenMP.*/
template<size_t Nq, typename Scalar, typename MpoScalar>
void precalc_blockStructure (const Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > &L, 
                             const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                             const vector<vector<SparseMatrix<MpoScalar> > > &W, 
                             const vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                             const Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                             vector<qarray<Nq> > qloc, 
                             vector<std::array<size_t,2> > &qlhs, 
                             vector<vector<std::array<size_t,4> > > &qrhs)
{
//	Heff.W = W;
	
	unordered_map<std::array<size_t,2>, vector<std::array<size_t,4> > > lookup;
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t qL=0; qL<L.dim; ++qL)
	{
		tuple<qarray3<Nq>,size_t,size_t> ix;
		bool FOUND_MATCH = AWA(L.in(qL), L.out(qL), L.mid(qL), s1,s2, qloc, Abra,Aket, ix);
		
		if (FOUND_MATCH == true)
		{
			auto qR = R.dict.find(get<0>(ix));
			
			if (qR != R.dict.end())
			{
				bool ALL_BLOCKS_ARE_EMPTY = true;
				
				for (int k=0; k<W[s1][s2].outerSize(); ++k)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2],k); iW; ++iW)
				{
					if (L.block[qL][iW.row()][0].rows() != 0 and 
						R.block[qR->second][iW.col()][0].rows() != 0)
					{
						ALL_BLOCKS_ARE_EMPTY = false;
					}
				}
				if (ALL_BLOCKS_ARE_EMPTY == false)
				{
					std::array<size_t,2> key = {s1, get<1>(ix)};
					std::array<size_t,4> val = {s2, get<2>(ix), qL, qR->second};
					#ifndef DMRG_DONT_USE_OPENMP
					#pragma omp critical
					#endif
					{
					lookup[key].push_back(val);
					}
				}
			}
		}
	}
	
	qlhs.clear();
	qrhs.clear();
	qlhs.reserve(lookup.size());
	qrhs.reserve(lookup.size());
	
	for (auto it=lookup.begin(); it!=lookup.end(); ++it)
	{
		qlhs.push_back(it->first);
		qrhs.push_back(it->second);
	}
}

#endif
