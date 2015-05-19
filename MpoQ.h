#ifndef STRAWBERRY_MPO_WITH_Q
#define STRAWBERRY_MPO_WITH_Q

#include <Eigen/SparseCore>
#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
typedef Eigen::SparseMatrix<double,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
using namespace Eigen;

#include "SuperMatrix.h"
#include "qarray.h"
#include "Biped.h"
#include "DmrgPivotStuffQ.h"
#include "Mpo.h"

//map<size_t,size_t> homogGdict (size_t L)
//{
//	map<size_t,size_t> mout;
//	mout.insert({0,0});
//	for (size_t l=1; l<L-1; ++l)
//	{
//		mout.insert({l,1});
//	}
//	mout.insert({L-1,2});
//	return mout;
//}

//map<size_t,size_t> trivialGdict (size_t L)
//{
//	map<size_t,size_t> mout;
//	for (size_t l=0; l<L; ++l)
//	{
//		mout.insert({l,l});
//	}
//	return mout;
//}

/**Namespace VMPS to distinguish names from ED equivalents.*/
namespace VMPS{};

template<size_t Nq, typename MatrixType> class MpsQ;
template<size_t Nq, typename Scalar> class MpoQ;
template<size_t Nq, typename MpHamiltonian> class DmrgSolverQ;
template<size_t Nq, typename Scalar, typename MpoScalar> class MpsQCompressor;

/**Matrix Product Operator with conserved quantum numbers (Abelian symmetries). Just adds a target quantum number and a bunch of labels on top of Mpo.
\describe_Nq
\describe_Scalar*/
template<size_t Nq, typename Scalar=double>
class MpoQ : public Mpo<Scalar>
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

template<size_t Nq_, typename MpHamiltonian> friend class DmrgSolverQ;
template<size_t Nq_, typename S1, typename S2> friend class MpsQCompressor;
template<size_t Nq_, typename S1, typename S2> friend void HxV (const MpoQ<Nq_,S1> &H, const MpsQ<Nq_,S2> &Vin, MpsQ<Nq_,S2> &Vout);
template<size_t Nq_, typename S1, typename S2> friend void OxV (const MpoQ<Nq_,S1> &H, const MpsQ<Nq_,S2> &Vin, MpsQ<Nq_,S2> &Vout, DMRG::BROOM::OPTION TOOL);

public:
	
	//---constructors---
	///\{
	/**Do nothing.*/
	MpoQ (){};
	
	/**Construct with default values.
	MpoQ<Nq,Scalar>::qlabel and MpoQ<Nq,Scalar>::qloc will also be set to default values. Doing it by default argument produces errors with g++.*/
	MpoQ (size_t L_input, vector<qarray<Nq> > qloc_input, 
	      string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	
	/**Construct with all values.*/
	MpoQ (size_t L_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input, string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	
	/**Construct with all values and a SuperMatrix (useful when constructing an MpoQ by another MpoQ).*/
	MpoQ (size_t L_input, const SuperMatrix<Scalar> &G, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input, string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	
	/**Construct with all values and a vector of SuperMatrices (useful when constructing an MpoQ by another MpoQ).*/
	MpoQ (size_t L_input, const vector<SuperMatrix<Scalar> > &Gvec, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input, string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	///\}
	
	//---info stuff---
	///\{
	/**\describe_info*/
	string info() const;
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	///\}
	
	//---formatting stuff---
	///\{
	/**Format function for the quantum numbers (e.g.\ half-integers for S=1/2).*/
	string (*format)(qarray<Nq> qnum);
	/**How this MpoQ should be called in outputs.*/
	string label;
	/**Label for quantum numbers in output (e.g.\ \f$M\f$ for magnetization; \f$N_\uparrow\f$,\f$N_\downarrow\f$ for particle numbers etc.).*/
	std::array<string,Nq> qlabel;
	///\}
	
	//---return stuff---
	///\{
	/**Returns the total change in quantum numbers induced by the MpoQ.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
	/**Returns the local basis.*/
	inline vector<qarray<Nq> > locBasis (size_t loc) const {return qloc[loc];}
	inline vector<vector<qarray<Nq> > > locBasis()   const {return qloc;}
	///\}
	
//	MpoQ<D,Nq,complex<double> > bondsTevol (double dt, PARITY P);
	
	class qarrayIterator;
	
protected:
	
	vector<vector<qarray<Nq> > > qloc;
	
	qarray<Nq> Qtot;
};

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, vector<qarray<Nq> > qloc_input, 
      string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<Scalar>(L_input), label(label_input), format(format_input)
{
	this->qlocsize.resize(this->N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		this->qlocsize[l] = qloc[l].size();
	}
	qlabel = defaultQlabel<Nq>();
	Qtot = qvacuum<Nq>();
}

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<Scalar>(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input)
{
	this->qlocsize.resize(this->N_sites);
	qloc.resize(this->N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		qloc[l].resize(qloc_input.size());
		for (size_t s=0; s<qloc_input.size(); ++s)
		{
			qloc[l][s] = qloc_input[s];
		}
		this->qlocsize[l] = qloc[l].size();
	}
}

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, const SuperMatrix<Scalar> &G, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<Scalar>(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input)
{
	this->qlocsize.resize(this->N_sites);
	qloc.resize(this->N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		qloc[l].resize(qloc_input.size());
		for (size_t s=0; s<qloc_input.size(); ++s)
		{
			qloc[l][s] = qloc_input[s];
		}
		this->qlocsize[l] = qloc[l].size();
	}
	this->Daux = G.auxdim();
	this->N_sv = this->Daux;
	this->construct(G, this->W, this->Gvec);
}

template<size_t Nq, typename Scalar>
MpoQ<Nq,Scalar>::
MpoQ (size_t L_input, const vector<SuperMatrix<Scalar> > &Gvec, vector<qarray<Nq> > qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input, string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<Scalar>(L_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input)
{
	this->qlocsize.resize(this->N_sites);
	qloc.resize(this->N_sites);
	for (size_t l=0; l<this->N_sites; ++l)
	{
		qloc[l].resize(qloc_input.size());
		for (size_t s=0; s<qloc_input.size(); ++s)
		{
			qloc[l][s] = qloc_input[s];
		}
		this->qlocsize[l] = qloc[l].size();
	}
	this->Daux = Gvec[0].auxdim();
	this->N_sv = this->Daux;
	this->construct(Gvec, this->W, this->Gvec);
}

template<size_t Nq, typename Scalar>
string MpoQ<Nq,Scalar>::
info() const
{
	stringstream ss;
	ss << label << ": " << "L=" << this->N_sites << ", ";
	
//	ss << "(";
//	for (size_t q=0; q<Nq; ++q)
//	{
//		ss << qlabel[q];
//		if (q != Nq-1) {ss << ",";}
//	}
//	ss << ")={";
//	for (size_t q=0; q<D; ++q)
//	{
//		ss << format(qloc[q]);
//		if (q != D-1) {ss << ",";}
//	}
//	ss << "}, ";
	
	ss << "Daux=" << this->Daux << ", ";
//	ss << "trunc_weight=" << this->truncWeight.sum() << ", ";
	ss << "mem=" << round(this->memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<size_t Nq, typename Scalar>
inline double MpoQ<Nq,Scalar>::
overhead (MEMUNIT memunit) const
{
	return 0.;
}

//template<size_t Nq, typename Scalar>
//MpoQ<D,Nq,complex<double> > MpoQ<Nq,Scalar>::
//bondsTevol (double dt, PARITY P)
//{
//	string TevolLabel = "exp("+label+"),";
//	TevolLabel += (P==EVEN)? "even" : "odd";
//	MpoQ<D,Nq,complex<double> > Mout(this->N_sites, qloc, qvacuum<Nq>(), qlabel, TevolLabel, format);
//	Mout.Daux = D*D;
//	Mout.W.resize(this->N_sites);
//	
//	if (P == ODD)
//	{
//		for (size_t l=0; l<this->N_sites; l+=this->N_sites-1)
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		{
//			Mout.W[l][s1][s2].resize(1,1);
//			if (s1 == s2)
//			{
//				Mout.W[l][s1][s2].coeffRef(0,0) = 1.;
//			}
//		}
//	}
//	
//	size_t l_frst = (P==EVEN)? 0 : 1;
//	size_t l_last = (P==EVEN)? this->N_sites-2 : this->N_sites-3;
//	
//	for (size_t l=l_frst; l<=l_last; l+=2)
//	{
//		MatrixType Hloc = kroneckerProduct(this->Gvec[1](1,0), this->Gvec[1](this->Daux-1,1));
//		for (size_t a=2; a<this->Daux-1; ++a)
//		{
//			Hloc += kroneckerProduct(this->Gvec[1](a,0), this->Gvec[1](this->Daux-1,a));
//		}
//		MatrixType Id(D,D); Id.setIdentity();
//		Hloc += kroneckerProduct(this->Gvec[1](this->Daux-1,0), Id);
//		
////		MatrixType Hloc = kroneckerProduct(this->Gvec[1](this->Daux-1,0), Id);
////		Hloc += kroneckerProduct(Id, this->Gvec[1](this->Daux-1,0));
//		
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t r1=0; r1<D; ++r1)
//		for (size_t s2=0; s2<D; ++s2)
//		for (size_t r2=0; r2<D; ++r2)
//		{
//			cout << s1 << ", " << r1 << ", " <<  s2 << ", " << r2 << " : " << Hloc(s1+r1*D,s2+r2*D) << endl;
//		}
//		
//		if (l==l_frst)
//		{
//			cout << Hloc << endl << endl;
//		}
//		
//		SelfAdjointEigenSolver<MatrixType> Eugen(Hloc);
//		Matrix<complex<double>,Dynamic,Dynamic> Hexp = Eugen.eigenvectors() * 
//		                                               (Eugen.eigenvalues()*complex<double>(0,-dt)).array().exp().matrix().asDiagonal() * 
//		                                               Eugen.eigenvectors().adjoint();
//		
//		#ifdef DONT_USE_LAPACK_SVD
//		JacobiSVD<MatrixXd> Jack;
//		#else
//		LapackSVD<complex<double> > Jack;
//		#endif
//		
//		#ifdef DONT_USE_LAPACK_SVD
//		Jack.compute(Hexp,ComputeThinU|ComputeThinV);
//		#else
//		Jack.compute(Hexp);
//		#endif
//		
//		MatrixXcd U1(D*D,D*D);
//		U1 = Jack.matrixU() * Jack.singularValues().cwiseSqrt().asDiagonal();
//		#ifdef DONT_USE_LAPACK_SVD
//		MatrixXcd U2(D*D,D*D);
//		U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixV().adjoint();
//		#else
//		MatrixXcd U2(D*D,D*D);
//		U2 = Jack.singularValues().cwiseSqrt().asDiagonal() * Jack.matrixVT();
//		#endif
//		
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t r1=0; r1<D; ++r1)
//		for (size_t s2=0; s2<D; ++s2)
//		for (size_t r2=0; r2<D; ++r2)
//		{
//			Mout.W[l][s1][r1].resize(1,D*D);
//			if (l != this->N_sites-1)
//			{
//				Mout.W[l+1][s1][r1].resize(D*D,1);
//			}
//			for (size_t k=0; k<D*D; ++k)
//			{
//				if (U1(s1+r1*D,k) != 0.)
//				{
//					Mout.W[l][s1][r1].coeffRef(0,k) = U1(s1+r1*D,k);
//				}
//				if (l != this->N_sites-1)
//				{
//					if (U2(k,s2+r2*D) != 0.)
//					{
//						Mout.W[l+1][s1][r1].coeffRef(k,0) = U2(k,s2+r2*D);
//					}
//				}
//			}
//		}
//	}
//	
////	for (size_t l=0; l<this->N_sites; ++l)
////	{
////		cout << "l=" << l << endl;
////		for (size_t s1=0; s1<D; ++s1)
////		for (size_t r1=0; r1<D; ++r1)
////		{
////			cout << Mout.W[l][s1][r1].rows() << "\t" << Mout.W[l][s1][r1].cols() << endl;
////		}
////	}
//	
//	return Mout;
//}

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
//rightSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL)
//{
//	ArrayXd truncWeightSub(outset[loc].size());
//	truncWeightSub.setZero();
//	
//	#ifndef DMRG_DONT_USE_OPENMP
//	#pragma omp parallel for
//	#endif
//	for (size_t qout=0; qout<outset[loc].size(); ++qout)
//	{
//		// determine how many A's to glue together
//		vector<size_t> qvec, Nrowsvec;
//		vector<pair<size_t,size_t> > svec;
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		for (size_t q=0; q<W[loc][s1][s2].dim; ++q)
//		{
//			if (W[loc][s1][s2].out[q] == outset[loc][qout])
//			{
//				svec.push_back({s1,s2});
//				qvec.push_back(q);
//				Nrowsvec.push_back(W[loc][s1][s2].block[q].rows());
//			}
//		}
//		
//		// do the glue
//		size_t Ncols = W[loc][svec[0].first][svec[0].second].block[qvec[0]].cols();
//		for (size_t i=1; i<svec.size(); ++i) {assert(W[loc][svec[i].first][svec[i].second].block[qvec[i]].cols() == Ncols);}
//		size_t Nrows = accumulate(Nrowsvec.begin(),Nrowsvec.end(),0);
//		
//		MatrixXd Aclump(Nrows,Ncols);
//		size_t stitch = 0;
//		for (size_t i=0; i<svec.size(); ++i)
//		{
//			Aclump.block(stitch,0, Nrowsvec[i],Ncols) = W[loc][svec[i].first][svec[i].second].block[qvec[i]];
//			stitch += Nrowsvec[i];
//		}
//		
//		// do the decomposition
//		JacobiSVD<MatrixXd> Jack;
//		size_t Nret = Ncols; // retained states
//		Jack.compute(Aclump,ComputeThinU|ComputeThinV);
//		if (TOOL == DMRG::BROOM::SVD)
//		{
//			Nret = (Jack.singularValues().array() > eps_svd).count();
//		}
//		else if (TOOL == DMRG::BROOM::BRUTAL_SVD)
//		{
//			Nret = min(static_cast<size_t>(Jack.singularValues().rows()), N_sv);
//		}
//		Nret = max(Nret,1ul);
//		truncWeightSub(qout) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
//		
//		// update W[loc]
//		stitch = 0;
//		for (size_t i=0; i<svec.size(); ++i)
//		{
//			W[loc][svec[i].first][svec[i].second].block[qvec[i]] = Jack.matrixU().block(stitch,0, Nrowsvec[i],Nret).sparseView(1.,1e-10);
//			stitch += Nrowsvec[i];
//		}
//		
//		// update W[loc+1]
//		if (loc != this->N_sites-1)
//		{
//			for (size_t s1=0; s1<D; ++s1)
//			for (size_t s2=0; s2<D; ++s2)
//			for (size_t q=0; q<W[loc+1][s1][s2].dim; ++q)
//			{
//				if (W[loc+1][s1][s2].in[q] == outset[loc][qout])
//				{
//					W[loc+1][s1][s2].block[q] = (Jack.singularValues().head(Nret).asDiagonal() * 
//					                            (Jack.matrixV().adjoint()).topRows(Nret) * 
//					                             W[loc+1][s1][s2].block[q]).sparseView(1.,1e-10);
//				}
//			}
//		}
//	}
//	
//	truncWeight(loc) = truncWeightSub.sum();
////	pivot = (loc==this->N_sites-1)? this->N_sites-1 : loc+1;
//}

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
//leftSweepStep (size_t loc, DMRG::BROOM::OPTION TOOL)
//{
//	ArrayXd truncWeightSub(inset[loc].size());
//	truncWeightSub.setZero();
//	
//	#ifndef DMRG_DONT_USE_OPENMP
//	#pragma omp parallel for
//	#endif
//	for (size_t qin=0; qin<inset[loc].size(); ++qin)
//	{
//		vector<size_t> qvec, Ncolsvec;
//		vector<pair<size_t,size_t> > svec;
//		for (size_t s1=0; s1<D; ++s1)
//		for (size_t s2=0; s2<D; ++s2)
//		for (size_t q=0; q<W[loc][s1][s2].dim; ++q)
//		{
//			if (W[loc][s1][s2].in[q] == inset[loc][qin])
//			{
//				svec.push_back({s1,s2});
//				qvec.push_back(q);
//				Ncolsvec.push_back(W[loc][s1][s2].block[q].cols());
//			}
//		}
//		
//		// do the glue
//		size_t Nrows = W[loc][svec[0].first][svec[0].second].block[qvec[0]].rows();
//		for (size_t i=1; i<svec.size(); ++i) {assert(W[loc][svec[i].first][svec[i].second].block[qvec[i]].rows() == Nrows);}
//		size_t Ncols = accumulate(Ncolsvec.begin(), Ncolsvec.end(), 0);
//		
//		MatrixXd Aclump(Nrows,Ncols);
//		size_t stitch = 0;
//		for (size_t i=0; i<svec.size(); ++i)
//		{
//			Aclump.block(0,stitch, Nrows,Ncolsvec[i]) = W[loc][svec[i].first][svec[i].second].block[qvec[i]];
//			stitch += Ncolsvec[i];
//		}
//		
//		// do the decomposition
//		size_t Nret = Nrows; // retained states
//		JacobiSVD<MatrixXd> Jack(Aclump,ComputeThinU|ComputeThinV);
//		
//		if (TOOL == DMRG::BROOM::SVD)
//		{
//			Nret = (Jack.singularValues().array() > eps_svd).count();
//		}
//		else if (TOOL == DMRG::BROOM::BRUTAL_SVD)
//		{
//			Nret = min(static_cast<size_t>(Jack.singularValues().rows()), N_sv);
//		}
//		Nret = max(Nret,1ul);
//		truncWeightSub(qin) = Jack.singularValues().tail(Jack.singularValues().rows()-Nret).cwiseAbs2().sum();
//		
//		// update W[loc]
//		stitch = 0;
//		for (size_t i=0; i<svec.size(); ++i)
//		{
//			W[loc][svec[i].first][svec[i].second].block[qvec[i]] = Jack.matrixV().adjoint().block(0,stitch, Nret,Ncolsvec[i]).sparseView(1.,1e-10);
//			stitch += Ncolsvec[i];
//		}
//		
//		// update W[loc-1]
//		if (loc != 0)
//		{
//			for (size_t s1=0; s1<D; ++s1)
//			for (size_t s2=0; s2<D; ++s2)
//			for (size_t q=0; q<W[loc-1][s1][s2].dim; ++q)
//			{
//				if (W[loc-1][s1][s2].out[q] == inset[loc][qin])
//				{
//					W[loc-1][s1][s2].block[q] = (W[loc-1][s1][s2].block[q] * 
//					                            Jack.matrixU().leftCols(Nret) * 
//					                            Jack.singularValues().head(Nret).asDiagonal()).sparseView(1.,1e-10);
//				}
//			}
//		}
//	}
//	
//	truncWeight(loc) = truncWeightSub.sum();
////	this->pivot = (loc==0)? 0 : loc-1;
//}

//template<size_t Nq, typename Scalar>
//void MpoQ<Nq,Scalar>::
//compress (DMRG::BROOM::OPTION TOOL, DMRG::DIRECTION::OPTION DIR)
//{
//	if (DIR == DMRG::DIRECTION::RIGHT)
//	{
//		for (size_t l=0; l<N_sites-1; ++l)
//		{
//			rightSweepStep(l,TOOL);
//		}
//	}
//	else
//	{
//		for (size_t l=N_sites-1; l>0; --l)
//		{
//			leftSweepStep(l,TOOL);
//		}
//	}
//}

template<size_t Nq, typename Scalar>
class MpoQ<Nq,Scalar>::qarrayIterator
{
public:
	
	/**
	\param qloc_input : vector of local bases
	\param l_frst : first site
	\param l_last : last site
	*/
	qarrayIterator (const vector<vector<qarray<Nq> > > &qloc_input, int l_frst, int l_last)
	:qloc(qloc_input[0])
	{
		N_sites = (l_last < 0 or l_frst >= qloc_input.size())? 0 : l_last-l_frst+1;
		
		// determine dq
		for (size_t q=0; q<Nq; ++q)
		{
			set<int> qset;
			for (size_t s=0; s<qloc.size(); ++s) {qset.insert(qloc[s][q]);}
			set<int> diffqset;
			for (auto it=qset.begin(); it!=qset.end(); ++it)
			{
				int prev;
				if (it==qset.begin()) {prev=*it;}
				else
				{
					diffqset.insert(*it-prev);
					prev = *it;
				}
			}
		
			assert(diffqset.size()==1 and 
			       "Unable to understand quantum number increments!");
			dq[q] = *diffqset.begin();
		}
		
		// determine qmin, qmax
		qmin = N_sites * (*min_element(qloc.begin(),qloc.end()));
		qmax = N_sites * (*max_element(qloc.begin(),qloc.end()));
		
		// setup NestedLoopIterator
		vector<size_t> ranges(Nq);
		for (size_t q=0; q<Nq; ++q)
		{
			ranges[q] = (qmax[q]-qmin[q])/dq[q]+1;
		}
		Nelly = NestedLoopIterator(Nq,ranges);
	};
	
	/**Returns the value of the quantum number.*/
	qarray<Nq> operator*() {return value;}
	
	qarrayIterator& operator= (const qarray<Nq> a) {value=a;}
	bool operator!= (const qarray<Nq> a) {return value!=a;}
	bool operator<= (const qarray<Nq> a) {return value<=a;}
	bool operator<  (const qarray<Nq> a) {return value< a;}
	
	qarray<Nq> begin()
	{
		Nelly = Nelly.begin();
		return qmin;
	}
	
	qarray<Nq> end()
	{
		qarray<Nq> qout = qmax;
		qout[0] += dq[0];
		return qout;
	}
	
	void operator++()
	{
		++Nelly;
		if (Nelly==Nelly.end())
		{
			value = qmax;
			value[0] += dq[0];
		}
		else
		{
			value = qmin;
			for (size_t q=0; q<Nq; ++q)
			{
				value[q] += Nelly(q)*dq[q];
			}
		}
	}
	
private:
	
	qarray<Nq> value;
	
	NestedLoopIterator Nelly;
	
	qarray<Nq> qmin;
	qarray<Nq> qmax;
	
	vector<qarray<Nq> > qloc;
	qarray<Nq> dq;
	size_t N_sites;
};

template<size_t Nq, typename Scalar>
ostream &operator<< (ostream& os, const MpoQ<Nq,Scalar> &O)
{
	assert (O.format and "Empty pointer to format function in MpoQ!");
	
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "MpoQ: L=" << O.length() << ", Daux=" << O.auxdim();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<O.length(); ++l)
	{
		for (size_t s1=0; s1<O.locBasis(l).size(); ++s1)
		for (size_t s2=0; s2<O.locBasis(l).size(); ++s2)
		{
			os << "[l=" << l << "]\t|" << O.format(O.locBasis(l)[s1]) << "><" << O.format(O.locBasis(l)[s2]) << "|:" << endl;
			os << Matrix<Scalar,Dynamic,Dynamic>(O.W_at(l)[s1][s2]) << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != O.length()-1) {os << endl;}
	}
	return os;
}

#endif
