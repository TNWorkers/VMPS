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

template<size_t D, size_t Nq, typename MatrixType> class MpsQ;
template<size_t D, size_t Nq, typename Scalar> class MpoQ;
template<size_t D, size_t Nq, typename MpHamiltonian> class DmrgSolverQ;
template<size_t D, size_t Nq, typename MatrixType> class MpsQCompressor;

/**Matrix Product Operator with conserved quantum numbers (Abelian symmetries). Just adds a target quantum number and a bunch of labels on top of Mpo.
@describe_D
@describe_Nq*/
template<size_t D, size_t Nq, typename Scalar=double>
class MpoQ : public Mpo<D,Scalar>
{
template<size_t D_, size_t Nq_, typename MpHamiltonian> friend class DmrgSolverQ;
template<size_t D_, size_t Nq_, typename S_> friend class MpsQCompressor;
template<size_t D_, size_t Nq_, typename S1, typename S2> friend void HxV (const MpoQ<D_,Nq_,S1> &H, const MpsQ<D_,Nq_,S2> &Vin, MpsQ<D_,Nq_,S2> &Vout);
template<size_t D_, size_t Nq_, typename S1, typename S2> friend void OxV (const MpoQ<D_,Nq_,S1> &H, const MpsQ<D_,Nq_,S2> &Vin, MpsQ<D_,Nq_,S2> &Vout, DMRG::BROOM::OPTION TOOL);

public:
	
	//---constructors---
	///@{
	MpoQ (){};
	/**Construct with default values.
	MpoQ<D,Nq,Scalar>::qlabel and MpoQ<D,Nq,Scalar>::qloc will also be set to default values. Doing it by default argument produces errors with g++.*/
	MpoQ (size_t L_input, std::array<qarray<Nq>,D> qloc_input, 
	      string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	/**Construct with all values.*/
	MpoQ (size_t L_input, std::array<qarray<Nq>,D> qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input, string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	/**Construct with all values and a SuperMatrix (useful when constructing an MpoQ by another MpoQ).*/
	MpoQ (size_t L_input, const SuperMatrix<D,Scalar> &G, std::array<qarray<Nq>,D> qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input, string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	/**Construct with all values and a vector of SuperMatrices (useful when constructing an MpoQ by another MpoQ).*/
	MpoQ (size_t L_input, const vector<SuperMatrix<D,Scalar> > &Gvec, std::array<qarray<Nq>,D> qloc_input, qarray<Nq> Qtot_input, 
	      std::array<string,Nq> qlabel_input, string label_input="MpoQ", string (*format_input)(qarray<Nq> qnum)=noFormat);
	///@}
	
	//---info stuff---
	///@{
	/**\describe_info*/
	string info() const;
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=MB) const;
	///@}
	
	//---formatting stuff---
	///@{
	/**Format function for the quantum numbers (e.g.\ half-integers for S=1/2).*/
	string (*format)(qarray<Nq> qnum);
	/**How this MpoQ should be called in outputs.*/
	string label;
	/**Label for quantum numbers in output (e.g.\ \f$M\f$ for magnetization; \f$N_\uparrow\f$,\f$N_\downarrow\f$ for particle numbers etc.).*/
	std::array<string,Nq> qlabel;
	///@}
	
	//---return stuff---
	///@{
	/**Returns the total change in quantum numbers induced by the MpoQ.*/
	inline qarray<Nq> Qtarget() const {return Qtot;};
	/**Returns the local basis.*/
	inline std::array<qarray<Nq>,D> locBasis() const {return qloc;}
	///@}
	
private:
	
	qarray<Nq> Qtot;
	std::array<qarray<Nq>,D> qloc;
};

template<size_t D, size_t Nq, typename Scalar>
MpoQ<D,Nq,Scalar>::
MpoQ (size_t L_input, std::array<qarray<Nq>,D> qloc_input, 
      string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<D,Scalar>(L_input), qloc(qloc_input), label(label_input), format(format_input)
{
	qlabel = defaultQlabel<Nq>();
	Qtot = qvacuum<Nq>();
}

template<size_t D, size_t Nq, typename Scalar>
MpoQ<D,Nq,Scalar>::
MpoQ (size_t L_input, std::array<qarray<Nq>,D> qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input,  string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<D,Scalar>(L_input), qloc(qloc_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input)
{}

template<size_t D, size_t Nq, typename Scalar>
MpoQ<D,Nq,Scalar>::
MpoQ (size_t L_input, const SuperMatrix<D,Scalar> &G, std::array<qarray<Nq>,D> qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input,  string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<D,Scalar>(L_input), qloc(qloc_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input)
{
	this->Daux = G.auxdim();
	this->N_sv = this->Daux;
	this->construct(G, this->W, this->Gvec);
}

template<size_t D, size_t Nq, typename Scalar>
MpoQ<D,Nq,Scalar>::
MpoQ (size_t L_input, const vector<SuperMatrix<D,Scalar> > &Gvec, std::array<qarray<Nq>,D> qloc_input, qarray<Nq> Qtot_input, 
      std::array<string,Nq> qlabel_input,  string label_input, string (*format_input)(qarray<Nq> qnum))
:Mpo<D,Scalar>(L_input), qloc(qloc_input), Qtot(Qtot_input), qlabel(qlabel_input), label(label_input), format(format_input)
{
	this->Daux = Gvec[0].auxdim();
	this->N_sv = this->Daux;
	this->construct(Gvec, this->W, this->Gvec);
}

template<size_t D, size_t Nq, typename Scalar>
string MpoQ<D,Nq,Scalar>::
info() const
{
	stringstream ss;
	ss << label << ": " << "L=" << this->N_sites << ", ";
	
	ss << "(";
	for (size_t q=0; q<Nq; ++q)
	{
		ss << qlabel[q];
		if (q != Nq-1) {ss << ",";}
	}
	ss << ")={";
	for (size_t q=0; q<D; ++q)
	{
		ss << format(qloc[q]);
		if (q != D-1) {ss << ",";}
	}
	ss << "}, ";
	
	ss << "Daux=" << this->Daux << ", ";
//	ss << "trunc_weight=" << this->truncWeight.sum() << ", ";
	ss << "mem=" << round(this->memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<size_t D, size_t Nq, typename Scalar>
inline double MpoQ<D,Nq,Scalar>::
overhead (MEMUNIT memunit) const
{
	return 0.;
}

//template<size_t D, size_t Nq, typename Scalar>
//void MpoQ<D,Nq,Scalar>::
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

//template<size_t D, size_t Nq, typename Scalar>
//void MpoQ<D,Nq,Scalar>::
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

//template<size_t D, size_t Nq, typename Scalar>
//void MpoQ<D,Nq,Scalar>::
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

template<size_t D, size_t Nq, typename Scalar>
ostream &operator<< (ostream& os, const MpoQ<D,Nq,Scalar> &O)
{
	assert (O.format and "Empty pointer to format function in MpoQ!");
	
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "MpoQ: L=" << O.length() << ", D=" << D << ", Daux=" << O.auxdim();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<O.length(); ++l)
	{
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			os << "[l=" << l << "]\t|" << O.format(O.locBasis()[s1]) << "><" << O.format(O.locBasis()[s2]) << "|:" << endl;
			os << MatrixXd(O.W_at(l)[s1][s2]) << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != O.length()-1) {os << endl;}
	}
	return os;
}

#endif
