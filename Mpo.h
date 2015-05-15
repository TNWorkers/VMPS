#ifndef VANILLA_MPO
#define VANILLA_MPO

enum PARITY {EVEN=0, ODD=1};

#include "SuperMatrix.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Eigenvalues>
#include "LapackWrappers.h"

template<size_t D, typename Scalar> class Mps;
template<size_t D, typename Scalar> class MpsCompressor;

/**Matrix Product Operator.
@describe_D*/
template<size_t D, typename Scalar=double>
class Mpo
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

template<size_t D_, typename MpHamiltonian> friend class DmrgSolver;
template<size_t D_, typename S_> friend class MpsCompressor;
template<size_t D_, typename S1, typename S2> friend void HxV (const Mpo<D_,S1> &H, const Mps<D_,S2> &Vin, Mps<D_,S2> &Vout);

public:
	
	Mpo (size_t L_input);
	
	// set special
	void setLocal (size_t loc, const MatrixType &Op);
	void setLocal (size_t loc1, const MatrixType &Op1, size_t loc2, const MatrixType &Op2);
	void setProductSum (const MatrixType &Op1, const MatrixType &Op2);
	void setLocalSum (const MatrixType &Op);
	
	/**Makes a linear transformation of the MpoQ: \f$H' = factor*H + offset\f$.*/
	void scale (double factor=1., double offset=0.);
	
	// info
	string info() const;
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	inline size_t length() const {return N_sites;}
	/**\describe_Daux*/
	inline size_t auxdim() const {return Daux;}
	inline const std::array<std::array<SparseMatrix<Scalar>,D>,D> &W_at   (size_t loc) const {return W[loc];};
	inline const std::array<std::array<SparseMatrix<Scalar>,D>,D> &Wsq_at (size_t loc) const {return Wsq[loc];};
	inline bool check_SQUARE() const {return GOT_SQUARE;}
	
//protected:
	
	bool GOT_SQUARE;
	
	size_t N_sites;
	size_t Daux;
	
	size_t N_sv;
	double eps_svd;
	ArrayXd truncWeight;
	
	void construct (const SuperMatrix<D,Scalar> &G, vector<std::array<std::array<SparseMatrix<Scalar>,D>,D> > &Wstore, vector<SuperMatrix<D,Scalar> > &Gstore);
	void construct (const vector<SuperMatrix<D,Scalar> > &Gvec, vector<std::array<std::array<SparseMatrix<Scalar>,D>,D> > &Wstore, vector<SuperMatrix<D,Scalar> > &Gstore);
	
	vector<SuperMatrix<D,Scalar> > Gvec;
	vector<std::array<std::array<SparseMatrix<Scalar>,D>,D> > W;
	
	vector<SuperMatrix<D,Scalar> > GvecSq;
	vector<std::array<std::array<SparseMatrix<Scalar>,D>,D> > Wsq;
};

template<size_t D, typename Scalar>
Mpo<D,Scalar>::
Mpo (size_t L_input)
:N_sites(L_input), eps_svd(1e-7)
{
	truncWeight.resize(N_sites);
	truncWeight.setZero();
	GOT_SQUARE = false;
}

template<size_t D, typename Scalar>
string Mpo<D,Scalar>::
info() const
{
	stringstream ss;
	ss << "Mpo: " << "L=" << N_sites << ", ";
	ss << "Daux=" << Daux << ", ";
	ss << "trunc_weight=" << truncWeight.sum() << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	return ss.str();
}

template<size_t D, typename Scalar>
double Mpo<D,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	
	if (W.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			res += calc_memory(W[l][s1][s2],memunit);
			if (GOT_SQUARE == true)
			{
				res += calc_memory(Wsq[l][s1][s2],memunit);
			}
		}
	}
	
	if (Gvec.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			res += Gvec[l].memory(memunit);
			if (GOT_SQUARE == true)
			{
				res += GvecSq[l].memory(memunit);
			}
		}
	}
	
	return res;
}

template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
construct (const SuperMatrix<D,Scalar> &G, vector<std::array<std::array<SparseMatrix<Scalar>,D>,D> >  &Wstore, vector<SuperMatrix<D,Scalar> > &Gstore)
{
	vector<SuperMatrix<D,Scalar> > Gvec(N_sites);
	
//	make W^[0] from last row
	Gvec[0].setRowVector(G.auxdim());
	for (size_t i=0; i<G.cols(); ++i)
	{
		Gvec[0](0,i) = G(G.rows()-1,i);
	}
	
//	make W^[i], i=1,...,L-2
	for (size_t l=1; l<N_sites-1; ++l)
	{
		Gvec[l].setMatrix(G.auxdim());
		Gvec[l] = G;
	}
	
//	make W^[L-1] from first column
	Gvec[N_sites-1].setColVector(G.auxdim());
	for (size_t i=0; i<G.rows(); ++i)
	{
		Gvec[N_sites-1](i,0) = G(i,0);
	}
	
//	make Mpo
	construct(Gvec,Wstore,Gstore);
}

template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
construct (const vector<SuperMatrix<D,Scalar> > &Gvec, vector<std::array<std::array<SparseMatrix<Scalar>,D>,D> >  &Wstore, vector<SuperMatrix<D,Scalar> > &Gstore)
{
	Wstore.resize(N_sites);
	Gstore = Gvec;
	
	for (size_t l=0; l<N_sites;  ++l)
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	{
		Wstore[l][s1][s2].resize(Gvec[l].rows(), Gvec[l].cols());
		
		for (size_t a1=0; a1<Gvec[l].rows(); ++a1)
		for (size_t a2=0; a2<Gvec[l].cols(); ++a2)
		{
			double val = Gvec[l](a1,a2)(s1,s2);
			if (val != 0.)
			{
				Wstore[l][s1][s2].insert(a1,a2) = val;
			}
		}
	}
}

// O(loc)
template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
setLocal (size_t loc, const MatrixType &Op)
{
	assert(Op.rows() == D and Op.cols() == D);
	assert(loc < N_sites);
	
	Daux = 1;
	N_sv = Daux;
	vector<SuperMatrix<D,Scalar> > M(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		M[l].setMatrix(Daux);
		(l==loc)? M[l](0,0)=Op : M[l](0,0).setIdentity();
	}
	
	construct(M, W, Gvec);
}

// O1(loc1) * O2(loc2)
template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
setLocal (size_t loc1, const MatrixType &Op1, size_t loc2, const MatrixType &Op2)
{
	assert(Op1.rows() == D and Op1.cols() == D and 
	       Op2.rows() == D and Op2.cols() == D);
	assert(loc1 < N_sites and loc2 < N_sites);
	
	Daux = 1;
	N_sv = Daux;
	vector<SuperMatrix<D,Scalar> > M(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		M[l].setMatrix(Daux);
		if      (l==loc1) {M[l](0,0)=Op1;}
		else if (l==loc2) {M[l](0,0)=Op2;}
		else              {M[l](0,0).setIdentity();}
	}
	
	construct(M, W, Gvec);
}

// O(1)+O(2)+...+O(L)
template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
setLocalSum (const MatrixType &Op)
{
	assert(Op.rows() == D and Op.cols() == D);
	
	Daux = 2;
	N_sv = Daux;
	vector<SuperMatrix<D,Scalar> > M(N_sites);
	
	M[0].setRowVector(Daux);
	M[0](0,0) = Op;
	M[0](0,1).setIdentity();
	
	for (size_t l=1; l<N_sites-1; ++l)
	{
		M[l].setMatrix(Daux);
		M[l](0,0).setIdentity();
		M[l](0,1).setZero();
		M[l](1,0) = Op;
		M[l](1,1).setIdentity();
	}
	
	M[N_sites-1].setColVector(Daux);
	M[N_sites-1](0,0).setIdentity();
	M[N_sites-1](1,0) = Op;
	
	construct(M, W, Gvec);
}

// O1(1)*O2(2)+O1(2)*O1(3)+...+O1(L-1)*O2(L)
template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
setProductSum (const MatrixType &Op1, const MatrixType &Op2)
{
	assert(Op1.rows() == D and Op1.cols() == D and 
	       Op2.rows() == D and Op2.cols() == D);
	
	Daux = 3;
	N_sv = Daux;
	vector<SuperMatrix<D,Scalar> > M(N_sites);
	
	M[0].setRowVector(Daux);
	M[0](0,0).setIdentity();
	M[0](0,1) = Op1;
	M[0](0,2).setIdentity();
	
	for (size_t l=1; l<N_sites-1; ++l)
	{
		M[l].setMatrix(Daux);
		M[l].setZero();
		M[l](0,0).setIdentity();
		M[l](1,0) = Op1;
		M[l](2,0).setIdentity();
		M[l](2,1) = Op2;
		M[l](2,2).setIdentity();
	}
	
	M[N_sites-1].setColVector(Daux);
	M[N_sites-1](0,0).setIdentity();
	M[N_sites-1](1,0) = Op2;
	M[N_sites-1](2,0).setIdentity();
	
	construct(M, W, Gvec);
}

template<size_t D, typename Scalar>
void Mpo<D,Scalar>::
scale (double factor, double offset)
{
	/**Example for where to apply the scaling factor, 3-site Heisenberg:
	\f$\left(f \cdot h \cdot S^x_1, -f \cdot J \cdot S^z_1, I\right)
	
	\cdot 
	
	\left(
	\begin{array}{lll}
	I & 0 & 0 \\
	S^z_2 & 0 & 0 \\
	f\cdot h\cdot S^x_2 & -f\cdot J\cdot S2z & I
	\end{array}
	\right)
	
	\cdot
	
	\left(
	\begin{array}{l}
	I \\
	S^z_3 \\
	f\cdot h\cdot S^x_3
	\end{array}
	\right)
	
	= f \cdot h \cdot (S^x_1 + S^x_2 + S^x_3) + f \cdot J \cdot (S^z_1 \cdot S^z_2 + S^z_2 \cdot S^z_3)
	= f \cdot H\f$*/
	
	// apply to Gvec
	for (size_t l=0; l<N_sites-1; ++l)
	{
		size_t a1 = (l==0)? 0 : Daux-1;
		for (size_t a2=0; a2<Daux-1; ++a2)
		{
			Gvec[l](a1,a2) *= factor;
		}
	}
	Gvec[N_sites-1](Daux-1,0) *= factor;
	
	for (size_t l=0; l<N_sites; ++l)
	{
		size_t a1 = (l==0)? 0 : Daux-1;
		Gvec[l](a1,0) += offset/N_sites * Matrix<Scalar,D,D>::Identity();
	}
	
	// calc W from Gvec
	if (factor != 1.)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			size_t a1 = (l==0)? 0 : Daux-1;
			for (size_t s1=0; s1<D; ++s1)
			for (size_t s2=0; s2<D; ++s2)
			for (size_t a2=0; a2<Daux-1; ++a2)
			{
				W[l][s1][s2].coeffRef(a1,a2) *= factor;
			}
		}
		
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			W[N_sites-1][s1][s2].coeffRef(Daux-1,0) *= factor;
		}
	}
	if (offset != 0.)
	{
		// apply offset to local part:
		// leftmost element on first site
		// downmost element on last site
		// down left corner element for the rest
		for (size_t l=0; l<N_sites; ++l)
		{
			size_t a1 = (l==0)? 0 : Daux-1;
			for (size_t s=0; s<D; ++s)
			{
				W[l][s][s].coeffRef(a1,0) += offset/N_sites;
			}
		}
	}
	
	if (GOT_SQUARE == true)
	{
		// apply to GvecSq
		for (size_t l=0; l<N_sites; ++l)
		{
			GvecSq[l] = tensor_product(Gvec[l],Gvec[l]);
		}
		
		// calc Wsq to GvecSq
		for (size_t l=0; l<N_sites; ++l)
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			Wsq[l][s1][s2].resize(GvecSq[l].rows(), GvecSq[l].cols());
			
			for (size_t a1=0; a1<GvecSq[l].rows(); ++a1)
			for (size_t a2=0; a2<GvecSq[l].cols(); ++a2)
			{
				double val = GvecSq[l](a1,a2)(s1,s2);
				if (val != 0.)
				{
					Wsq[l][s1][s2].insert(a1,a2) = val;
				}
			}
		}
	}
}

template<size_t D, typename Scalar>
ostream &operator<< (ostream& os, const Mpo<D,Scalar> &O)
{
	os << setfill('-') << setw(30) << "-" << setfill(' ');
	os << "Mpo: L=" << O.length() << ", D=" << D << ", Daux=" << O.auxdim();
	os << setfill('-') << setw(30) << "-" << endl << setfill(' ');
	
	for (size_t l=0; l<O.length(); ++l)
	{
		for (size_t s1=0; s1<D; ++s1)
		for (size_t s2=0; s2<D; ++s2)
		{
			os << "[l=" << l << "]\t|" << s1 << "><" << s2 << "|:" << endl;
			os << Matrix<Scalar,Dynamic,Dynamic>(O.W_at(l)[s1][s2]) << endl;
		}
		os << setfill('-') << setw(80) << "-" << setfill(' ');
		if (l != O.length()-1) {os << endl;}
	}
	return os;
}

#endif
