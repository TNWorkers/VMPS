#ifndef STRAWBERRY_PEIERLSSUBSTITUTION
#define STRAWBERRY_PEIERLSSUBSTITUTION

#include "MpHubbardModel.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.*/
class PeierlsSubstitution : public MpoQ<2,complex<double> >
{
public:
	
	/**
	\param L_input : chain length
	\param U_input : \f$U\f$
	\param A_input : \f$A\f$
	\param V_input : \f$V\f$
	\param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	PeierlsSubstitution (size_t L_input, double U_input, double A_input=0., double V_input=0., bool CALC_SQUARE=true);
	
	static SuperMatrix<complex<double> > Generator (double U, double A=0., double V=0.);
	
	MpoQ<2,complex<double> > Hsq();
	
	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
	The quantum numbers are \f$N_{\uparrow}\f$ and \f$N_{\downarrow}\f$. Used by default.*/
	static const std::array<qarray<2>,4> qloc;
	
	/**Real MpsQ for convenient reference (no need to specify D, Nq all the time).*/
//	typedef MpsQ<2,double>                                      StateXd;
	/**Complex MpsQ for convenient reference (no need to specify D, Nq all the time).*/
	typedef MpsQ<2,complex<double> >                            StateXcd;
	typedef DmrgSolverQ<2,PeierlsSubstitution,complex<double> > Solver;
//	typedef MpsQCompressor<2,double,double>                     CompressorXd;
//	typedef MpsQCompressor<2,complex<double>,double>            CompressorXcd;
//	typedef MpoQ<2,double>                                      OperatorXd;
//	typedef MpoQ<2,complex<double> >                            OperatorXcd;
	
private:
	
	double U;
	double V = 0.;
	double A = 0.;
};

SuperMatrix<complex<double> > PeierlsSubstitution::
Generator (double U, double A, double V)
{
	size_t Daux = 6;
	complex<double> t = exp(-1.i*A);
	if (V != 0.) {Daux += 1;}
	
	vector<MatrixXcd> col;
	vector<MatrixXcd> row;
	
	// first col (except corner element)
	col.push_back(MatrixXcd::Identity(4,4));
	col.push_back(FermionBase::cUP.cast<complex<double> >().transpose());
	col.push_back(FermionBase::cDN.cast<complex<double> >().transpose());
	col.push_back(FermionBase::cUP.cast<complex<double> >());
	col.push_back(FermionBase::cDN.cast<complex<double> >());
	if (V != 0.)
	{
		col.push_back(FermionBase::n);
	}
	
	// last row (except corner element)
	row.push_back(-conj(t)*FermionBase::fsign * FermionBase::cUP.cast<complex<double> >());
	row.push_back(-conj(t)*FermionBase::fsign * FermionBase::cDN.cast<complex<double> >());
	row.push_back( t*FermionBase::fsign * FermionBase::cUP.cast<complex<double> >().transpose());
	row.push_back( t*FermionBase::fsign * FermionBase::cDN.cast<complex<double> >().transpose());
	if (V != 0.)
	{
		row.push_back(V * FermionBase::n);
	}
	row.push_back(MatrixXcd::Identity(4,4));
	
	SuperMatrix<complex<double> > G;
	G.setMatrix(Daux,4);
	G.setZero();
	
	for (size_t i=0; i<Daux-1; ++i)
	{
		G(i,0)        = col[i];
		G(Daux-1,i+1) = row[i];
	}
	
	// corner element
	G(Daux-1,0) = U * FermionBase::d.cast<complex<double> >();
	
	return G;
}

PeierlsSubstitution::
PeierlsSubstitution (size_t L_input, double U_input, double A_input, double V_input, bool CALC_SQUARE)
:MpoQ<2,complex<double> > (L_input, vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), {0,0}, HubbardModel::Nlabel, "PeierlsSubstitution"),
U(U_input), A(A_input), V(V_input)
{
	stringstream ss;
	ss << "(U=" << U << ",A=" << A << ",V=" << V << ")";
	this->label += ss.str();
	this->Daux = (V==0.)? 6 : 7;
	
	SuperMatrix<complex<double> > G = Generator(U,A,V);
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

MpoQ<2,complex<double> > PeierlsSubstitution::
Hsq()
{
	SuperMatrix<complex<double> > G = Generator(U,A,V);
	MpoQ<2,complex<double> > Mout(this->N_sites, tensor_product(G,G), vector<qarray<2> >(begin(HubbardModel::qloc),end(HubbardModel::qloc)), 
	                              {0,0}, HubbardModel::Nlabel, "PeierlsSubstitution H^2");
	return Mout;
}

}

#endif
