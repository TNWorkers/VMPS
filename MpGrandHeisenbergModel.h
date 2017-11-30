#ifndef VANILLA_GRANDHEISENBERGMODEL
#define VANILLA_GRANDHEISENBERGMODEL

#include "MpHeisenbergModel.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) - J_z \sum_{<ij>} S^z_iS^z_j - B_z \sum_i S^z_i - B_x \sum_i S^x_i 
\f$.
\param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
\note \f$J<0\f$ : antiferromagnetic*/
class GrandHeisenbergModel : public MpoQ<Sym::U0,double>
{
	typedef Sym::U0 Symmetry;
	typedef typename Symmetry::qType qType;
public:
	
	GrandHeisenbergModel (int Lx_input, double Jxy_input=-1., double Jz_input=numeric_limits<double>::infinity(), 
	                      double Bz_input=0., double Bx_input=0., int Ly_input=1, bool CALC_SQUARE=true, size_t D_input=2);
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<Sym::U0,double>                           StateXd;
	typedef MpsQ<Sym::U0,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Sym::U0,GrandHeisenbergModel>      Solver;
	typedef MpsQCompressor<Sym::U0,double,double>          CompressorXd;
	typedef MpsQCompressor<Sym::U0,complex<double>,double> CompressorXcd;
	typedef MpoQ<Sym::U0>                                  Operator;
	///@}
	
	static const vector<qarray<0> > qloc (size_t N_legs=1, size_t D=2);
	
	MpoQ<Sym::U0> SzSz (size_t loc1, size_t loc2);
	MpoQ<Sym::U0> Sz   (size_t loc);
	
private:
	
	double Jxy, Jz;
	double Bz, Bx;
	size_t D;
	SpinBase<Symmetry> B;
};

const vector<qarray<0> > GrandHeisenbergModel::
qloc (size_t N_legs, size_t D)
{
	vector<qarray<0> > vout(pow(D,N_legs),qarray<0>{});
	return vout;
};

GrandHeisenbergModel::
GrandHeisenbergModel (int Lx_input, double Jxy_input, double Jz_input, double Bz_input, double Bx_input, int Ly_input, bool CALC_SQUARE, size_t D_input)
:MpoQ<Sym::U0> (Lx_input, Ly_input, GrandHeisenbergModel::qloc(Ly_input,D_input), vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), {}, labeldummy, ""),
Jxy(Jxy_input), Jz(Jz_input), Bz(Bz_input), Bx(Bx_input), D(D_input)
{
	if (Jz==numeric_limits<double>::infinity()) {Jz=Jxy;} // default: Jxy=Jz
	assert(Jxy != 0. or Jz != 0.);
	this->label = HeisenbergModel::create_label(D,Jxy,Jz,0.,Bz,Bx);
	
	B = SpinBase<Symmetry>(N_legs,D);
	HamiltonianTermsXd<Symmetry> Terms = HeisenbergModel::set_operators(B, Jxy,Jz,Bz,Bx);

	SuperMatrix<Symmetry,double> G = ::Generator(Terms);
	this->Daux = Terms.auxdim();
	this->construct(G, this->W, this->Gvec);
	
	if (CALC_SQUARE == true)
	{
		vector<qType> qOpSq_;
		qOpSq_.push_back({});
		vector<vector<qType> > qOpSq(this->N_sites,qOpSq_);
		this->setOpBasisSq(qOpSq);
		this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
		this->GOT_SQUARE = true;
	}
	else
	{
		this->GOT_SQUARE = false;
	}
}

MpoQ<Sym::U0> GrandHeisenbergModel::
Sz (size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
//	MpoQ<Sym::U0 > Mout(N_sites, N_legs, vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)), vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)),
//						{}, labeldummy, "");
	MpoQ<Sym::U0 > Mout(N_sites, N_legs, qarray<0>{}, vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "");
	for (size_t l=0; l<N_sites; ++l)
	{
		Mout.setLocBasis(B.basis(),l);
	}
	Mout.setLocal(loc, B.Scomp(SZ));
	return Mout;
}

MpoQ<Sym::U0> GrandHeisenbergModel::
SzSz (size_t loc1, size_t loc2)
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "Sz(" << loc1 << ")" <<  "Sz(" << loc2 << ")";
//	MpoQ<Sym::U0 > Mout(N_sites, N_legs, vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)), vector<qarray<0> >(begin(qloc2dummy),end(qloc2dummy)),
//						{}, labeldummy, "");
	MpoQ<Sym::U0 > Mout(N_sites, N_legs, qarray<0>{}, vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "");
	for (size_t l=0; l<N_sites; ++l)
	{
		Mout.setLocBasis(B.basis(),l);
	}
	Mout.setLocal({loc1, loc2}, {B.Scomp(SZ), B.Scomp(SZ)});
	return Mout;
}

}

#endif
