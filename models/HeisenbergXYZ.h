#ifndef STRAWBERRY_HEISENBERGXYZ
#define STRAWBERRY_HEISENBERGXYZ

#include "models/Heisenberg.h"

namespace VMPS
{

class HeisenbergXYZ : public MpoQ<Sym::U0,complex<double> >
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	//---constructors---
	///\{
	HeisenbergXYZ() : MpoQ<Symmetry,complex<double> >() {};
	
	/**
	   \param Lx_input : chain length
	   \describe_params
	   \param Ly_input : amount of legs in ladder
	   \param CALC_SQUARE : If \p true, calculates and stores \f$H^2\f$
	*/
	HeisenbergXYZ (variant<size_t,std::array<size_t,2> > L, vector<Param> params);
	///\}
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
//	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
//	typedef DmrgSolverQ<Symmetry,Heisenberg>                Solver;
//	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
//	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
//	typedef MpoQ<Symmetry>                                  Operator;
	///@}
	
//	///@{
//	/**Observables.*/
//	MpoQ<Symmetry> SzSz (size_t loc1, size_t loc2);
//	MpoQ<Symmetry> Sz   (size_t loc);
//	///@}
	
	void add_operators (HamiltonianTerms<Symmetry,complex<double> > &T, const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc);
	
protected:
	
	SpinBase<Symmetry> B;
};

void HeisenbergXYZ::
add_operators (HamiltonianTerms<Symmetry,complex<double> > &Terms, const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc)
{
	stringstream ss;
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	// Heisenberg terms
	
	double Jx = P.get_default<double>("Jx");
	double Jy = P.get_default<double>("Jy");
	ArrayXXd Jxpara(B.orbitals(),B.orbitals()); Jxpara.setZero();
	ArrayXXd Jypara(B.orbitals(),B.orbitals()); Jypara.setZero();
	
	if (P.HAS("Jx",loc))
	{
		Jx = P.get<double>("Jx",loc);
		Jxpara.matrix().diagonal().setConstant(Jx);
		ss << ",Jx=" << Jx << B.alignment(Jx);
	}
	else if (P.HAS("Jxpara",loc))
	{
		if (P.HAS("Jxpara",loc))
		{
			Jxpara = P.get<ArrayXXd>("Jxpara",loc);
		}
		ss << ",Jx=" << Jx;
	}
	if (P.HAS("Jy",loc))
	{
		Jy = P.get<double>("Jy",loc);
		Jypara.matrix().diagonal().setConstant(Jy);
		ss << ",Jy=" << Jy << B.alignment(Jy);
	}
	else if (P.HAS("Jypara",loc))
	{
		if (P.HAS("Jypara",loc))
		{
			Jypara = P.get<ArrayXXd>("Jypara",loc);
		}
		ss << ",Jy=" << Jy;
	}
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jxpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jxpara(i,j), B.Scomp(SX,i).template cast<complex<double> >(), 
			                                              B.Scomp(SX,j).template cast<complex<double> >()));
		}
		if (Jypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jypara(i,j), -1.i*B.Scomp(iSY,i).cast<complex<double> >(), 
			                                              -1.i*B.Scomp(iSY,j).cast<complex<double> >()));
		}
	}
	
	// DM terms
	
	SiteOperator<Symmetry,complex<double> > Sx = B.Scomp(SX).cast<complex<double> >();
	SiteOperator<Symmetry,complex<double> > Sy = -1.i*B.Scomp(iSY).cast<complex<double> >();
	SiteOperator<Symmetry,complex<double> > Sz = B.Scomp(SZ).cast<complex<double> >();
	SiteOperator<Symmetry,complex<double> > Id = B.Id().cast<complex<double> >();
	
	double Dx = P.get_default<double>("Dx");
	double Dz = P.get_default<double>("Dz");
	
	if (P.HAS("Dx") or P.HAS("Dz"))
	{
		Dx = P.get<double>("Dx");
		Dz = P.get<double>("Dz");
		
		if (Dx!=0. or Dz!=0.)
		{
			Terms.tight.push_back(make_tuple(1., Sy, +Dz*Sx-Dx*Sz));
			Terms.tight.push_back(make_tuple(1., +Dx*Sz-Dz*Sx, Sy));
		}
		ss << ",Dx=" << Dx << ",Dz=" << Dz;
	}
	
	double Dxprime = P.get_default<double>("Dx");
	double Dzprime = P.get_default<double>("Dz");
	
	if (P.HAS("Dxprime",loc) or P.HAS("Dzprime",loc))
	{
		Dxprime = P.get<double>("Dxprime",loc);
		Dzprime = P.get<double>("Dzprime",loc);
		
		if (Dxprime!=0. or Dzprime!=0.)
		{
			Terms.nextn.push_back(make_tuple(1., Sy, +Dzprime*Sx-Dxprime*Sz, Id));
			Terms.nextn.push_back(make_tuple(1., +Dxprime*Sz-Dzprime*Sx, Sy, Id));
		}
		
		ss << ",Dx'=" << Dxprime << ",Dz'=" << Dzprime;
	}
	
	Terms.name = (P.HAS("Dx") or P.HAS("Dy") or P.HAS("Dxprime") or P.HAS("Dzprime"))? "Dzyaloshinsky-Moriya":"HeisenbergXYZ";
	Terms.info += ss.str();
}

HeisenbergXYZ::
HeisenbergXYZ (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
:MpoQ<Symmetry,complex<double> > (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0], 
                                  holds_alternative<size_t>(L)? 1        :get<1>(L)[1], 
                                  qarray<0>({}), vector<qarray<0> >(begin(qloc1dummy),end(qloc1dummy)), labeldummy, "")
{
	ParamHandler P(params,HeisenbergU1::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,complex<double> > > G;
	vector<HamiltonianTerms<Symmetry,complex<double> > > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		B = SpinBase<Symmetry>(N_legs, P.get<size_t>("D",l%Lcell));
		setLocBasis(B.get_basis(),l);
		
		Terms[l] = HeisenbergU1::set_operators(B,P,l%Lcell).cast<complex<double> >();
		add_operators(Terms[l],B,P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

}

#endif
