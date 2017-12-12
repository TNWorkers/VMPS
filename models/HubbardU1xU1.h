#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

#include "symmetry/U1xU1.h"
#include "MpoQ.h"
#include "FermionBase.h"

namespace VMPS
{

/**MPO representation of 
\f$
H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
    - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
    + U \sum_i n_{i\uparrow} n_{i\downarrow}
    + V \sum_{<ij>} n_{i} n_{j}
\f$.
\note If the nnn-hopping is positive, the ground state energy is lowered.
\warning \f$J>0\f$ is antiferromagnetic*/
class HubbardU1xU1 : public MpoQ<Sym::U1xU1<double>,double>
{
typedef Sym::U1xU1<double> Symmetry;

public:
	
	HubbardU1xU1() : MpoQ(){};
	HubbardU1xU1 (variant<size_t,std::array<size_t,2> > L, vector<Param> params);
	
	template<typename Symmetry_> 
	static HamiltonianTermsXd<Symmetry_> set_operators (const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc=0);
	
//	/**single-site local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
//	The quantum numbers are \f$N_{\uparrow}\f$ and \f$N_{\downarrow}\f$. Used by default.*/
//	static const std::array<qarray<2>,4> qssNupNdn;
//	
//	/**local basis: \f$\{ \left|0,0\right>, \left|\uparrow,0\right>, \left|0,\downarrow\right>, \left|\uparrow\downarrow\right> \}\f$.
//	The quantum numbers are \f$N=N_{\uparrow}+N_{\downarrow}\f$ and \f$2M=N_{\uparrow}-N_{\downarrow}\f$. Used in combination with KondoModel.*/
//	static const std::array<qarray<2>,4> qssNM;
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,2> Nlabel;
	
	static const vector<qarray<2> > qOp();
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,HubbardU1xU1,double>          Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry,double>                           OperatorXd;
	typedef MpoQ<Symmetry,complex<double> >                 OperatorXcd;
	///@}
	
	///@{
	MpoQ<Symmetry> Auger (size_t locx, size_t locy=0);
	MpoQ<Symmetry,complex<double> > doublonPacket (complex<double> (*f)(int));
	MpoQ<Symmetry> eta();
	MpoQ<Symmetry> Aps (size_t locx, size_t locy=0);
	MpoQ<Symmetry> c (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry,complex<double> > electronPacket (complex<double> (*f)(int));
	MpoQ<Symmetry,complex<double> > holePacket (complex<double> (*f)(int));
	MpoQ<Symmetry> d (size_t locx, size_t locy=0);
	MpoQ<Symmetry> dtot();
	MpoQ<Symmetry> s (size_t locx, size_t locy=0);
	MpoQ<Symmetry> n (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> nn (SPIN_INDEX sigma1, size_t locx1, SPIN_INDEX sigma2, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<Symmetry> hh (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<Symmetry> Sz (size_t locx, size_t locy=0);
	MpoQ<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<Symmetry> SaSa (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t locy1=0, size_t locy2=0);
	MpoQ<Symmetry> triplon (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy=0);
	MpoQ<Symmetry> quadruplon (size_t locx, size_t locy=0);
	///@}
	
protected:
	
	const std::map<string,std::any> defaults = 
	{
		{"U",0.}, {"V",0.}, {"Bz",0.}, {"J",0.}, {"mu",0.},
		{"t",1.}, {"tPara",0.}, {"tPerp",0.},
		{"tPrime",0.}, {"J3site",0.},
		{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
	};
	
	FermionBase<Symmetry> F;
};

//const std::array<qarray<2>,4> HubbardU1xU1::qssNupNdn {qarray<2>{0,0}, qarray<2>{1,0}, qarray<2>{0,1},  qarray<2>{1,1}};
//const std::array<qarray<2>,4> HubbardU1xU1::qssNM     {qarray<2>{0,0}, qarray<2>{1,1}, qarray<2>{1,-1}, qarray<2>{2,0}};
const std::array<string,2>    HubbardU1xU1::Nlabel {"N↑","N↓"};

const vector<qarray<2> > HubbardU1xU1::
qOp()
{
	vector<qarray<2> > vout;
	vout.push_back({0,0});
	vout.push_back({+1,0});
	vout.push_back({-1,0});
	vout.push_back({0,+1});
	vout.push_back({0,-1});
	vout.push_back({+1,-1});
	vout.push_back({-1,+1});
	return vout;
}

HubbardU1xU1::
HubbardU1xU1 (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0], 
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1], 
                 qarray<Symmetry::Nq>({0,0}), HubbardU1xU1::qOp(), HubbardU1xU1::Nlabel, "")
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		F = FermionBase<Symmetry>(N_legs,!isfinite(P.get<double>("U",l%Lcell)));
		setLocBasis(F.get_basis(),l);
		
		Terms[l] = set_operators(F,P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l])); // boost::multi_array has stupid assignment
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> HubbardU1xU1::
set_operators (const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry_> Terms;
	
	stringstream ss;
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	// hopping terms
	
	double t = P.get_default<double>("t");
	
	ArrayXXd tPara(F.orbitals(),F.orbitals()); tPara.setZero();
	tPara.matrix().diagonal().setConstant(t);
	
	if (P.HAS("t",loc))
	{
		t = P.get<double>("t",loc);
		tPara.matrix().diagonal().setConstant(t);
		ss << "t=" << t;
	}
	else if (P.HAS("tPara",loc))
	{
		tPara = P.get<ArrayXXd>("tPara",loc);
		ss << ",t∥=" << tPara.format(CommaInitFmt);
	}
	else
	{
		ss << "t=" << t; // print hopping first no matter what
	}
	
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(UP,i), F.sign() * F.c(UP,j)));
			Terms.tight.push_back(make_tuple(-tPara(i,j), F.cdag(DN,i), F.sign() * F.c(DN,j)));
			Terms.tight.push_back(make_tuple(+tPara(i,j), F.c(UP,i),    F.sign() * F.cdag(UP,j)));
			Terms.tight.push_back(make_tuple(+tPara(i,j), F.c(DN,i),    F.sign() * F.cdag(DN,j)));
		}
	}
	
	// V & J terms
	
	double V = P.get_default<double>("V");
	double J = P.get_default<double>("J");
	
	if (P.HAS("V",loc))
	{
		V = P.get<double>("V");
		for (int i=0; i<F.orbitals(); ++i)
		{
			Terms.tight.push_back(make_tuple(V, F.n(i), F.n(i)));
		}
		ss << ",V=" << V;
	}
	if (P.HAS("J",loc))
	{
		V = P.get<double>("J");
		for (int i=0; i<F.orbitals(); ++i)
		{
			Terms.tight.push_back(make_tuple(0.5*J, F.Sp(i), F.Sm(i)));
			Terms.tight.push_back(make_tuple(0.5*J, F.Sm(i), F.Sp(i)));
			Terms.tight.push_back(make_tuple(J,     F.Sz(i), F.Sz(i)));
		}
		ss << ",J=" << J;
	}
	
	/// NNN-terms
	
	double tPrime = P.get_default<double>("tPrime");
	double J3site = P.get_default<double>("J3site");
	
	if (P.HAS("tPrime",loc))
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");
		tPrime = P.get<double>("tPrime");
		
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(UP), F.sign() * F.c(UP),    F.sign()));
		Terms.nextn.push_back(make_tuple(-tPrime, F.cdag(DN), F.sign() * F.c(DN),    F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(UP),    F.sign() * F.cdag(UP), F.sign()));
		Terms.nextn.push_back(make_tuple(+tPrime, F.c(DN),    F.sign() * F.cdag(DN), F.sign()));
	}
	
	/**MPO representation of \f$H = H_{3-site}\f$ with:
	- \f$ H_{3-site} = - \frac{J}{4} \sum_{<ijk>\sigma} (c^\dagger_{i\sigma} n_{j,-\sigma} c_{k\sigma} 
	                                                   - c^\dagger_{i\sigma} S^{-\sigma}_j c_{k,-\sigma} + h.c.) \f$
	\note useful reference: 
	"Effect of the Three-Site Hopping Term on the t-J Model" 
	(Ammon, Troyer, Tsunetsugu, 1995), 
	http://arxiv.org/pdf/cond-mat/9502037v1.pdf
	*/
	if (P.HAS("J3site",loc))
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with 3-site terms!");
		J3site = P.get<double>("J3site");
		
		// three-site terms without spinflip
		Terms.nextn.push_back(make_tuple(-0.25*J3site, F.cdag(UP), F.sign()*F.c(UP),    F.n(DN)*F.sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site, F.cdag(DN), F.sign()*F.c(DN),    F.n(UP)*F.sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site, F.c(UP),    F.sign()*F.cdag(UP), F.n(DN)*F.sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site, F.c(DN),    F.sign()*F.cdag(DN), F.n(UP)*F.sign()));
		
		// three-site terms with spinflip
		Terms.nextn.push_back(make_tuple(+0.25*J3site, F.cdag(DN), F.sign()*F.c(UP),    F.Sp()*F.sign()));
		Terms.nextn.push_back(make_tuple(+0.25*J3site, F.cdag(UP), F.sign()*F.c(DN),    F.Sm()*F.sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site, F.c(DN),    F.sign()*F.cdag(UP), F.Sm()*F.sign()));
		Terms.nextn.push_back(make_tuple(-0.25*J3site, F.c(UP),    F.sign()*F.cdag(DN), F.Sp()*F.sign()));
		
		ss << ",J3site=" << J3site;
	}
	
	// local terms
	
	// U
	
	ArrayXd Uloc(F.orbitals()); Uloc.setZero();
	double U = P.get_default<double>("U");
	
	if (P.HAS("Uloc",loc))
	{
		Uloc = P.get<double>("Uloc",loc);
		ss << ",U=" << Uloc.format(CommaInitFmt);
	}
	else if (P.HAS("U",loc))
	{
		U = P.get<double>("U",loc);
		Uloc = U;
		ss << ",U=" << U;
	}
	
	// t⟂
	
	double tPerp = P.get_default<double>("tPerp");
	
	if (P.HAS("t",loc))
	{
		tPerp = P.get<double>("t",loc);
	}
	else if (P.HAS("tPerp",loc))
	{
		tPerp = P.get<double>("tPerp",loc);
		ss << ",t⟂=" << tPerp;
	}
	
	// mu
	
	ArrayXd muloc(F.orbitals()); muloc.setZero();
	double mu = P.get_default<double>("mu");
	
	if (P.HAS("muloc",loc))
	{
		muloc = P.get<double>("muloc",loc);
		ss << ",mu=" << muloc.format(CommaInitFmt);
	}
	if (P.HAS("mu",loc))
	{
		mu = P.get<double>("mu",loc);
		muloc = mu;
		ss << ",mu=" << mu;
	}
	
	// Bz
	
//	ArrayXd Bzloc(F.orbitals()); Bzloc.setZero();
//	double Bz = P.get_default<double>("Bz");
//	
//	if (P.HAS("Bzloc",loc))
//	{
//		Bzloc = P.get<double>("Bzloc",loc);
//		ss << ",Bz=" << Bzloc.format(CommaInitFmt);
//	}
//	else if (P.HAS("Bz",loc))
//	{
//		Bz = P.get<double>("Bz",loc);
//		Bzloc = Bz;
//		ss << ",Bz=" << Bz;
//	}
	
	auto [Bz,Bzloc,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F.orbitals(),loc);
	Terms.info.push_back(Bzlabel);
	
	if (isfinite(Uloc.sum()))
	{
		Terms.name = "Hubbard";
	}
	else
	{
		Terms.name = (P.HAS("J") or P.HAS("J3site"))? "t-J":"U=∞-Hubbard";
	}
	Terms.info = ss.str();
	
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(Uloc,muloc,Bzloc,tPerp,V,J, P.get<bool>("CYLINDER"))));
	
	return Terms;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
Auger (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Auger(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({-1,-1}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal(locx, F.c(UP,locy)*F.c(DN,locy));
	return Mout;
}

//MpoQ<Sym::U1xU1<double>,complex<double> > HubbardU1xU1::
//doublonPacket (complex<double> (*f)(int))
//{
//	stringstream ss;
//	ss << "doublonPacket";
//	
//	MpoQ<Symmetry,complex<double> > Mout(N_sites, N_legs, qarray<Symmetry::Nq>({-1,-1}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
//	
//	Mout.setLocalSum(F.c(UP)*F.c(DN), f);
//	return Mout;
//}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
eta()
{
	assert(N_legs == 1);
	stringstream ss;
	ss << "eta";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({-1,-1}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocalSum(F.c(UP)*F.c(DN), stagger);
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
Aps (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Aps(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({+1,+1}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal(locx, F.cdag(DN,locy)*F.cdag(UP,locy));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
c (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "c(" << locx << "," << locy << ",σ=" << sigma << ")";
	
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-1,0} : qdiff = {0,-1};
	
	vector<SuperMatrix<Symmetry,double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = (sigma==UP)? F.sign_local(locy)*F.c(UP,locy) : F.sign_local(locy)*F.c(DN,locy);
	for (size_t l=locx+1; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.Id();
	}
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>(qdiff), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
cdag (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "c†(" << locx << "," << locy << ",σ=" << sigma << ")";
	
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+1,0} : qdiff = {0,+1};
	
	vector<SuperMatrix<Symmetry,double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = (sigma==UP)? F.sign_local(locy)*F.cdag(UP,locy) : F.sign_local(locy)*F.cdag(DN,locy);
	for (size_t l=locx+1; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.Id();
	}
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>(qdiff), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	return Mout;
}

//MpoQ<Sym::U1xU1<double>,complex<double> > HubbardU1xU1::
//electronPacket (complex<double> (*f)(int))
//{
//	assert(N_legs==1);
//	stringstream ss;
//	ss << "electronPacket";
//	
//	qarray<2> qdiff = {+1,0};
//	
//	vector<SuperMatrix<Symmetry,complex<double> > > M(N_sites);
//	M[0].setRowVector(2,F.dim());
////	M[0](0,0) = f(0) * F.cdag(UP);
//	M[0](0,0).data = f(0) * F.cdag(UP).data; M[0](0,0).Q = F.cdag(UP).Q;
//	M[0](0,1) = F.Id();
//	
//	for (size_t l=1; l<N_sites-1; ++l)
//	{
//		M[l].setMatrix(2,F.dim());
////		M[l](0,0) = complex<double>(1.,0.) * F.sign();
//		M[l](0,0).data = complex<double>(1.,0.) * F.sign().data; M[l](0,0).Q = F.sign().Q;
////		M[l](1,0) = f(l) * F.cdag(UP);
//		M[l](1,0).data = f(l) * F.cdag(UP).data; M[l](1,0).Q = F.cdag(UP).Q;
//		M[l](0,1).setZero();
//		M[l](1,1) = F.Id();
//	}
//	
//	M[N_sites-1].setColVector(2,F.dim());
////	M[N_sites-1](0,0) = complex<double>(1.,0.) * F.sign();
//	M[N_sites-1](0,0).data = complex<double>(1.,0.) * F.sign().data; M[N_sites-1](0,0).Q = F.sign().Q;
////	M[N_sites-1](1,0) = f(N_sites-1) * F.cdag(UP);
//	M[N_sites-1](1,0).data = f(N_sites-1) * F.cdag(UP).data; M[N_sites-1](1,0).Q = F.cdag(UP).Q;
//	
//	MpoQ<Symmetry,complex<double> > Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>(qdiff), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
//	return Mout;
//}

//MpoQ<Sym::U1xU1<double>,complex<double> > HubbardU1xU1::
//holePacket (complex<double> (*f)(int))
//{
//	assert(N_legs==1);
//	stringstream ss;
//	ss << "holePacket";
//	
//	qarray<2> qdiff = {-1,0};
//	
//	vector<SuperMatrix<Symmetry,complex<double> > > M(N_sites);
//	M[0].setRowVector(2,F.dim());
//	M[0](0,0) = f(0) * F.c(UP);
//	M[0](0,1) = F.Id();
//	
//	for (size_t l=1; l<N_sites-1; ++l)
//	{
//		M[l].setMatrix(2,F.dim());
//		M[l](0,0) = complex<double>(1.,0.) * F.sign();
//		M[l](1,0) = f(l) * F.c(UP);
//		M[l](0,1).setZero();
//		M[l](1,1) = F.Id();
//	}
//	
//	M[N_sites-1].setColVector(2,F.dim());
//	M[N_sites-1](0,0) = complex<double>(1.,0.) * F.sign();
//	M[N_sites-1](1,0) = f(N_sites-1) * F.c(UP);
//	
//	MpoQ<Symmetry,complex<double> > Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>(qdiff), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
//	return Mout;
//}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
	
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
	vector<SuperMatrix<Symmetry,double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	// c(locx,UP)*c(locx,DN)
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
	// c(locx+1,UP|DN)
	M[locx+1].setMatrix(1,F.dim());
	M[locx+1](0,0) = (sigma==UP)? F.c(UP,locy) : F.c(DN,locy);
	for (size_t l=locx+2; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.Id();
	}
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>(qdiff), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
	
	qarray<2> qdiff;
	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
	vector<SuperMatrix<Symmetry,double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.sign();
	}
	// c†(locx,DN)*c†(locx,UP)
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = F.cdag(DN,locy)*F.cdag(UP,locy);
	// c†(locx+1,UP|DN)
	M[locx+1].setMatrix(1,F.dim());
	M[locx+1](0,0) = (sigma==UP)? F.cdag(UP,locy) : F.cdag(DN,locy);
	for (size_t l=locx+2; l<N_sites; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.Id();
	}
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>(qdiff), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
quadruplon (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
	
	vector<SuperMatrix<Symmetry,double> > M(N_sites);
	for (size_t l=0; l<locx; ++l)
	{
		M[l].setMatrix(1,F.dim());
		M[l](0,0) = F.Id();
	}
	// c(loc,UP)*c(loc,DN)
	M[locx].setMatrix(1,F.dim());
	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
	// c(loc+1,UP)*c(loc+1,DN)
	M[locx+1].setMatrix(1,F.dim());
	M[locx+1](0,0) = F.c(UP,locy)*F.c(DN,locy);
	for (size_t l=locx+2; l<N_sites; ++l)
	{
		M[l].setMatrix(1,4);
		M[l](0,0) = F.Id();
	}
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, M, qarray<Symmetry::Nq>({-2,-2}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
d (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal(locx, F.d(locy));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
dtot()
{
	stringstream ss;
	ss << "double_occ_total";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocalSum(F.d());
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
s (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "single_occ(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal(locx, F.n(UP,locy)+F.n(DN,locy)-2.*F.d(locy));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
n (SPIN_INDEX sigma, size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "n(" << locx << "," << locy << ",σ=" << sigma << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal(locx, F.n(sigma,locy));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
hh (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << "h(" << locx1 << "," << locy1 << ")h" << "(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {F.d(locy1)-F.n(locy1)+F.Id(),F.d(locy2)-F.n(locy2)+F.Id()});
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
nn (SPIN_INDEX sigma1, size_t locx1, SPIN_INDEX sigma2, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << "n(" << locx1 << "," << locy1 << ")n" << "(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {F.n(sigma1,locy1),F.n(sigma2,locy2)});
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
Sz (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<N_legs);
	stringstream ss;
	ss << "Sz(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal(locx, F.Sz(locy));
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << "Sz(" << locx1 << "," << locy1 << ")" << "Sz(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>({0,0}), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {F.Sz(locy1),F.Sz(locy2)});
	return Mout;
}

MpoQ<Sym::U1xU1<double> > HubbardU1xU1::
SaSa (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t locy1, size_t locy2)
{
	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
	stringstream ss;
	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs, qarray<Symmetry::Nq>(F.getQ(SOP1)+F.getQ(SOP2)), {{0,0}}, HubbardU1xU1::Nlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F.get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {F.Scomp(SOP1,locy1),F.Scomp(SOP2,locy2)});
	return Mout;
}

}

#endif
