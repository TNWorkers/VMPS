#ifndef HUBBARDMODELSU2XU1_H_
#define HUBBARDMODELSU2XU1_H_

#include <variant>

#include "bases/fermions/BaseSU2xU1.h"
#include "symmetry/SU2xU1.h"
#include "Mpo.h"
#include "DmrgExternal.h"
#include "ParamHandler.h"

namespace VMPS
{

/** \class HubbardSU2xU1
  * \ingroup Hubbard
  *
  * \brief Hubbard Model
  *
  * MPO representation of 
  * 
  \f$
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  + U \sum_i n_{i\uparrow} n_{i\downarrow}
  + V \sum_{<ij>} n_{i} n_{j}
  \f$.
  *
  \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
  \note If the nnn-hopping is positive, the ground state energy is lowered.
  \warning \f$J>0\f$ is antiferromagnetic
  \todo Implement more observables.
  */
class HubbardSU2xU1 : public MpoQ<Sym::SU2xU1<double> ,double>
{
public:
	typedef Sym::SU2xU1<double> Symmetry;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HubbardSU2xU1() : MpoQ(){};
	HubbardSU2xU1 (const size_t &L, const vector<Param> &params);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const fermions::BaseSU2xU1<> &F, const ParamHandler &P, size_t loc=0);
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,Symmetry::Nq> SNlabel;
	
	///@{
	MpoQ<Symmetry> cc (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> eta(size_t locx, size_t locy=0);
//	MpoQ<Symmetry> Aps (size_t locx, size_t locy=0);
	MpoQ<Symmetry> c (size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdag (size_t locx, size_t locy=0);
	MpoQ<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	MpoQ<Symmetry> d (size_t locx, size_t locy=0);
	MpoQ<Symmetry> n (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> S (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> Sdag (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> SSdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
//	MpoQ<Symmetry> EtaEtadag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);	
//	MpoQ<Symmetry> triplon (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> antitriplon (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> quadruplon (size_t locx, size_t locy=0);
	///@}
	
	static const map<string,any> defaults;
	
protected:
	
	vector<fermions::BaseSU2xU1<> > F;
};

const std::array<string,Sym::SU2xU1<double>::Nq> HubbardSU2xU1::SNlabel{"S","N"};

const map<string,any> HubbardSU2xU1::defaults = 
{
	{"t",1.}, {"tPerp",0.}, {"tPrime",0.}, 
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

HubbardSU2xU1::
HubbardSU2xU1 (const size_t &L, const vector<Param> &params)
:MpoQ<Symmetry> (L, qarray<Symmetry::Nq>({1,0}), HubbardSU2xU1::SNlabel, "", SfromD_noFormat)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = fermions::BaseSU2xU1<>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)));
		setLocBasis(F[l].get_basis(),l);
		
		Terms[l] = set_operators(F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, false, P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

HamiltonianTermsXd<Sym::SU2xU1<double> > HubbardSU2xU1::
set_operators (const fermions::BaseSU2xU1<> &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F.orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F.orbitals(),loc);
	save_label(Vlabel);
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",F.orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			// wrong:
//			auto Otmp = OperatorType::prod(F.sign(),F.c(j),{2,-1});
//			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.), F.cdag(i).plain<double>(), Otmp.plain<double>()));
//			
//			Otmp = OperatorType::prod(F.sign(),F.cdag(j),{2,1});
//			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.), F.c(i).plain<double>(), Otmp.plain<double>()));
			
			// correct?:
			auto cdagF = OperatorType::prod(F.cdag(i),F.sign(),{2,+1});
			auto cF    = OperatorType::prod(F.c(i),   F.sign(),{2,-1});
			
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.), cdagF.plain<double>(), F.c(j).plain<double>()));
			// SU(2) spinors commute on different sites, hence no sign flip here:
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.), cF.plain<double>(),    F.cdag(j).plain<double>()));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j), F.n(i).plain<double>(), F.n(j).plain<double>()));
		}
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-sqrt(3)*Jpara(i,j), F.Sdag(i).plain<double>(), F.S(j).plain<double>()));
		}
	}
	
	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);
	
	if (tPrime.x != 0.)
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");
		
		// wrong:
//		auto Otmp = OperatorType::prod(F.sign(),F.c(),{2,-1});
//		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.), F.cdag().plain<double>(), Otmp.plain<double>(), F.sign().plain<double>()));
//		Otmp = OperatorType::prod(F.sign(),F.cdag(),{2,1});
//		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.), F.c().plain<double>(), Otmp.plain<double>(), F.sign().plain<double>()));
		
		// correct?:
		auto cF    = OperatorType::prod(F.c(),   F.sign(),{2,-1});
		auto cdagF = OperatorType::prod(F.cdag(),F.sign(),{2,+1});
		/**\todo: think about crazy fermionic signs here:*/
		Terms.nextn.push_back(make_tuple(+tPrime.x*sqrt(2.), cdagF.plain<double>(), F.c().plain<double>(),    F.sign().plain<double>()));
		Terms.nextn.push_back(make_tuple(+tPrime.x*sqrt(2.), cF.plain<double>()   , F.cdag().plain<double>(), F.sign().plain<double>()));
	}
	
	// local terms
	
	// Hubbard-U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F.orbitals(),loc);
	save_label(Ulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F.orbitals(),loc);
	save_label(t0label);
	
	// μ
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	save_label(mulabel);
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// V⟂
	param0d Vperp = P.fill_array0d<double>("V","Vperp",loc);
	save_label(Vperp.label);
	
	// J⟂
	param0d Jperp = P.fill_array0d<double>("J","Jperp",loc);
	save_label(Jperp.label);
	
	Terms.local.push_back(make_tuple(1.,F.HubbardHamiltonian(Uorb,t0orb-muorb,tPerp.x,Vperp.x,Jperp.x, P.get<bool>("CYLINDER")).plain<double>()));
	
	Terms.name = "Hubbard SU(2)⊗U(1)";
	
	return Terms;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
c (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, {2,-1}, HubbardSU2xU1::SNlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	/**\todo: think about crazy fermionic signs here:*/
	Mout.setLocal(locx, pow(-1.,locx+1)*F[locx].c(locy).plain<double>(), F[0].sign().plain<double>());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
cdag (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c†(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(N_sites, {2,+1}, HubbardSU2xU1::SNlabel, ss.str());
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	/**\todo: think about crazy fermionic signs here:*/
	Mout.setLocal(locx, pow(-1.,locx+1)*F[locx].cdag(locy).plain<double>(), F[0].sign().plain<double>());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, Symmetry::qvacuum(), HubbardSU2xU1::SNlabel, ss.str(), SfromD_noFormat);
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }
	
	auto cdag = F[locx1].cdag(locy1);
	auto c    = F[locx2].c   (locy2);
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
	}
	/**\todo: think about crazy fermionic signs here:*/
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.)*OperatorType::prod(cdag, F[locx1].sign(), {2,+1}).plain<double>(), 
		                               pow(-1.,locx2-locx1+1)*c.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	else if (locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.)*OperatorType::prod(c, F[locx2].sign(), {2,-1}).plain<double>(), 
		                               pow(-1.,locx1-locx2+1)*cdag.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
d (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "double_occ(" << locx << "," << locy << ")";
	
	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, Symmetry::qvacuum(), HubbardSU2xU1::SNlabel, ss.str(), SfromD_noFormat);
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }
	
	Mout.setLocal(locx, F[locx].d(locy).plain<double>());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
cc (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c(" << locx << "," << locy << "," << UP << ")"
	   << "c(" << locx << "," << locy << "," << DN << ")";
	
	MpoQ<Symmetry> Mout(N_sites, {1,-2}, HubbardSU2xU1::SNlabel, ss.str());
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }
	
	Mout.setLocal(locx, F[locx].Eta(locy).plain<double>());
	return Mout;
}

MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
n (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "n(" << locx << "," << locy << ")";
	
	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, Symmetry::qvacuum(), HubbardSU2xU1::SNlabel, ss.str());
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }
	
	Mout.setLocal(locx, F[locx].n(locy).plain<double>());
	return Mout;
}

//// MpoQ<SymSU2<double> > HubbardSU2xU1::
//// S (size_t locx, size_t locy)
//// {
//// 	assert(locx<N_sites and locy<F[locx].dim());
//// 	stringstream ss;
//// 	ss << "S(" << locx << "," << locy << ")";

//// 	vector<vector<qType> > qOptmp(N_sites);
//// 	for (size_t l=0; l<N_sites; l++)
//// 	{
//// 		qOptmp[l].resize(1);
//// 		qOptmp[l][0] = (l == locx) ? 3 : 1;
//// 	}

//// 	MpoQ<Symmetry> Mout(N_sites, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2xU1::Slabel, ss.str());
//// 	Mout.setLocal(locx, F.S(locy));
//// 	return Mout;
//// }

//// MpoQ<SymSU2<double> > HubbardSU2xU1::
//// Sdag (size_t locx, size_t locy)
//// {
//// 	assert(locx<N_sites and locy<F[locx].dim());
//// 	stringstream ss;
//// 	ss << "S†(" << locx << "," << locy << ")";

//// 	vector<vector<qType> > qOptmp(N_sites);
//// 	for (size_t l=0; l<N_sites; l++)
//// 	{
//// 		qOptmp[l].resize(1);
//// 		qOptmp[l][0] = (l == locx) ? 3 : 1;
//// 	}

//// 	MpoQ<Symmetry> Mout(N_sites, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2xU1::Slabel, ss.str());
//// 	Mout.setLocal(locx, F.Sdag(locy));
//// 	return Mout;
//// }

//MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
//SSdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1<this->N_sites and locx2<this->N_sites);
//	stringstream ss;
//	ss << "S†(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";

//	MpoQ<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }

//	auto Sdag = F.Sdag(locy1);
//	auto S = F.S(locy2);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xU1::Slabel;
//	if(locx1 == locx2)
//	{
//		auto product = sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
//		Mout.setLocal(locx1,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else
//	{
//		Mout.setLocal({locx1, locx2}, {sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
//		return Mout;
//	}
//}

//MpoQ<Sym::SU2xU1<double> > HubbardSU2xU1::
//EtaEtadag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1<this->N_sites and locx2<this->N_sites);
//	stringstream ss;
//	ss << "η†(" << locx1 << "," << locy1 << ")" << "η(" << locx2 << "," << locy2 << ")";

//	MpoQ<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }

//	auto Etadag = F.Etadag(locy1);
//	auto Eta = F.Eta(locy2);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xU1::Slabel;
//	if(locx1 == locx2)
//	{
//		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
//		Mout.setLocal(locx1,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else
//	{
//		Mout.setLocal({locx1, locx2}, {Etadag, Eta}, {{1,2},{1,-2}});
//		return Mout;
//	}
//}


















// MpoQ<SymSU2<double> > HubbardSU2xU1::
// SSdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
// 	stringstream ss;
// 	ss << "S†S(" << locx1 << "," << locy1 << ")" << "Sz(" << locx2 << "," << locy2 << ")";
// 	vector<vector<qType> > qOptmp(N_sites);
// 	for (size_t l=0; l<N_sites; l++)
// 	{
// 		qOptmp[l].resize(1);
// 		qOptmp[l][0] = (l == locx1 or l == locx2) ? 3 : 1;
// 	}

// 	MpoQ<Symmetry> Mout(N_sites, MpoQ<Symmetry>::qloc, qOptmp, {1}, HubbardSU2xU1::Slabel, ss.str());
// 	Mout.setLocal({locx1,locx2}, {F.S(locy1),F.Sdag(locy2)});
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<F[locx].dim());
// 	stringstream ss;
// 	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
// 	qstd::array<2> qdiff;
// 	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0) = F.sign();
// 	}
// 	// c(locx,UP)*c(locx,DN)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	// c(locx+1,UP|DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = (sigma==UP)? F.c(UP,locy) : F.c(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2xU1::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<F[locx].dim());
// 	stringstream ss;
// 	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
// 	qstd::array<2> qdiff;
// 	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0) = F.sign();
// 	}
// 	// c†(locx,DN)*c†(locx,UP)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.cdag(DN,locy)*F.cdag(UP,locy);
// 	// c†(locx+1,UP|DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = (sigma==UP)? F.cdag(UP,locy) : F.cdag(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2xU1::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2xU1::
// quadruplon (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<F[locx].dim());
// 	stringstream ss;
// 	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
// 	// c(loc,UP)*c(loc,DN)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	// c(loc+1,UP)*c(loc+1,DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,4);
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return MpoQ<Symmetry>(N_sites, M, MpoQ<Symmetry>::qloc, {-2,-2}, HubbardSU2xU1::Nlabel, ss.str());
// }

} // end namespace VMPS::models

#endif
