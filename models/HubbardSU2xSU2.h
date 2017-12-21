#ifndef HUBBARDMODELSU2XSU2_H_
#define HUBBARDMODELSU2XSU2_H_

#include <variant>

#include "FermionBaseSU2xSU2.h"
#include "symmetry/SU2xSU2.h"
#include "MpoQ.h"
#include "DmrgExternalQ.h"
#include "ParamHandler.h"

namespace VMPS
{

/** \class HubbardSU2xSU2
  * \ingroup Hubbard
  *
  * \brief Hubbard Model
  *
  * MPO representation of 
  * 
  \f[
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  + U \sum_i \left[\left(n_{i\uparrow}-\frac{1}{2}\right)\left(n_{i\downarrow}-\frac{1}{2}\right) -\frac{1}{4}\right]
  \f]
  *
  \note Take use of the Spin SU(2) symmetry and SU(2) charge symmetry.
  \warning Bipartite hopping structure is mandatory! (Particle-hole symmetry)
  \warning \f$J>0\f$ is antiferromagnetic
  \todo Implement more observables.
  */
class HubbardSU2xSU2 : public MpoQ<Sym::SU2xSU2<double> ,double>
{
public:
	typedef Sym::SU2xSU2<double> Symmetry;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HubbardSU2xSU2() : MpoQ(){};
	HubbardSU2xSU2 (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc=0);
	
	/**Labels the conserved quantum numbers as \f$N_\uparrow\f$, \f$N_\downarrow\f$.*/
	static const std::array<string,Symmetry::Nq> STlabel;
	
//	MpoQ<Symmetry> Auger (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> eta(size_t locx, size_t locy=0);
//	MpoQ<Symmetry> Aps (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> c (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> cdag (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
//	MpoQ<Symmetry> d (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> n (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> S (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> Sdag (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> SSdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
//	MpoQ<Symmetry> EtaEtadag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);	
//	MpoQ<Symmetry> triplon (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> antitriplon (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> quadruplon (size_t locx, size_t locy=0);
	
	static const map<string,any> defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
};

const std::array<string,Sym::SU2xSU2<double>::Nq> HubbardSU2xSU2::STlabel{"S","T"};

const map<string,any> HubbardSU2xSU2::defaults = 
{
	{"t",1.}, {"tPerp",0.},
	{"U",0.}, {"J",0.}, {"Jperp",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
};

HubbardSU2xSU2::
HubbardSU2xSU2 (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0], 
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1], 
                 qarray<Symmetry::Nq>({1,0}), HubbardSU2xSU2::STlabel, "")
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		F[l] = FermionBase<Symmetry>(N_legs,P.get<SUB_LATTICE>("subL",l%Lcell));
		setLocBasis(F[l].get_basis(),l);
	}
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(F,P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, false, P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

HamiltonianTermsXd<Sym::SU2xSU2<double> > HubbardSU2xSU2::
set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F[loc].orbitals(),loc);
	save_label(tlabel);
		
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",F[loc].orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<F[loc].orbitals(); ++i)
	for (int j=0; j<F[loc+1].orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			// auto Otmp = OperatorType::prod(F.sign(),F.c(j),{2,2});
			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.)*sqrt(2.), F.cdag(i).plain<double>(), Otmp.plain<double>()));
			auto cdagF = OperatorType::prod(F[loc].cdag(i),F[loc].sign(),{2,2});
			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.)*sqrt(2.), cdagF.plain<double>(), F[loc+1].c(j).plain<double>()));

			// Otmp = OperatorType::prod(F.sign(),F.cdag(j),{2,1});
			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.), F.c(i).plain<double>(), Otmp.plain<double>()));
		}
		
		// if (Vpara(i,j) != 0.)
		// {
		// 	Terms.tight.push_back(make_tuple(Vpara(i,j), F.n(i).plain<double>(), F.n(j).plain<double>()));
		// }
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-sqrt(3)*Jpara(i,j), F[loc].Sdag(i).plain<double>(), F[loc+1].S(j).plain<double>()));
		}
	}
	
	// NNN terms
	
	// param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	// save_label(tPrime.label);
	
//	if (tPrime.x != 0.)
//	{
//		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");
//		
//		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.), F.cdag(), Operator::prod(F.sign(),F.c(),{2,-1}), F.sign()));
//		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.), F.c(), Operator::prod(F.sign(),F.cdag(),{2,1}), F.sign()));
//	}
	
	// local terms
	
	// Hubbard-U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);
		
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	cout << "tPerp=" << tPerp.x << endl;
	// J⟂
	param0d Jperp = P.fill_array0d<double>("Jperp","Jperp",loc);
	save_label(Jperp.label);
	
	Terms.local.push_back(make_tuple(1.,F[loc].HubbardHamiltonian(Uorb,tPerp.x,0.,Jperp.x, P.get<bool>("CYLINDER")).plain<double>()));
	
	Terms.name = "Hubbard SU(2)⊗SU(2)";
	
	return Terms;
}

//HubbardSU2xSU2::
//HubbardSU2xSU2 (size_t Lx_input, double U_input, double V_input, double tPrime_input, size_t Ly_input, bool CALC_SQUARE, double t_input)
//	:MpoQ<Symmetry> (Lx_input, Ly_input),
//	U(U_input), V(V_input), tPrime(tPrime_input), t(t_input)
//{
//	assert(N_legs>1 and tPrime==0. or N_legs==1 and "Cannot build a ladder with t'-hopping!");

//	//assign stuff
//	this->label = "HubbardSU2⊗U1";
//	stringstream ss;
//	ss << "(U=" << U << ",V=" << V << ",t'=" << tPrime << ")";
//	this->label += ss.str();
//	
//	F = fermions::BaseSU2xU1<>(N_legs,!isfinite(U));
//	for (size_t l=0; l<N_sites; l++)
//	{
//		qloc__[l] = F.basis();
//		qloc[l] = qloc__[l].qloc();
//		qOp[l] = HubbardSU2xSU2::getqOp();		
//	}
//	Qtot = {1,0};
//	qlabel = HubbardSU2xSU2::Slabel;

//	HamiltonianTermsXd<Sym::SU2xSU2<double> > Terms = set_operators(F, U, V, tPrime, t, 0.,false,t);
//	auto G = ::Generator(Terms);
//	this->Daux = Terms.auxdim();
//	
//	this->construct(G);

//	// this->Wloc = F.c();
//	// this->Wlocdag = F.cdag();
//	if (CALC_SQUARE == true)
//	{
//		// this->construct(tensor_product(G,G), this->Wsq, this->GvecSq);
//		this->GOT_SQUARE = true;
//	}
//	else
//	{
//		this->GOT_SQUARE = false;
//	}
//}


// MpoQ<Symmetry> HubbardSU2xSU2::
// Auger (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "Auger(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {1}, HubbardSU2xSU2::Slabel, ss.str());
// 	Mout.setLocal(locx, F.c(UP,locy)*F.c(DN,locy));
// 	return Mout;
// }

// MpoQ<Symmetry> HubbardSU2xSU2::
// eta()
// {
// 	stringstream ss;
// 	ss << "eta";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {-1,-1}, HubbardSU2xSU2::Nlabel, ss.str());
// 	SparseMatrixXd etaloc = MatrixXd::Identity(F.dim(),F.dim()).sparseView();
// 	for (int ly=0; ly<N_legs; ++ly) {etaloc = etaloc * pow(-1.,ly) * F.c(UP,ly)*F.c(DN,ly);}
// 	Mout.setLocalSum(etaloc, true);
// 	return Mout;
// }

// MpoQ<Symmetry> HubbardSU2xSU2::
// Aps (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << "Aps(" << locx << "," << locy << ")";
// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, {+1,+1}, HubbardSU2xSU2::Nlabel, ss.str());
// 	Mout.setLocal(locx, F.cdag(DN,locy)*F.cdag(UP,locy));
// 	return Mout;
// }











//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//c (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<N_legs);
//	stringstream ss;
//	ss << "c(" << locx << "," << locy << ")";

//	MpoQ<Sym::SU2xSU2<double> > Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	Mout.label = ss.str();
//	Mout.setQtarget({2,-1});
//	Mout.qlabel = HubbardSU2xSU2::Slabel;

//	Mout.setLocal(locx, F.c(locy), {2,-1}, F.sign());
//	return Mout;
//}

//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//cdag (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<N_legs);
//	stringstream ss;
//	ss << "c†(" << locx << "," << locy << ")";

//	MpoQ<Sym::SU2xSU2<double> > Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	Mout.label = ss.str();
//	Mout.setQtarget({2,+1});
//	Mout.qlabel = HubbardSU2xSU2::Slabel;

//	Mout.setLocal(locx, F.cdag(locy), {2,+1}, F.sign());
//	return Mout;
//}

//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//cdagc (size_t loc1x, size_t loc2x, size_t loc1y, size_t loc2y)
//{
//	assert(loc1x<this->N_sites and loc2x<this->N_sites);
//	stringstream ss;
//	ss << "c†(" << loc1x << "," << loc1y << ")" << "c(" << loc2x << "," << loc2y << ")";

//	MpoQ<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	auto cdag = F.cdag(loc1y);
//	auto c = F.c(loc2y);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xSU2::Slabel;
//	if(loc1x == loc2x)
//	{
//		auto product = sqrt(2.)*Operator::prod(cdag,c,Symmetry::qvacuum());
//		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else if(loc1x<loc2x)
//	{

//		Mout.setLocal({loc1x, loc2x}, {sqrt(2.)*cdag, Operator::prod(F.sign(),c,{2,-1})}, {{2,1},{2,-1}}, F.sign());
//		return Mout;
//	}
//	else if(loc1x>loc2x)
//	{

//		Mout.setLocal({loc1x, loc2x}, {sqrt(2.)*Operator::prod(F.sign(),cdag,{2,+1}), c}, {{2,1},{2,-1}}, F.sign());
//		return Mout;
//	}
//}

//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//d (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<N_legs);
//	stringstream ss;
//	ss << "double_occ(" << locx << "," << locy << ")";
//	
//	MpoQ<Sym::SU2xSU2<double> > Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xSU2::Slabel;
//	Mout.setLocal(locx, F.d(locy), Symmetry::qvacuum());
//	return Mout;
//}

//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//n (size_t locx, size_t locy)
//{
//	assert(locx<N_sites and locy<N_legs);
//	stringstream ss;
//	ss << "n(" << locx << "," << locy << ")";

//	MpoQ<Sym::SU2xSU2<double> > Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xSU2::Slabel;
//	Mout.setLocal(locx, F.n(locy), Symmetry::qvacuum());
//	return Mout;
//}

//// MpoQ<SymSU2<double> > HubbardSU2xSU2::
//// S (size_t locx, size_t locy)
//// {
//// 	assert(locx<N_sites and locy<N_legs);
//// 	stringstream ss;
//// 	ss << "S(" << locx << "," << locy << ")";

//// 	vector<vector<qType> > qOptmp(N_sites);
//// 	for (size_t l=0; l<N_sites; l++)
//// 	{
//// 		qOptmp[l].resize(1);
//// 		qOptmp[l][0] = (l == locx) ? 3 : 1;
//// 	}

//// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2xSU2::Slabel, ss.str());
//// 	Mout.setLocal(locx, F.S(locy));
//// 	return Mout;
//// }

//// MpoQ<SymSU2<double> > HubbardSU2xSU2::
//// Sdag (size_t locx, size_t locy)
//// {
//// 	assert(locx<N_sites and locy<N_legs);
//// 	stringstream ss;
//// 	ss << "S†(" << locx << "," << locy << ")";

//// 	vector<vector<qType> > qOptmp(N_sites);
//// 	for (size_t l=0; l<N_sites; l++)
//// 	{
//// 		qOptmp[l].resize(1);
//// 		qOptmp[l][0] = (l == locx) ? 3 : 1;
//// 	}

//// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {3}, HubbardSU2xSU2::Slabel, ss.str());
//// 	Mout.setLocal(locx, F.Sdag(locy));
//// 	return Mout;
//// }

//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//SSdag (size_t loc1x, size_t loc2x, size_t loc1y, size_t loc2y)
//{
//	assert(loc1x<this->N_sites and loc2x<this->N_sites);
//	stringstream ss;
//	ss << "S†(" << loc1x << "," << loc1y << ")" << "S(" << loc2x << "," << loc2y << ")";

//	MpoQ<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	auto Sdag = F.Sdag(loc1y);
//	auto S = F.S(loc2y);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xSU2::Slabel;
//	if(loc1x == loc2x)
//	{
//		auto product = sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
//		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else
//	{
//		Mout.setLocal({loc1x, loc2x}, {sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
//		return Mout;
//	}
//}

//MpoQ<Sym::SU2xSU2<double> > HubbardSU2xSU2::
//EtaEtadag (size_t loc1x, size_t loc2x, size_t loc1y, size_t loc2y)
//{
//	assert(loc1x<this->N_sites and loc2x<this->N_sites);
//	stringstream ss;
//	ss << "η†(" << loc1x << "," << loc1y << ")" << "η(" << loc2x << "," << loc2y << ")";

//	MpoQ<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F.basis(),l); }

//	auto Etadag = F.Etadag(loc1y);
//	auto Eta = F.Eta(loc2y);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xSU2::Slabel;
//	if(loc1x == loc2x)
//	{
//		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
//		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else
//	{
//		Mout.setLocal({loc1x, loc2x}, {Etadag, Eta}, {{1,2},{1,-2}});
//		return Mout;
//	}
//}


















// MpoQ<SymSU2<double> > HubbardSU2xSU2::
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

// 	MpoQ<Symmetry> Mout(N_sites, N_legs, MpoQ<Symmetry>::qloc, qOptmp, {1}, HubbardSU2xSU2::Slabel, ss.str());
// 	Mout.setLocal({locx1,locx2}, {F.S(locy1),F.Sdag(locy2)});
// 	return Mout;
// }

// MpoQ<SymSU2<double> > HubbardSU2xSU2::
// triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
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
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2xSU2::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2xSU2::
// antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
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
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, qdiff, HubbardSU2xSU2::Nlabel, ss.str());
// }

// MpoQ<SymSU2<double> > HubbardSU2xSU2::
// quadruplon (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
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
	
// 	return MpoQ<Symmetry>(N_sites, N_legs, M, MpoQ<Symmetry>::qloc, {-2,-2}, HubbardSU2xSU2::Nlabel, ss.str());
// }

} //end namespace VMPS::models

#endif
