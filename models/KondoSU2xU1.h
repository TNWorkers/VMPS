#ifndef KONDOMODELSU2XU1_H_
#define KONDOMODELSU2XU1_H_

#include "spins/BaseSU2xU1.h"
#include "fermions/BaseSU2xU1.h"
#include "MpoQ.h"

namespace VMPS
{
/** \class KondoSU2xU1
  * \ingroup Kondo
  *
  * \brief Kondo Model
  *
  * MPO representation of
  \f[
  H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
  \f].
  *
   where further parameters from HubbardSU2xU1 and HeisenbergSU2 are possible.
  \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
  \note If the nnn-hopping is positive, the ground state energy is lowered.
  \warning \f$J<0\f$ is antiferromagnetic
  */
class KondoSU2xU1 : public MpoQ<Sym::SU2xU1<double>,double>
{
public:
	typedef Sym::SU2xU1<double> Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	///@{
	KondoSU2xU1 ():MpoQ() {};
	KondoSU2xU1 (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params);
	///@}

	/**
	   \param B : Base class from which the local spin-operators are received
	   \param F : Base class from which the local fermion-operators are received
	   \param P : The parameters
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseSU2xU1<> &B, const fermions::BaseSU2xU1<> &F,
	                                                    const ParamHandler &P, size_t loc=0);

	/**Makes half-integers in the output for the magnetization quantum number.*/
	static std::string N_halveM (qType qnum);
	
	/**Labels the conserved quantum numbers as "N", "M".*/
	static const std::array<std::string,2> SNlabel;
			
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;

	///@{
	// MpoQ<Symmetry> n (std::size_t locx, std::size_t locy=0);
	// MpoQ<Symmetry> d (std::size_t locx, std::size_t locy=0);
	// MpoQ<Symmetry> ninj (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	// MpoQ<Symmetry> c (std::size_t locx, std::size_t locy=0);
	// MpoQ<Symmetry> cdag (std::size_t locx, std::size_t locy=0);
	// MpoQ<Symmetry> cdagc (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	// MpoQ<Symmetry> EtaEtadag (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	
	// MpoQ<Symmetry> SimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	// MpoQ<Symmetry> SsubSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	// MpoQ<Symmetry> SimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y=0, std::size_t loc2y=0);
	// MpoQ<Symmetry> SimpSsubSimpSimp (std::size_t loc1x, std::size_t loc2x,
	// 								 std::size_t loc3x, std::size_t loc4x,
	// 								 std::size_t loc1y=0, std::size_t loc2y=0, std::size_t loc3y=0, std::size_t loc4y=0);
	// MpoQ<Symmetry> SimpSsubSimpSsub (std::size_t loc1x, std::size_t loc2x,
	// 								 std::size_t loc3x, std::size_t loc4x,
	// 								 std::size_t loc1y=0, std::size_t loc2y=0, std::size_t loc3y=0, std::size_t loc4y=0);
	///@}

	static const std::map<string,std::any> defaults;

protected:

	vector<fermions::BaseSU2xU1<> > F;
	vector<spins::BaseSU2xU1<> > B;
};

const std::map<string,std::any> KondoSU2xU1::defaults =
{
	{"t",1.}, {"tPerp",0.},{"tPrime",0.},
	{"J",-1.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"mu",0.}, {"t0",0.},
	{"D",2ul},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}
};

const std::array<std::string,Sym::SU2xU1<double>::Nq> KondoSU2xU1::SNlabel{"S","N"};

KondoSU2xU1::
KondoSU2xU1 (const variant<size_t,std::array<size_t,2> > &L, const vector<Param> &params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0],
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1],
                 qarray<Symmetry::Nq>({1,0}), KondoSU2xU1::SNlabel, "")
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites); F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		F[l] = fermions::BaseSU2xU1<>(N_legs, !isfinite(P.get<double>("U",l%Lcell))); //true means basis n,m
		B[l] = spins::BaseSU2xU1<>(N_legs, P.get<size_t>("D",l%Lcell));
		
		setLocBasis(B[l].get_basis().combine(F[l].get_basis()),l);

		Terms[l] = set_operators(B[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, false, P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

bool KondoSU2xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[1],2); //electrons have spin 1/2
	frac Smax = S_elec;
	for (size_t l=0; l<N_sites; ++l) { Smax+=B[l].orbitals()*frac(B[l].get_D()-1,2); } //add local spins to Smax
	
	frac S_tot(qnum[0]-1,2);
	cout << S_tot << "\t" << Smax << endl;
	if (Smax.denominator()==S_tot.denominator() and S_tot<=Smax and qnum[0]<=2*static_cast<int>(this->N_sites*this->N_legs) and qnum[0]>0) {return true;}
	else {return false;}
}

std::string KondoSU2xU1::
N_halveM (qType qnum)
{
	std::stringstream ss;
	ss << "not implemented";
	return ss.str();
}

HamiltonianTermsXd<Sym::SU2xU1<double> > KondoSU2xU1::
set_operators (const spins::BaseSU2xU1<> &B, const fermions::BaseSU2xU1<> &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;

	frac S = frac(B.get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};

	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F.orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F.orbitals(),loc);
	save_label(Vlabel);
	for (int i=0; i<F.orbitals(); ++i)
	for (int j=0; j<F.orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			auto Otmp = OperatorType::prod(OperatorType::outerprod(B.Id(),F.sign(),{1,0}),OperatorType::outerprod(B.Id(),F.c(j),{2,-1}),{2,-1});
			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
											 OperatorType::outerprod(B.Id(),F.cdag(i),{2,+1}).plain<double>(),
											 Otmp.plain<double>()));
			Otmp = OperatorType::prod(OperatorType::outerprod(B.Id(),F.sign(),{1,0}),OperatorType::outerprod(B.Id(),F.cdag(j),{2,+1}),{2,+1});
			Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
											 OperatorType::outerprod(B.Id(),F.c(i),{2,-1}).plain<double>(),
											 Otmp.plain<double>()));
		}
		
		if (Vpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Vpara(i,j),
											 OperatorType::outerprod(B.Id(),F.n(i),{1,0}).plain<double>(),
											 OperatorType::outerprod(B.Id(),F.n(j),{1,0}).plain<double>()));
		}		
	}

	// NNN terms
	
	param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	save_label(tPrime.label);

	if (tPrime.x != 0.)
	{
		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");

		auto Otmp = OperatorType::prod(F.sign(),F.c(),{2,-1});
		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.),
										 OperatorType::outerprod(B.Id(),F.cdag(),{2,+1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),Otmp,{2,-1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),F.sign(),{1,0}).plain<double>()));
		Otmp = OperatorType::prod(F.sign(),F.cdag(),{2,+1});
		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.),
										 OperatorType::outerprod(B.Id(),F.c(),{2,-1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),Otmp,{2,+1}).plain<double>(),
										 OperatorType::outerprod(B.Id(),F.sign(),{1,0}).plain<double>()));
	}

	// local terms
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// V⟂
	param0d Vperp = P.fill_array0d<double>("Vperp","Vperp",loc);
	save_label(Vperp.label);
	
	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F.orbitals(),loc);
	save_label(Ulabel);
	
	// mu
	auto [mu,muorb,mulabel] = P.fill_array1d<double>("mu","muorb",F.orbitals(),loc);
	save_label(mulabel);
	
	// t0
	auto [t0,t0orb,t0label] = P.fill_array1d<double>("t0","t0orb",F.orbitals(),loc);
	save_label(t0label);

	OperatorType KondoHamiltonian({1,0},B.get_basis().combine(F.get_basis()));

	//set Hubbard part of Kondo Hamiltonian
	KondoHamiltonian = OperatorType::outerprod(B.Id(),F.HubbardHamiltonian(Uorb,t0orb-muorb,tPerp.x,Vperp.x,0., P.get<bool>("CYLINDER")),{1,0});

	//set Heisenberg part of Hamiltonian
	KondoHamiltonian += OperatorType::outerprod(B.HeisenbergHamiltonian(0.,P.get<bool>("CYLINDER")),F.Id(),{1,0});

	// Kondo-J
	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F.orbitals(),loc);
	save_label(Jlabel);

	//set interaction part of Hamiltonian.
	for (int i=0; i<F.orbitals(); ++i)
	{
		KondoHamiltonian += -Jorb(i)*std::sqrt(3.)*OperatorType::outerprod(B.Sdag(i),F.S(i),{1,0});
	}

	Terms.name = "Kondo SU(2)⊗U(1)";
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
	
	return Terms;
}

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// d (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "double_occ(" << locx << "," << locy << ")";
	
// 	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	if (auto it=std::find(imploc.begin(),imploc.end(),locx) == imploc.end())
// 	{
// 		Mout.label = ss.str();
// 		Mout.setQtarget(Symmetry::qvacuum());
// 		Mout.qlabel = HubbardSU2xU1::Slabel;
// 		Mout.setLocal(locx, F.d(locy), Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	auto d = Operator::outerprod(Spins.Id(),F.d(locy),Symmetry::qvacuum());
// 	Mout.setLocal({locx}, {d}, {Symmetry::qvacuum()});
// 	return Mout;
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// n (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "occ(" << locx << "," << locy << ")";
	
// 	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	if (auto it=std::find(imploc.begin(),imploc.end(),locx) == imploc.end())
// 	{
// 		Mout.label = ss.str();
// 		Mout.setQtarget(Symmetry::qvacuum());
// 		Mout.qlabel = HubbardSU2xU1::Slabel;
// 		Mout.setLocal(locx, F.n(locy), Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	auto n = Operator::outerprod(Spins.Id(),F.n(locy),Symmetry::qvacuum());
// 	Mout.setLocal({locx}, {n}, {Symmetry::qvacuum()});
// 	return Mout;
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// ninj (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "n(" << loc1x << "," << loc1y << ")"  << "n(" << loc2x << "," << loc2y << ")";
	
// 	MpoQ<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto n = Operator::outerprod(Spins.Id(),F.n(loc2y),Symmetry::qvacuum());
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = Operator::prod(n,n,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({loc1x, loc2x}, {n, n}, {Symmetry::qvacuum(),Symmetry::qvacuum()});
// 		return Mout;
// 	}

// 	return Mout;
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// c (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "c(" << locx << "," << locy << ")";

// 	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	Mout.label = ss.str();
// 	Mout.setQtarget({2,-1});
// 	Mout.qlabel = KondoSU2xU1::SNlabel;

// 	Mout.setLocal(locx, Operator::outerprod(Spins.Id(),F.c(locy),{2,-1}), {2,-1}, Operator::outerprod(Spins.Id(),F.sign(),{1,0}));
// 	return Mout;
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// cdag (std::size_t locx, std::size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	std::stringstream ss;
// 	ss << "c†(" << locx << "," << locy << ")";

// 	MpoQ<Sym::SU2xU1<double> > Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	Mout.label = ss.str();
// 	Mout.setQtarget({2,+1});
// 	Mout.qlabel = KondoSU2xU1::SNlabel;

// 	Mout.setLocal(locx, Operator::outerprod(Spins.Id(),F.cdag(locy),{2,+1}), {2,+1}, Operator::outerprod(Spins.Id(),F.sign(),{1,0}));
// 	return Mout;
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// cdagc (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "c†(" << loc1x << "," << loc1y << ")" << "c(" << loc2x << "," << loc2y << ")";

// 	MpoQ<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto cdag = Operator::outerprod(Spins.Id(),F.cdag(loc1y),{2,+1});
// 	auto c = Operator::outerprod(Spins.Id(),F.c(loc2y),{2,-1});
// 	auto sign = Operator::outerprod(Spins.Id(),F.sign(),{1,0});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = std::sqrt(2.)*Operator::prod(cdag,c,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else if(loc1x<loc2x)
// 	{

// 		Mout.setLocal({loc1x, loc2x}, {std::sqrt(2.)*cdag, Operator::prod(sign,c,{2,-1})}, {{2,1},{2,-1}}, sign);
// 		return Mout;
// 	}
// 	else if(loc1x>loc2x)
// 	{

// 		Mout.setLocal({loc1x, loc2x}, {std::sqrt(2.)*Operator::prod(sign,cdag,{2,+1}), c}, {{2,1},{2,-1}}, sign);
// 		return Mout;
// 	}
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// EtaEtadag (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "η†(" << loc1x << "," << loc1y << ")" << "η(" << loc2x << "," << loc2y << ")";

// 	MpoQ<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Etadag = Operator::outerprod(Spins.Id(),F.Etadag(loc1y),{1,2});
// 	auto Eta = Operator::outerprod(Spins.Id(),F.Eta(loc2y),{1,-2});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({loc1x, loc2x}, {Etadag, Eta}, {{1,2},{1,-2}});
// 		return Mout;
// 	}
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// SimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "S(" << loc1x << "," << loc1y << ")" << "S(" << loc2x << "," << loc2y << ")";

// 	MpoQ<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Sdag = Operator::outerprod(Spins.Sdag(loc1y),F.Id(),{3,0});
// 	auto S = Operator::outerprod(Spins.S(loc2y),F.Id(),{3,0});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = std::sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({loc1x, loc2x}, {std::sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
// 		return Mout;
// 	}
// }


// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// SsubSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "s(" << loc1x << "," << loc1y << ")" << "s(" << loc2x << "," << loc2y << ")";

// 	MpoQ<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Sdag = Operator::outerprod(Spins.Id(),F.Sdag(loc1y),{3,0});
// 	auto S = Operator::outerprod(Spins.Id(),F.S(loc2y),{3,0});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = std::sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({loc1x, loc2x}, {std::sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
// 		return Mout;
// 	}
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// SimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc1y, std::size_t loc2y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites);
// 	std::stringstream ss;
// 	ss << "S(" << loc1x << "," << loc1y << ")" << "s(" << loc2x << "," << loc2y << ")";

// 	MpoQ<Symmetry> Mout(N_sites, N_legs);
// 	for(std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Sdag = Operator::outerprod(Spins.Sdag(loc1y),F.Id(),{3,0});
// 	auto S = Operator::outerprod(Spins.Id(),F.S(loc2y),{3,0});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(loc1x == loc2x)
// 	{
// 		auto product = std::sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
// 		Mout.setLocal(loc1x,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({loc1x, loc2x}, {std::sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
// 		return Mout;
// 	}
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// SimpSsubSimpSimp (std::size_t loc1x, std::size_t loc2x, std::size_t loc3x, std::size_t loc4x,
//                   std::size_t loc1y, std::size_t loc2y, std::size_t loc3y, std::size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/F.dim(), MpoQ<2>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(S.Scomp(SOP4,loc4y),IdSub)});
// 	return Mout;
// }

// MpoQ<Sym::SU2xU1<double> > KondoSU2xU1::
// SimpSsubSimpSsub (std::size_t loc1x, std::size_t loc2x, std::size_t loc3x, std::size_t loc4x,
// 				  std::size_t loc1y, std::size_t loc2y, std::size_t loc3y, std::size_t loc4y)
// {
// 	assert(loc1x<this->N_sites and loc2x<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << loc1x << "," << loc1y << ")" << SOP2 << "(" << loc2x << "," << loc2y << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(MpoQ<2>::qloc[loc2x].size()/F.dim(), MpoQ<2>::qloc[loc2x].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({loc1x, loc2x, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,loc1y),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,loc2y)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP4,loc4y))}
// 		);
// 	return Mout;
// }


// bool KondoSU2xU1::
// validate (qType qnum) const
// {
// 	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
// 	return (qnum[0]-1+N_legs*Sx2*imploc.size())%2 == qnum[1]%2;
// }

} //end namespace VMPS

#endif
