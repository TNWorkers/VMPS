#ifndef STRAWBERRY_TRANSVERSEKONDOMODEL
#define STRAWBERRY_TRANSVERSEKONDOMODEL

#include "models/KondoU1xU1.h"

namespace VMPS
{
/** \class KondoU1
 * \ingroup Models
 *
 * \brief Kondo Model
 *
 * MPO representation of 
 \f$
 H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
 - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i - \sum_{i \in I} B_i^z S_i^z - \sum_{i \in I} B_i^x S_i^x
 \f$.
 *
 \note Take use of the U(1) particle conservation symmetry.
 \note The \f$S_z\f$ U(1) symmetry is borken due to the field in x-direction.
 \note The default variable settings can be seen in \p KondoU1::defaults.
 \note \f$J<0\f$ is antiferromagnetic
 \note If nnn-hopping is positive, the GS-energy is lowered.
 \todo Most of the observalbes need to be adjusted properly.
*/
class KondoU1 : public MpoQ<Sym::U1<double>,double>
{
public:
	typedef Sym::U1<double> Symmetry;
private:
	typedef typename Symmetry::qType qType;
public:
	/**Does nothing.*/
	KondoU1 () : MpoQ(){};
	KondoU1 (variant<size_t,std::array<size_t,2> > L, vector<Param> params);

	
	/**Labels the conserved quantum number as "N".*/
	static const std::array<string,1> Nlabel;	

	template<typename Symmetry_>
	static HamiltonianTermsXd<Symmetry_> add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const SpinBase<Symmetry_> &B, const FermionBase<Symmetry_> &F,
														const ParamHandler &P, size_t loc=0);

	///@{
	/**Typedef for convenient reference (no need to specify \p Nq, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                           StateXd;
	typedef MpsQ<Symmetry,complex<double> >                 StateXcd;
	typedef DmrgSolverQ<Symmetry,KondoU1>                   Solver;
	typedef MpsQCompressor<Symmetry,double,double>          CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double> CompressorXcd;
	typedef MpoQ<Symmetry>                                  Operator;
	///@}
	
	///@{
	/**Operator for the impurity spin.*/
	Operator Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0);
	
	/**Operator for the substrate spin.*/
	Operator Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy=0);
	
	/**Operator for impurity-substrate correlations.*/
	Operator SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	/**Operator for impurity-impurity correlations.*/
	Operator SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	/**Operator for substrate-substrate correlations.*/
	Operator SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	/***/
	Operator hopping (size_t locx, size_t locy=0);
	///@}
	
private:

	const std::map<string,std::any> defaults = 
	{
		{"J",-1.}, {"U",0.}, {"V",0.}, {"mu",0.},
		{"t",1.}, {"tpara",0.}, {"tperp",0.},{"tprime",0.},
		{"D",2ul}, {"K",0.},
		{"Bz",0.}, {"Bz_elec",0.}, {"Bx",1.}, {"Bx_elec",0.},
		{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
	};

	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::array<string,1> KondoU1::Nlabel{"N"};

KondoU1::
KondoU1 (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
	:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0],
					 holds_alternative<size_t>(L)? 1        :get<1>(L)[1],
					 qarray<Symmetry::Nq>({0}), KondoU1::Nlabel, "")
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);	F.resize(N_sites);

	for (size_t l=0; l<N_sites; ++l)
	{
		F[l] = FermionBase<Symmetry>(N_legs,!isfinite(P.get<double>("U",l%Lcell)));
		B[l] = SpinBase<Symmetry>(N_legs,P.get<size_t>("D",l%Lcell),true); //true means N is good quantum number
		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
		
		Terms[l] = KondoU1xU1::set_operators(B[l],F[l],P,l%Lcell);
		add_operators(Terms[l],B[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l])); // boost::multi_array has stupid assignment
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

// MpoQ<1> KondoU1::
// Simp (SPINOP_LABEL Sa, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << Sa << "_imp(" << locx << "," << locy << ")";
// 	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	Mout.setLocal(locx, kroneckerProduct(S.Scomp(Sa,locy),IdSub));
// 	return Mout;
// }

// MpoQ<1> KondoU1::
// Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<N_legs);
// 	stringstream ss;
// 	ss << Sa << "_sub(" << locx << "," << locy << ")";
// 	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
// 	MatrixXd IdImp(qloc[locx].size()/F.dim(), qloc[locx].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal(locx, kroneckerProduct(IdImp, F.Scomp(Sa,locy)));
// 	return Mout;
// }

// MpoQ<1> KondoU1::
// SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
// 	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(MpoQ<1>::qloc[locx2].size()/F.dim(), MpoQ<1>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1,locx2}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
// 	                              kroneckerProduct(IdImp,F.Scomp(SOP2,locy2))}
// 	             );
// 	return Mout;
// }

// MpoQ<1> KondoU1::
// SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
// 	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	Mout.setLocal({locx1,locx2}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
// 	                              kroneckerProduct(S.Scomp(SOP2,locy2),IdSub)}
// 	             );
// 	return Mout;
// }

// MpoQ<1> KondoU1::
// SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<N_sites and locx2<N_sites and locy1<N_legs and locy2<N_legs);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")";
// 	MpoQ<1> Mout(N_sites, N_legs, locBasis(), {0}, Nlabel, ss.str());
// 	MatrixXd IdImp1(MpoQ<1>::qloc[locx1].size()/F.dim(), MpoQ<1>::qloc[locx1].size()/F.dim()); IdImp1.setIdentity();
// 	MatrixXd IdImp2(MpoQ<1>::qloc[locx2].size()/F.dim(), MpoQ<1>::qloc[locx2].size()/F.dim()); IdImp2.setIdentity();
// 	Mout.setLocal({locx1,locx2}, {kroneckerProduct(IdImp1,F.Scomp(SOP1,locy1)), 
// 	                              kroneckerProduct(IdImp2,F.Scomp(SOP2,locy2))}
// 	             );
// 	return Mout;
// }

template<typename Symmetry_>
HamiltonianTermsXd<Symmetry_> KondoU1::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const SpinBase<Symmetry_> &B, const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};

	// Bx electronic sites
	auto [Bx_elec,Bx_elecorb,Bx_eleclabel] = P.fill_array1d<double>("Bx_elec","Bx_elecorb",F.orbitals(),loc);
	if(!Bx_eleclabel.empty()) {Terms.info.push_back(Bx_eleclabel);}

	// Bx spins
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F.orbitals(),loc);
	if(!Bxlabel.empty()) {Terms.info.push_back(Bxlabel);}

	ArrayXd zeros(F.orbitals()); zeros = 0.;
	auto H_Bxspins = kroneckerProduct(B.HeisenbergHamiltonian(0.,0.,zeros,Bxorb,zeros),F.Id());
	auto H_Bxelec = kroneckerProduct(B.Id(),F.HubbardHamiltonian(zeros,zeros,zeros,Bx_elecorb,0.,0.,0., P.get<bool>("CYLINDER")));
	auto H_Bx = H_Bxspins + H_Bxelec;

	Terms.local.push_back(make_tuple(1., H_Bx));

	Terms.name = "Transverse Kondo";

	return Terms;
}

};

#endif
