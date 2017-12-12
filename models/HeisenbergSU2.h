#ifndef STRAWBERRY_HEISENBERGSU2
#define STRAWBERRY_HEISENBERGSU2

#include "symmetry/SU2.h"
#include "spins/BaseSU2.h"
#include "MpoQ.h"
#include "DmrgExternalQ.h"
#include "ParamHandler.h"

namespace VMPS
{

/** \class HeisenbergSU2
  * \ingroup Models
  *
  * \brief Heisenberg Model
  *
  * MPO representation of 
  \f[
  H = -J \sum_{<ij>} \left(\mathbf{S_i}\mathbf{S_j}\right) -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right)
  \f]
  *
  \note Take use of the Spin SU(2) symmetry, which implies no magnetic fields. For using B-fields see VMPS::HeisenbergU1 or VMPS::Heisenberg.
  \note The default variable settings can be seen in \p HeisenbergSU2::defaults.
  \note \f$J<0\f$ is antiferromagnetic
  */
class HeisenbergSU2 : public MpoQ<Sym::SU2<double>,double>
{
public:
	typedef Sym::SU2<double> Symmetry;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::SparseMatrix<double> SparseMatrixType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> Operator;
	
public:
	
	//---constructors---
	///\{
	/**Do nothing.*/
	HeisenbergSU2() : MpoQ<Symmetry>() {};
	
	/**
	   \param Lx_input : chain length
	   \describe_params
	   \param Ly_input : amount of legs in ladder
	*/
	HeisenbergSU2 (variant<size_t,std::array<size_t,2> >, vector<Param> params);
	///\}
	
	/**
	   \param B : Base class from which the local operators are received
	   \param P : The parameters
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseSU2<> &B, const ParamHandler &P, size_t loc=0);
	
	/**Operator Quantum numbers: \f$\{ Id:k=\left|1\right>; S:k=\left|3\right>\}\f$ */
	static const std::vector<qType> qOp ();
	
	/**Labels the conserved quantum number as "S".*/
	static const std::array<string,1> Stotlabel;
	
	///@{
	/**Typedef for convenient reference (no need to specify \p Symmetry, \p Scalar all the time).*/
	typedef MpsQ<Symmetry,double>                                StateXd;
	typedef MpsQ<Symmetry,complex<double> >                      StateXcd;
	typedef DmrgSolverQ<Symmetry,HeisenbergSU2,double>           Solver;
	typedef MpsQCompressor<Symmetry,double,double>               CompressorXd;
	typedef MpsQCompressor<Symmetry,complex<double>,double>      CompressorXcd;
	typedef MpoQ<Symmetry,double>                                MpOperator;
	///@}
	
	///@{
	/**Observables.*/
	MpoQ<Symmetry,double> SS (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);	
	///@}
	
	/**Validates whether a given total quantum number \p qnum is a possible target quantum number for an MpsQ.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<1> qnum) const;
	
protected:
	
	const std::map<string,std::any> defaults = 
	{
		{"J",-1.}, {"Jprime",0.}, {"Jperp",0.}, {"D",2ul},
		{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}
	};
	
	vector<spins::BaseSU2<> > B;
};

const std::array<string,1> HeisenbergSU2::Stotlabel{"S"};

const std::vector<Sym::SU2<double>::qType> HeisenbergSU2::
qOp ()
{	
	std::vector<qType> vout(2);
	vout[0] = {1};
	vout[1] = {3};
	return vout;
};

HeisenbergSU2::
HeisenbergSU2 (variant<size_t,std::array<size_t,2> > L, vector<Param> params)
:MpoQ<Symmetry> (holds_alternative<size_t>(L)? get<0>(L):get<1>(L)[0], 
                 holds_alternative<size_t>(L)? 1        :get<1>(L)[1], 
                 qarray<Symmetry::Nq>({1}), HeisenbergSU2::qOp(), HeisenbergSU2::Stotlabel, "", halve)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		B[l] = spins::BaseSU2<>(N_legs,P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, false, P.get<bool>("OPEN_BC"));
	//false: For SU(2) symmetries the squared Hamiltonian can not be calculated in advance.
}

MpoQ<Sym::SU2<double> > HeisenbergSU2::
SS (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(N_sites, N_legs);
	for (std::size_t l=0; l<N_sites; l++)
	{
		Mout.setLocBasis(B[l].get_basis(),l);
	}
	
	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HeisenbergSU2::Stotlabel;
	if(locx1 == locx2)
	{
		auto product = std::sqrt(3.)*Operator::prod(B[locx1].Sdag(locy1),B[locx2].S(locy2),Symmetry::qvacuum());
		Mout.setLocal(locx1,product.plain<double>());
		return Mout;
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.)*B[locx1].Sdag(locy1)).plain<double>(), B[locx2].S(locy2).plain<double>()});
		return Mout;
	}
}

bool HeisenbergSU2::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0]-1,2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if(Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

HamiltonianTermsXd<Sym::SU2<double> > HeisenbergSU2::
set_operators (const spins::BaseSU2<> &B, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	frac S = frac(B.get_D()-1,2);
	stringstream ss;
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");
	
	Terms.name = "Heisenberg";
	
	// J-terms
	
	double J = P.get_default<double>("J");
	MatrixXd Jpara  (B.orbitals(),B.orbitals()); Jpara.setZero();
	if (P.HAS("J",loc))
	{
		J = P.get<double>("J",loc);
		Jpara.diagonal().setConstant(J);
		ss << "S=" << print_frac_nice(S) << ",J=" << J;
	}
	else if (P.HAS("Jpara",loc))
	{
		assert(B.orbitals() == Jpara.rows() and 
		       B.orbitals() == Jpara.cols());
		Jpara = P.get<MatrixXd>("Jpara",loc);
		ss << "S=" << print_frac_nice(S) << ",J∥=" << Jpara.format(CommaInitFmt);
	}
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-std::sqrt(3)*Jpara(i,j),
			                                 B.Sdag(i).plain<double>(),
			                                 B.S(j).plain<double>()));
		}
	}
	
	// J'-terms
	
	double Jprime = P.get_default<double>("Jprime");
	
	if (P.HAS("Jprime",loc))
	{
		Jprime = P.get<double>("Jprime",loc);
		assert((B.orbitals() == 1 or Jprime == 0) and "Cannot interpret Ly>1 and J'!=0");
		ss << ",J'=" << Jprime;
	}
	if (Jprime != 0)
	{
		Terms.nextn.push_back(make_tuple(-std::sqrt(3)*Jprime, B.Sdag(0).plain<double>(),
		                                 B.S(0).plain<double>(),
		                                 B.Id().plain<double>()));
	}
	
	// local terms
	
	double Jperp = P.get_default<double>("Jperp");
	
	if (P.HAS("J",loc))
	{
		Jperp  = P.get<double>("J",loc);
	}
	else if (P.HAS("Jperp",loc))
	{
		Jperp = P.get<double>("Jperp",loc);
		ss << ",J⟂=" << Jperp;
	}
	
	Terms.info = ss.str();
	
	if (B.orbitals() > 1)
	{
		Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jperp).plain<double>()));
	}
	
	return Terms;
}

} //end namespace VMPS

#endif
