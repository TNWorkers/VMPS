#ifndef STRAWBERRY_HEISENBERGSU2
#define STRAWBERRY_HEISENBERGSU2

#include "symmetry/SU2.h"
#include "spins/BaseSU2.h"
#include "MpoQ.h"
#include "DmrgExternalQ.h"

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
	HeisenbergSU2 (size_t Lx_input, initializer_list<Param> params, size_t Ly_input=1);
	///\}

	/**
	   \param B : Base class from which the local operators are received
	   \param P : The parameters
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const spins::BaseSU2<> &B, const ParamHandler &P);

	/**Operator Quantum numbers: \f$\{ \left|1\right>, \left|3\right>\}\f$ */
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
	
private:
	const std::map<string,std::any> defaults = 
	{
		{"J",-1.}, {"Jprime",0.}, {"Jpara",0.}, {"Jperp",0.}, {"D",2ul}
	};

	spins::BaseSU2<> B;
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
HeisenbergSU2 (size_t Lx_input, initializer_list<Param> params, size_t Ly_input)
:MpoQ<Symmetry> (Lx_input, Ly_input, qarray<Symmetry::Nq>({0}), HeisenbergSU2::qOp(), HeisenbergSU2::Stotlabel, "", halve)
{
	ParamHandler P(params,defaults);
	B = spins::BaseSU2<>(N_legs,P.get<size_t>("D"));
	
	for (size_t l=0; l<N_sites; ++l) { setLocBasis(B.basis(),l); }
	
	HamiltonianTermsXd<Symmetry> Terms = set_operators(B,P);
	this->label = Terms.info;
	SuperMatrix<Symmetry,double> G = Generator(Terms);
	this->Daux = Terms.auxdim();
	
	this->construct(G, this->W, this->Gvec, false);	//false: For SU(2) symmetries the squared Hamiltonian can not be calculated in advance.
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
		Mout.setLocBasis(B.basis(),l);
	}

	Mout.label = ss.str();
	Mout.setQtarget(Symmetry::qvacuum());
	Mout.qlabel = HeisenbergSU2::Stotlabel;
	if(locx1 == locx2)
	{
		auto product = std::sqrt(3.)*Operator::prod(B.Sdag(locy1),B.S(locy2),Symmetry::qvacuum());
		Mout.setLocal(locx1,product.plain<SparseMatrixType>());
		return Mout;
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.)*B.Sdag(locy1)).plain<SparseMatrixType>(), B.S(locy2).plain<SparseMatrixType>()});
		return Mout;
	}
}

HamiltonianTermsXd<Sym::SU2<double> > HeisenbergSU2::
set_operators (const spins::BaseSU2<> &B, const ParamHandler &P)
{
	HamiltonianTermsXd<Symmetry> Terms;
	frac S = frac(B.get_D()-1,2);
	stringstream ss;
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",", ",", "", "", "{", "}");

	// J-terms
	
	double J   = P.get_default<double>("J");
	MatrixXd Jpara  (B.orbitals(),B.orbitals()); Jpara.setZero();
	if (P.HAS("J"))
	{
		J = P.get<double>("J");
		Jpara.diagonal().setConstant(J);
		ss << "Heisenberg(S=" << S << ",J=" << J;
	}
	else if (P.HAS("Jpara"))
	{
		assert(B.orbitals() == Jpara.rows() and 
		       B.orbitals() == Jpara.cols());
		Jpara = P.get<MatrixXd>("Jpara");
		ss << "Heisenberg(S=" << S << ",J∥=" << Jpara.format(CommaInitFmt);
	}
	else
	{
		ss << "JustLocal(";
	}
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-std::sqrt(3)*Jpara(i,j),
											 B.Sdag(i).plain<SparseMatrixType>(),
											 B.S(j).plain<SparseMatrixType>()));
		}
	}

	// J'-terms

	double Jprime = P.get_default<double>("Jprime");

	if (P.HAS("Jprime"))
	{
		Jprime = P.get<double>("Jprime");
		assert((B.orbitals() == 1 or Jprime == 0) and "Cannot interpret Ly>1 and J'!=0");
		ss << ",J'=" << Jprime;
	}
	if(Jprime != 0)
	{
		Terms.nextn.push_back(make_tuple(-std::sqrt(3)*Jprime, B.S(0).plain<SparseMatrixType>(),
										 B.S(0).plain<SparseMatrixType>(),
										 B.Id().plain<SparseMatrixType>()));
	}
	
	// local terms

	double Jperp   = P.get_default<double>("Jperp");

	if (P.HAS("Jperp"))
	{
		Jperp = P.get<double>("Jperp");
		ss << ",J⟂=" << Jperp;
	}

	ss << ")";
	Terms.info = ss.str();

	if( B.orbitals() > 1 )
	{
		Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jperp).plain<SparseMatrixType>()));
	}

	return Terms;
}

} //end namespace VMPS

#endif
